import pickle
from collections import defaultdict

import yaml
import pytorch_lightning as pl
import numpy as np
import math
import torch
import torchmetrics
import transformers
from pandas import DataFrame

from torch import nn, Tensor
from torch.nn import Linear
from torch.types import Device
from transformers import PreTrainedTokenizer, BatchEncoding, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from kplmb.constants import EntityMarker
from kplmb.utils import secure_two_dimensions
from molbert.models.smiles import SmilesMolbertModel


class EmbeddingConfiguration:

    def __init__(
            self,
            id: str,
            target: str,
            embeddings: Tensor,
            entity_id_to_embedding_index: Dict[str, int],
            hidden_size: int,
            output_size: int,
            hidden_dropout: float,
            output_dropout: float,
            freeze_embeddings: bool
    ):
        self.id = id
        self.target = target
        self.embeddings = embeddings
        self.entity_id_to_embedding_index = entity_id_to_embedding_index
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_dropout = hidden_dropout
        self.output_dropout = output_dropout
        self.freeze_embeddings = freeze_embeddings


def build_embedding_module(
        embedding_config: EmbeddingConfiguration,
        size_factor: int = 1
) -> Tuple[nn.Embedding, nn.Sequential]:
    entity_embeddings = nn.Embedding(embedding_config.embeddings.shape[0] + 1, embedding_config.embeddings.shape[1])
    with torch.no_grad():
        entity_embeddings.weight[0, :] = 0
        entity_embeddings.weight[1:] = nn.Parameter(embedding_config.embeddings)

    entity_embeddings.requires_grad = not embedding_config.freeze_embeddings
    entity_embeddings.weight.requires_grad = not embedding_config.freeze_embeddings

    if embedding_config.hidden_size > 0:
        entity_mlp = nn.Sequential(
            nn.Linear(entity_embeddings.embedding_dim * size_factor, embedding_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(embedding_config.hidden_dropout),
            nn.Linear(embedding_config.hidden_size, embedding_config.output_size),
            nn.Dropout(embedding_config.output_dropout)
        )
    else:
        entity_mlp = nn.Sequential(
            nn.Linear(entity_embeddings.embedding_dim * size_factor, embedding_config.output_size),
            nn.ReLU(),
            nn.Dropout(embedding_config.output_dropout)
        )

    for module in entity_mlp._modules:
        if isinstance(module, Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity="relu")

    return entity_embeddings, entity_mlp


def predict_molbert(molbert_model: SmilesMolbertModel, input_ids: Tensor, valid: Tensor, device: Device) -> Tensor:
    input_cpu = input_ids.cpu()
    token_type_ids = np.zeros_like(input_cpu, dtype=np.int32)
    attention_mask = np.zeros_like(input_cpu, dtype=np.int32)

    attention_mask[input_cpu != 0] = 1

    input_ids = torch.tensor(input_cpu, dtype=torch.long, device=device).squeeze()
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device).squeeze()
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device).squeeze()

    if len(input_ids.shape) == 1:
        input_ids = input_ids.view(1, -1)
        token_type_ids = token_type_ids.view(1, -1)
        attention_mask = attention_mask.view(1, -1)

    outputs = molbert_model.model.bert(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )

    #sequence_output, pooled_output = outputs
    sequence_output = outputs["last_hidden_state"]
    pooled_output = outputs["pooler_output"]

    # set invalid outputs to 0s
    valid_tensor = torch.tensor(
        valid,
        dtype=sequence_output.dtype,
        device=sequence_output.device,
        requires_grad=False
    ).squeeze()

    if len(valid_tensor.shape) == 0:
        valid_tensor = torch.tensor(
            [0],
            dtype=sequence_output.dtype,
            device=sequence_output.device,
            requires_grad=False
        )

    pooled_output = pooled_output * valid_tensor[:, None]

    return pooled_output


class MoleculeStructureEncoderConfiguration:

    def __init__(
            self,
            id: str,
            target: str,
            encoder_type: str,
            model_path: Path,
            entity_id_to_smiles: Dict,
            max_length: int,
            freeze_encoder: bool
    ):
        self.id = id
        self.target = target
        self.encoder_type = encoder_type
        self.model_path = model_path
        self.entity_id_to_smiles = entity_id_to_smiles
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder


def build_molecule_structure_module(config: MoleculeStructureEncoderConfiguration):
    if config.encoder_type == "molbert":
        hparams_path = config.model_path / "hparams.yaml"

        with open(hparams_path) as yaml_file:
            model_config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        #model_config = Namespace(**model_config_dict)
        structure_model = SmilesMolbertModel(model_config_dict)
        structure_model.load_from_checkpoint(
            config.model_path / "checkpoints" / "last.ckpt",
            hparam_overrides=structure_model.__dict__,
            args=model_config_dict
        )

        # HACK: manually load model weights since they don't seem to load from checkpoint (PL v.0.8.5)
        checkpoint = torch.load(config.model_path / "checkpoints" / "last.ckpt",
                                map_location=lambda storage, loc: storage)
        structure_model.load_state_dict(checkpoint["state_dict"])

        if config.freeze_encoder:
            structure_model.model.bert.encoder.requires_grad_(False)
            structure_model.model.bert.embeddings.requires_grad_(False)

    elif config.encoder_type == "chemberta":
        structure_model = RobertaModel.from_pretrained(config.model_path)

    else:
        raise NotImplementedError(f"Unsupported encoder: {config.encoder_type}")

    return structure_model


class RelationClassificationOutput(SequenceClassifierOutput):

    def __init__(
            self,
            document_ids: np.ndarray,
            head_ids: np.ndarray,
            tail_ids: np.ndarray,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.document_ids = document_ids
        self.head_ids = head_ids
        self.tail_ids = tail_ids


class BatchEncodingWithMetaData(BatchEncoding):

    def __init__(self, batch: BatchEncoding, meta_information: Dict):
        super().__init__()
        self.data = batch.data
        self._encodings = batch._encodings
        self.meta_information = meta_information

    def __getstate__(self):
        return {
            "data": self.data,
            "encodings": self._encodings,
            "meta_information": self.meta_information
        }

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

        if "meta_information" in state:
            self.meta_information = state["meta_information"]


class RelationClassificationTransformer(pl.LightningModule):
    def __init__(
        self,
        id: str,
        transformer: str,
        tokenizer: PreTrainedTokenizer,
        num_relation_types: int,
        lr: float,
        max_length: int,
        optimized_metric: str,
        use_cls_token: bool,
        use_start_tokens: bool,
        use_end_tokens: bool,
        entity_marker: EntityMarker,
        input_prompt: str,
        add_text_prompt: bool,
        num_context_sentences: int,
        blind_entities: bool,
        use_norel_class: bool,
        add_info_lr: Optional[float] = None,
        weight_decay=0.0,
        use_xavier_init: bool = False,
        embedding_configurations: List[EmbeddingConfiguration] = None,
        structure_encoding_configurations: List[MoleculeStructureEncoderConfiguration] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.id = id
        self.weight_decay = weight_decay
        self.optimized_metric = optimized_metric
        self.entity_marker = entity_marker
        self.use_cls_token = use_cls_token
        self.use_start_tokens = use_start_tokens
        self.use_end_tokens = use_end_tokens
        self.max_length = max_length
        self.blind_entities = blind_entities
        self.use_norel_class = use_norel_class
        self.input_prompt = input_prompt
        self.add_text_prompt = add_text_prompt
        self.num_context_sentences = num_context_sentences

        embedding_configurations = embedding_configurations if embedding_configurations is not None else []
        structure_encoding_configurations = structure_encoding_configurations if structure_encoding_configurations is not None else []

        #self.use_none_class = use_none_class
        #self.blind_entities = blind_entities
        #self.max_length = max_length

        if self.entity_marker != EntityMarker.SPECIAL_TOKEN:
            assert (
                not self.use_start_tokens and not self.use_end_tokens
            ), "Starts and ends cannot be uniquely determined without additional special tokens"

        self.loss = nn.BCEWithLogitsLoss()
        self.tokenizer = tokenizer

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.transformer.resize_token_embeddings(len(self.tokenizer))

        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.num_labels = num_relation_types - (1 if self.use_norel_class else 0)

        pair_representation_size = 0
        if self.use_cls_token:
            pair_representation_size += self.transformer.config.hidden_size
        if self.use_start_tokens:
            pair_representation_size += 2 * self.transformer.config.hidden_size
        if self.use_end_tokens:
            pair_representation_size += 2 * self.transformer.config.hidden_size

        self.embedding_modules = []
        for emb_config in embedding_configurations:
            factor = 2 if emb_config.target == "both" else 1
            embeddings, mlp = build_embedding_module(emb_config, factor)

            self.__setattr__(f"{emb_config.id}_embeddings", embeddings)
            self.__setattr__(f"{emb_config.id}_mlp", mlp)

            self.embedding_modules.append((emb_config, embeddings, mlp))
            pair_representation_size += emb_config.output_size

        self.structure_encoders = []
        for str_encoder_config in structure_encoding_configurations:
            model = build_molecule_structure_module(str_encoder_config)
            self.structure_encoders.append((str_encoder_config, tokenizer, model))

            self.__setattr__(f"{str_encoder_config.id}_bert_model", model)

            factor = 2 if str_encoder_config.target == "both" else 1
            pair_representation_size += model.config.hidden_size * factor

        self.classifier = nn.Linear(pair_representation_size, num_relation_types)
        if use_xavier_init:
            torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.lr = lr
        self.add_info_lr = add_info_lr if add_info_lr else lr

        self.num_training_steps = None

        if self.num_labels > 1:
            self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels, average="micro")
            self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels, average="micro")
            self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_labels, average="micro")
            self.test_precision = torchmetrics.Precision(task="multilabel", num_labels=self.num_labels, average="micro")
            self.test_recall = torchmetrics.Recall(task="multilabel", num_labels=self.num_labels, average="micro")
        else:
            self.train_f1 = torchmetrics.F1Score(task="binary")
            self.val_f1 = torchmetrics.F1Score(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            self.test_precision = torchmetrics.Precision(task="binary")
            self.test_recall = torchmetrics.Recall(task="binary")

        for metric in [self.train_f1, self.val_f1, self.test_f1, self.test_precision, self.test_recall]:
            metric.to(self.device)

        self.val_outputs = []
        self.test_output = []
        self.test_metric = "test"

    def collate_fn(self, data):
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        batch = collator([entry["features"] for entry in data])

        meta_information = {}
        for meta_data in ["document_id", "head_id", "tail_id"]:
             meta_information[meta_data] = [entry[meta_data] for entry in data]

        batch_with_meta = BatchEncodingWithMetaData(batch, meta_information)

        return batch_with_meta

    def forward(self, batch):
        if "token_type_ids" in batch.data:
            output = self.transformer(
                input_ids=batch.data["input_ids"],
                token_type_ids=batch.data["token_type_ids"],
                attention_mask=batch.data["attention_mask"],
            )
        else:
            output = self.transformer(
                input_ids=batch.data["input_ids"],
                attention_mask=batch.data["attention_mask"],
            )

        sequence_embeddings = self.dropout(output.last_hidden_state)

        pair_representations = []

        if self.use_cls_token:
            pair_representations.append(sequence_embeddings[:, 0])

        if self.use_start_tokens:
            head_start_idx = torch.where(
                batch.data["input_ids"] == self.tokenizer.convert_tokens_to_ids(self.entity_marker.value[0])
            )
            head_start_rep = sequence_embeddings[head_start_idx]

            tail_start_idx = torch.where(
                batch.data["input_ids"] == self.tokenizer.convert_tokens_to_ids(self.entity_marker.value[2])
            )
            tail_start_rep = sequence_embeddings[tail_start_idx]

            start_pair_rep = torch.cat([head_start_rep, tail_start_rep], dim=1)
            pair_representations.append(start_pair_rep)

        if self.use_end_tokens:
            head_end_idx = torch.where(
                batch.data["input_ids"] == self.tokenizer.convert_tokens_to_ids(self.entity_marker.value[1])
            )
            head_end_rep = sequence_embeddings[head_end_idx]

            tail_end_idx = torch.where(
                batch.data["input_ids"] == self.tokenizer.convert_tokens_to_ids(self.entity_marker.value[3])
            )
            tail_end_rep = sequence_embeddings[tail_end_idx]

            end_pair_rep = torch.cat([head_end_rep, tail_end_rep], dim=1)
            pair_representations.append(end_pair_rep)

        pair_representations = torch.cat(pair_representations, dim=1)

        if len(self.embedding_modules) > 0:
            emb_representations = []
            for emb_conf, embeddings, mlp in self.embedding_modules:
                if emb_conf.target == "both":
                    head_embeddings = embeddings(batch.data[f"{emb_conf.id}_head_index"])
                    tail_embeddings = embeddings(batch.data[f"{emb_conf.id}_tail_index"])
                    pair_embeddings = torch.cat([head_embeddings, tail_embeddings], dim=1)
                    emb_representations.append(mlp(pair_embeddings))

                else:
                    entity_embeddings = embeddings(batch.data[f"{emb_conf.id}_{emb_conf.target}_index"])
                    emb_representations.append(mlp(entity_embeddings))

            pair_representations = torch.cat([pair_representations] + emb_representations, dim=1)

        if len(self.structure_encoders) > 0:
            str_representations = []
            for str_config, tokenizer, str_model in self.structure_encoders:
                if str_config.encoder_type == "molbert":
                    if str_config.target == "head" or str_config.target == "both":
                        head_str_embedding = predict_molbert(
                            molbert_model=str_model,
                            input_ids=batch.data[f"{str_config.id}_head_input_ids"],
                            valid=batch.data[f"{str_config.id}_head_valid"],
                            device=self.device
                        )
                        str_representations.append(secure_two_dimensions(head_str_embedding.squeeze()))

                    if str_config.target == "tail" or str_config.target == "both":
                        tail_str_embedding = predict_molbert(
                            molbert_model=str_model,
                            input_ids=batch.data[f"{str_config.id}_tail_input_ids"],
                            valid=batch.data[f"{str_config.id}_tail_valid"],
                            device=self.device
                        )
                        str_representations.append(secure_two_dimensions(tail_str_embedding.squeeze()))

                elif str_config.encoder_type == "chemberta":
                    if str_config.target == "head" or str_config.target == "both":
                        head_input_ids = secure_two_dimensions(
                            tensor=batch.data[f"{str_config.id}_head_input_ids"].squeeze()
                        )
                        head_attention_mask = secure_two_dimensions(
                            tensor=batch.data[f"{str_config.id}_head_attention_mask"].squeeze()
                        )

                        head_output = str_model(
                            input_ids=head_input_ids,
                            attention_mask=head_attention_mask,
                            return_dict=True
                        )
                        str_representations.append(secure_two_dimensions(head_output.pooler_output.squeeze()))

                    if str_config.target == "tail" or str_config.target == "both":
                        tail_input_ids = secure_two_dimensions(
                            tensor=batch.data[f"{str_config.id}_head_input_ids"].squeeze()
                        )
                        tail_attention_mask = secure_two_dimensions(
                            tensor=batch.data[f"{str_config.id}_head_attention_mask"].squeeze()
                        )

                        tail_output = str_model(
                            input_ids=tail_input_ids,
                            attention_mask=tail_attention_mask,
                            return_dict=True
                        )
                        str_representations.append(secure_two_dimensions(tail_output.pooler_output.squeeze()))

                else:
                    raise NotImplementedError(f"Unsupported encoder type: {str_config.encoder_type}")

            pair_representations = torch.cat([pair_representations] + str_representations, dim=1)

        logit_values = self.classifier(pair_representations)
        if "labels" in batch.data:
            loss = self.loss(logit_values, batch.data["labels"])
        else:
            loss = None

        return RelationClassificationOutput(
            document_ids=batch.meta_information["document_id"],
            head_ids=batch.meta_information["head_id"],
            tail_ids=batch.meta_information["tail_id"],
            loss=loss,
            logits=logit_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def on_train_start(self) -> None:
        self.best_val_f1 = 0.0

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)

        indicators = self.logits_to_indicators(output.logits).float()
        labels = batch["labels"].to(self.device).long()

        if self.use_norel_class:
           indicators = indicators[:, 1:]
           labels = labels[:, 1:]

        self.train_f1(indicators, labels)

        self.log("train/loss", output.loss, prog_bar=True)
        self.log(f"train/f1", self.train_f1, prog_bar=True)

        self.lr_schedulers().step()

        return output.loss

    def on_validation_start(self) -> None:
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)

        indicators = self.logits_to_indicators(output.logits).float()
        labels = batch["labels"].to(self.device).long()

        if self.use_norel_class:
            indicators = indicators[:, 1:]
            labels = labels[:, 1:]

        self.val_f1(indicators, labels)

        self.log("val/loss", output.loss, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=True)

        self.val_outputs.append((indicators, labels, output))

        return output.loss

    def on_validation_epoch_end(self):
         self.val_outputs.clear()

    def on_test_start(self) -> None:
        self.test_output = []

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        indicators = self.logits_to_indicators(output.logits).float()
        labels = batch["labels"].to(self.device).long()

        if self.use_norel_class:
            indicators = indicators[:, 1:]
            labels = labels[:, 1:]

        self.test_output.append((indicators, labels, output))

        return output.loss

    def on_test_epoch_end(self) -> None:
        indicators = torch.concat([indicators for (indicators, _,  _) in self.test_output], dim=0)
        labels = torch.concat([label for (_, label, _) in self.test_output], dim=0)

        epoche_precision = self.test_precision(indicators, labels)
        self.log(f"{self.test_metric}/precision", epoche_precision, prog_bar=False)

        epoche_recall = self.test_recall(indicators, labels)
        self.log(f"{self.test_metric}/recall", epoche_recall, prog_bar=False)

        epoche_f1 = self.test_f1(indicators, labels)
        self.log(f"{self.test_metric}/f1", epoche_f1, prog_bar=False)

        self.save_predictions(self.test_output, Path(f"{self.test_metric}.tsv"))
        self.save_logits(self.test_output, Path(f"{self.test_metric}.pkl"))

        self.test_output.clear()

    def configure_optimizers(self):
        assert self.num_training_steps > 0

        params = list(self.named_parameters())

        default_parameters = [p for n, p in params if n.startswith("classifier") or n.startswith("transformer")]
        additional_parameters = [p for n, p in params if not (n.startswith("classifier") or n.startswith("transformer"))]

        grouped_parameters = [
            {"params": default_parameters, 'lr': self.lr},
            {"params": additional_parameters, 'lr': self.add_info_lr},
        ]

        optimizer = torch.optim.Adam(
            params=grouped_parameters,
            lr=self.lr
            # weight_decay=self.weight_decay
        )

        schedule = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * self.num_training_steps,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [schedule]

    def logits_to_indicators(self, logits: torch.FloatTensor) -> torch.LongTensor:
        # if isinstance(self.loss, ATLoss):
        #     thresholds = logits[:, 0].unsqueeze(1)
        #     return logits > thresholds
        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits.to(self.device)) > 0.5
        else:
            raise ValueError

    def aggregate_predictions(self, output_data: List[Tuple]):
        indicators = torch.concat([indicators for (indicators, _,  _) in output_data], dim=0)
        labels = torch.concat([label for (_, label, _) in output_data], dim=0)

        doc_ids = np.concatenate([np.array(out[2].document_ids) for out in output_data], axis=0)
        head_ids = np.concatenate([np.array(out[2].head_ids) for out in output_data], axis=0)
        tails_ids = np.concatenate([np.array(out[2].tail_ids) for out in output_data], axis=0)

        records = []
        for tuple in zip(doc_ids, head_ids, tails_ids, indicators, labels):
            pred_labels = None
            pred_indices = (tuple[3] == 1.0).nonzero().squeeze().cpu().numpy()
            if pred_indices.size > 0:
                pred_indices = pred_indices if pred_indices.size > 1 else [pred_indices]
                pred_labels = ",".join([str(idx) for idx in pred_indices])

            gold_labels = None
            gold_indices = (tuple[4] == 1.0).nonzero().squeeze().cpu().numpy()
            if gold_indices.size > 0:
                gold_indices = gold_indices if gold_indices.size > 1 else [gold_indices]
                gold_labels = ",".join([str(idx) for idx in gold_indices])

            records.append(list(tuple) + [pred_labels, gold_labels])

        columns = ["doc_id", "head_id", "tail_id", "pred_tensor", "gold_tensor", "pred_label", "gold_label"]
        prediction_data = DataFrame.from_records(records, columns=columns)

        return prediction_data

    def save_predictions(
            self,
            output_data: List[Tuple],
            output_file: Path
    ):
        prediction_data = self.aggregate_predictions(output_data)
        prediction_data.to_csv(output_file, sep="\t", index=False)

    def save_logits(
            self,
            output_data: List[Tuple],
            output_file: Path
    ) -> None:
        doc_ids = np.concatenate([np.array(out[2].document_ids) for out in output_data], axis=0)
        head_ids = np.concatenate([np.array(out[2].head_ids) for out in output_data], axis=0)
        tails_ids = np.concatenate([np.array(out[2].tail_ids) for out in output_data], axis=0)

        all_logits = torch.concat([output.logits for ( _, _,  output) in output_data], dim=0).float().cpu().numpy()

        prediction_dict = defaultdict(dict)
        for doc_id, head_id, tail_id, logits in zip(doc_ids, head_ids, tails_ids, all_logits):
            assert (head_id,tail_id) not in prediction_dict[doc_id]
            prediction_dict[doc_id][(head_id,tail_id)] = logits

        with output_file.open("wb") as out_stream:
            pickle.dump(prediction_dict, out_stream)
