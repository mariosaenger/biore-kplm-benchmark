import hydra
import pytorch_lightning as pl
import pandas as pd
import shutil
import torch

from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader
from typing import List
from transformers import AutoTokenizer

from kplmb import utils
from kplmb.data import BigBioRelationClassificationDataset
from kplmb.constants import EntityMarker
from kplmb.evaluation import EvaluationCallback, evaluate_by_file, evaluate_document_level_by_file
from kplmb.models.transformer import EmbeddingConfiguration, MoleculeStructureEncoderConfiguration

log = utils.get_logger(__name__)


@hydra.main(config_path="../_configs/", config_name="config.yaml")
def train(config: DictConfig):
    """
        Contains training pipeline.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)

    # Log and save configuration
    log.info(config)

    # Information for w&b logging
    tags = [config.model.id]

    if config.debug:
        tags.append("DEBUG")

    if config.tag:
        tags += [config.tag] if isinstance(config.tag, str) else config.tag

    entity_side_information = None
    embedding_configurations = None
    str_encoder_configurations = None

    # Build side information configurations
    if "side_info" in config:
        entity_side_information = {}
        embedding_configurations = []
        str_encoder_configurations = []

        for side_info_key, side_info_config in config.side_info.items():
            tags.append(side_info_key)

            if side_info_config.type == "text":
                tsv_file = hydra.utils.to_absolute_path(side_info_config.tsv_file)
                side_data = pd.read_csv(tsv_file, sep="\t", index_col=side_info_config.id_column)

                for entity_id, value in side_data[side_info_config.text_column].items():
                    entity_side_information[str(entity_id)] = value

            elif side_info_config.type == "embedding":
                embedding_file = hydra.utils.to_absolute_path(side_info_config.embedding_file)
                embeddings = torch.load(embedding_file, map_location=torch.device("cpu"))

                mapping_file = hydra.utils.to_absolute_path(side_info_config.mapping_file)
                mapping_data = pd.read_csv(mapping_file, sep="\t", index_col="entity_id")
                mapping_data.index = mapping_data.index.map(lambda id: str(id))
                entity_id_to_index = mapping_data["index"].to_dict()

                embedding_conf = EmbeddingConfiguration(
                    id=side_info_key,
                    embeddings=embeddings,
                    entity_id_to_embedding_index=entity_id_to_index,
                    target=side_info_config.target,
                    hidden_size=side_info_config.hidden_size,
                    hidden_dropout=side_info_config.hidden_dropout,
                    output_size=side_info_config.output_size,
                    output_dropout=side_info_config.output_dropout,
                    freeze_embeddings=side_info_config.freeze_embeddings
                )

                assert side_info_config.target in ["both", "head", "tail"]
                embedding_configurations.append(embedding_conf)

            elif side_info_config.type == "structure":
                smiles_file = Path(hydra.utils.to_absolute_path(side_info_config.smiles_file))
                smiles_data = pd.read_csv(smiles_file, sep="\t", index_col="entity_id")
                entity_id_to_smiles = smiles_data["smiles"].to_dict()

                model_path = side_info_config.model_path
                if side_info_config.is_local_model:
                    model_path = Path(hydra.utils.to_absolute_path(model_path))

                str_encoder_config = MoleculeStructureEncoderConfiguration(
                    id=side_info_key,
                    target=side_info_config.target,
                    encoder_type=side_info_config.encoder_type,
                    model_path=model_path,
                    entity_id_to_smiles=entity_id_to_smiles,
                    max_length=side_info_config.max_length,
                    freeze_encoder=side_info_config.freeze_encoder,
                )

                assert side_info_config.target in ["both", "head", "tail"]
                str_encoder_configurations.append(str_encoder_config)

    # Load mention mapping
    mention_mapping_file = hydra.utils.to_absolute_path(config.data.mention_mapping)
    mm_data = pd.read_csv(mention_mapping_file, sep="\t", index_col="mention_id")

    mention_mapping = {}
    for mention_id, row in mm_data.iterrows():
        mention_mapping[str(mention_id)] = str(row["entity_id"])

    if config.model.transformer.startswith("_resources"):
        config.model.transformer = hydra.utils.to_absolute_path(config.model.transformer)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.transformer, use_fast=True)

    # Add special tokens for entity marking to tokenizer if necessary
    entity_marker = EntityMarker.__getitem__(config.model.entity_marker)
    if entity_marker == EntityMarker.SPECIAL_TOKEN:
        special_tokens = [token for token in entity_marker.value]
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    # Load data set
    log.info(f"Loading dataset {config.data.dataset_name}")
    data_path = (
        config.data.dataset_path
        if not config.data.local
        else hydra.utils.to_absolute_path(config.data.dataset_path)
    )

    data = load_dataset(data_path, name=config.data.dataset_name)

    # Automatically create train/val/test splits - if necessary
    if config.data.create_val_from_train:
        train_dev = data["train"].train_test_split(
            test_size=config.data.val_size,
            seed=config.data.split_seed
        )

        data = DatasetDict({
            "train": train_dev["train"],
            "validation": train_dev["test"],
            "test": data["test"] if "test" in data else data["validation"]
        })

    elif config.data.create_val_test_from_train:
        train_test = data["train"].train_test_split(
            test_size=config.data.test_size,
            seed=config.data.split_seed
        )

        train_val = train_test["train"].train_test_split(
            test_size=config.data.val_size,
            seed=config.data.split_seed
        )

        data = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_test["test"]
        })

    if config.data.reduce_train:
        reduced_train = data["train"].train_test_split(
            train_size=config.data.reduce_train_size,
            seed=config.data.reduce_train_seed
        )

        data = DatasetDict({
            "train": reduced_train["train"],
            "validation": data["test"],
            "test": data["test"]
        })

    additional_train_data = None
    if "aug_data" in config:
        additional_train_data = []
        llm_reda_ds_path = hydra.utils.to_absolute_path("bigbio/llm_reda")

        for data_id, aug_data_config in config.aug_data.items():
            aug_data = load_dataset(
                path=llm_reda_ds_path,
                name="llm_reda_bigbio_kb",
                data_files=[hydra.utils.to_absolute_path(aug_data_config.data_file)]
            )["train"]

            if aug_data_config.limit_documents:
                aug_data = aug_data.train_test_split(
                    train_size=aug_data_config.limit_documents,
                    seed=aug_data_config.seed
                )["train"]

            additional_train_data.append(aug_data)
            tags.append(data_id)

    datasets, rel_to_id, pair_types = BigBioRelationClassificationDataset.create(
        data=data,
        mention_mapping=mention_mapping,
        splits=["train", "validation", "test"],
        tokenizer=tokenizer,
        pair_generation_strategy=config.data.pair_generation_strategy,
        entity_marker=entity_marker,
        max_sentence_distance=config.data.max_sentence_distance,
        max_length=config.model.max_length,
        input_prompt=config.model.input_prompt,
        add_text_prompt=config.model.add_text_prompt,
        num_context_sentences=config.model.num_context_sentences,
        blind_entities=config.model.blind_entities,
        use_no_relation_class=config.model.use_norel_class,
        limit_documents=config.data.limit_documents,
        entity_side_information=entity_side_information,
        embedding_configurations=embedding_configurations,
        structure_encoding_configurations=str_encoder_configurations,
        additional_train_data=additional_train_data
    )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        _convert_="partial",
        tokenizer=tokenizer,
        num_relation_types=len(rel_to_id),
        entity_marker=entity_marker,
        embedding_configurations=embedding_configurations,
        structure_encoding_configurations=str_encoder_configurations
    )

    # Init Lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                kwargs = {}
                if lg_conf._target_ == "pytorch_lightning.loggers.wandb.WandbLogger":
                    kwargs["tags"] = tags
                    kwargs["group"] = config.data.id
                    kwargs["notes"] = utils.build_experiment_id(config)

                    # Work-around to guarantee that W&B logs into run working dir
                    lg_conf["dir"] = str(Path(".").absolute())

                    # Run W&B initialization to prevent errors if running in multirun-mode
                    import wandb
                    wandb.init(project=lg_conf.project, **kwargs)

                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf, **kwargs))

    evaluation_callback = EvaluationCallback(
        metric_name="val/f1_epoch",
        dataset=data["validation"],
        relation_to_id=rel_to_id,
        limit_documents=config.data.limit_documents,
        is_document_level=config.data.is_doc_level,
    )

    callbacks = [evaluation_callback] + callbacks

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    config.trainer.enable_checkpointing = "model_checkpoint" in config.callbacks
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial"
    )

    log.info("Initializing")
    utils.init(config=config, model=model, trainer=trainer, callbacks=callbacks, logger=logger)
    utils.log_hyperparameters(config=config, model=model, trainer=trainer)

    train_loader = DataLoader(
        dataset=datasets["train"],
        shuffle=True,
        collate_fn=model.collate_fn,
        batch_size=config.batch_size,
        num_workers=int(config.workers / 2) if config.workers > 1 else config.workers,
        #pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=datasets["validation"],
        shuffle=False,
        collate_fn=model.collate_fn,
        batch_size=config.val_batch_size,
        num_workers=int(config.workers / 2) if config.workers > 1 else config.workers,
        #pin_memory=True,
    )

    model.num_training_steps = len(train_loader) * trainer.max_epochs

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_score = trainer.callback_metrics[config.model.optimized_metric]
    log.info(f"Best {config.model.optimized_metric} score: {best_score})")

    if config.run_test_evaluation:
        # Reload best model from training
        best_model_path = Path(trainer.checkpoint_callback.best_model_path)
        log.info(f"Best model path: {best_model_path}")
        model = model.load_from_checkpoint(best_model_path)

        # Re-run evaluation on validation set
        model.test_metric = "val_bm"
        trainer.test(model=model, dataloaders=val_loader)

        # Run evaluation on test set
        test_loader = DataLoader(
            dataset=datasets["test"],
            shuffle=False,
            collate_fn=model.collate_fn,
            batch_size=config.val_batch_size,
            num_workers=int(config.workers/2) if config.workers > 1 else config.workers,
            #pin_memory=True,
        )

        model.test_metric = "test_bm"
        trainer.test(model=model, dataloaders=test_loader)

        result = evaluate_by_file(
            dataset=data["test"],
            prediction_file=Path("test_bm.tsv"),
            relation_to_id=rel_to_id,
            limit_documents=config.data.limit_documents
        )
        utils.log_results(result, "test_eval", logger, model.num_training_steps)

        if config.data.is_doc_level:
            result = evaluate_document_level_by_file(
                dataset=data["test"],
                prediction_file=Path("test_bm.tsv"),
                relation_to_id=rel_to_id,
                limit_documents=config.data.limit_documents
            )
            utils.log_results(result, "test_eval_doc_level", logger, model.num_training_steps)

        # Delete best model
        if config.delete_checkpoint:
            shutil.rmtree(best_model_path.parent, ignore_errors=True)

    # Make sure everything closed properly
    utils.finish(config=config, model=model, trainer=trainer, callbacks=callbacks, logger=logger)


if __name__ == "__main__":
    train()
