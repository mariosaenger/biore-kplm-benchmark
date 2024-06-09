import segtok.segmenter
import torch

from collections import defaultdict
from datasets import DatasetDict
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizer
from typing import Tuple, Dict, List, Optional, Union

from kplmb import utils
from kplmb.constants import EntityMarker, NO_RELATION_CLASS, ENTITY_TYPE_TO_DEFAULT_NAME
from kplmb.models.transformer import EmbeddingConfiguration, MoleculeStructureEncoderConfiguration
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer


log = utils.get_logger("DATA")


PAIR_GENERATION_STRATEGY_ALL = "all"
PAIR_GENERATION_STRATEGY_DDI = "ddi"


def build_label_dictionary(data: DatasetDict, add_no_relation_class: bool) -> Dict[str, int]:
    relation_types = sorted(set([
        relation["type"]
        for split in data.keys()
        for document in data[split]
        for relation in document["relations"]
    ]))

    if add_no_relation_class:
        relation_types = [NO_RELATION_CLASS] + relation_types

    relation_type_to_id = {
        relation_type: i
        for i, relation_type in enumerate(relation_types)
    }

    return relation_type_to_id


def build_relation_pair_types(data: DatasetDict) -> List[Tuple[str, str]]:
    documents = [
        document
        for split in data.keys()
        for document in data[split]
    ]

    relation_pair_types = set()
    for document in documents:
        entity_id_to_entity = {
            entity["id"]: entity
            for entity in document["entities"]
        }

        for relation in document["relations"]:
            head_type = entity_id_to_entity[relation["arg1_id"]]["type"].lower()
            tail_type = entity_id_to_entity[relation["arg2_id"]]["type"].lower()

            relation_pair_types.add((head_type, tail_type))

    return list(relation_pair_types)


def split_into_sentences(text: str) -> List[Tuple[int, int, int, str]]:
    # Split text into sentences via segtok
    sentences = segtok.segmenter.split_single(text)

    # Re-construct sentence tuples consisting of (id, start, end, sentence-text)
    remaining_text = text
    sentence_offset = 0
    sentence_tuples = []

    for i, sentence in enumerate(sentences):
        start = remaining_text.find(sentence)
        assert start >= 0

        sentence_tuples.append((
            i,
            sentence_offset + start,
            sentence_offset + start + len(sentence),
            sentence
        ))

        remaining_text = remaining_text[start + len(sentence):]
        sentence_offset += start + len(sentence)

    for (i, start, end, sentence) in sentence_tuples:
        assert sentence == text[start:end]

    return sentence_tuples


def get_sentence(entity: Dict, sentences: List[Tuple[int, int, int, str]]) -> Optional[Tuple[int, int, int, str]]:
    entity_start, entity_end = entity["offsets"][0]

    for (id, start, end, sentence) in sentences:
        if entity_start >= start and entity_end <= end:
            return (id, start, end, sentence)

    return None


def insert_pair_markers(
        head_entity: Dict,
        tail_entity: Dict,
        text: str,
        text_offset: int,
        entity_marker: EntityMarker,
        blind_entities: bool
):
    # Check which entity occurs first in the sentence
    if head_entity["offsets"][0][0] < tail_entity["offsets"][0][1]:
        # Head entity is first
        first_entity = head_entity
        first_start, first_end = head_entity["offsets"][0]
        first_stoken, first_etoken = entity_marker.value[0:2]

        second_entity = tail_entity
        second_start, second_end = tail_entity["offsets"][0]
        second_stoken, second_etoken = entity_marker.value[2:4]
    else:
        # Tail entity is first
        first_entity = tail_entity
        first_start, first_end = tail_entity["offsets"][0]
        first_stoken, first_etoken = entity_marker.value[2:4]

        second_entity = head_entity
        second_start, second_end = head_entity["offsets"][0]
        second_stoken, second_etoken = entity_marker.value[0:2]

    if blind_entities:
        first_entity_text = first_entity["type"]
        second_entity_text = second_entity["type"]
    else:
        first_entity_text = text[first_start-text_offset:first_end-text_offset]
        second_entity_text = text[second_start-text_offset:second_end-text_offset]

    sentence_text = "".join([
        text[0: first_start-text_offset],                   # text before first entity
        first_stoken,                                       # first entity (incl. marking)
        first_entity_text,
        first_etoken,
        text[first_end-text_offset: second_start-text_offset], # text between the entities
        second_stoken,                                        # second entity (incl. marking)
        second_entity_text,
        second_etoken,
        text[second_end-text_offset:]                         # text after the second entity
    ])

    return sentence_text


def augment_with_context_sentence(
    pair_text: str,
    start_sentence_id: int,
    end_sentence_id: int,
    sentences: List[Tuple[int, int, int, str]],
    max_length: int,
    num_context_sentences: int,
    tokenizer: PreTrainedTokenizer
) -> str:
    pair_text_encoded = tokenizer.encode_plus(
        text=pair_text,
        truncation="longest_first",
        max_length=max_length,
    )
    pair_text_length = len(pair_text_encoded.input_ids)

    remaining_length = max_length - pair_text_length
    remaining_context = num_context_sentences
    sentence_index_offset = 1

    while remaining_context > 0:
        # First, try to append a sentence following the sentences under investigation
        next_sentence_index = end_sentence_id + sentence_index_offset
        if next_sentence_index < len(sentences):
            next_sentence = " " + sentences[next_sentence_index][3]
            sentence_encoded = tokenizer.encode_plus(
                text=next_sentence,
                truncation="longest_first",
                max_length=max_length,
            )
            sentence_length = len(sentence_encoded.input_ids)
            if sentence_length > remaining_length:
                break

            remaining_length -= sentence_length
            pair_text += next_sentence

        # First, try to prepend a sentences that occurs before the sentence currently under investigation
        prev_sentence_index = start_sentence_id - sentence_index_offset
        if prev_sentence_index >= 0:
            prev_sentence = sentences[prev_sentence_index][3] + " "
            sentence_encoded = tokenizer.encode_plus(
                text=prev_sentence,
                truncation="longest_first",
                max_length=max_length,
            )
            sentence_length = len(sentence_encoded.input_ids)
            if sentence_length > remaining_length:
                break

            remaining_length -= sentence_length - 1
            pair_text = prev_sentence + pair_text

        sentence_index_offset += 1
        remaining_context -= 1

    return pair_text


def generate_examples(
        dataset: Dict,
        mention_mapping: Dict[str, str],
        splits: List[str],
        pair_generation_strategy: str,
        tokenizer: PreTrainedTokenizer,
        pair_types: List[Tuple[str, str]],
        entity_marker: EntityMarker,
        max_sentence_distance: int,
        max_length: int,
        blind_entities: bool,
        use_norel_class: bool,
        input_prompt: str,
        add_text_prompt: bool,
        num_context_sentences: int,
        label_to_id: Dict[str, int],
        limit_documents: Optional[int] = None,
        textual_context_information: Dict[str, str] = None,
        embedding_configurations: List[EmbeddingConfiguration] = None,
        structure_encoding_configurations: List[MoleculeStructureEncoderConfiguration] = None,
) -> List[Dict]:
    """ Generates all training examples for a given dataset and split.

    :param dataset: Loaded BigBio data set
    :param mention_mapping: Mapping between mention ids and entity knowledge base ids
    :param splits: Data set split for which the examples should be created
    :param pair_generation_strategy: Pair generation strategy to be used
    :param tokenizer: Tokenizer for processing the sentences
    :param pair_types: List of valid entity type pairs (e.g., [(Chemical, Disease)])
    :param entity_marker: Entity marker strategy to be used
    :param max_sentence_distance: Maximal sentence distance between two entities consituting a pair
    :param max_length: Max length of the examples text
    :param blind_entities: Indicates whether to blind entities or not
    :param use_norel_class: Indicates whether to use a special no relation class
    :param input_prompt: Prompt which should be prepended to the input text
    :param add_text_prompt: Additional text prompt
    :param num_context_sentences: Number of additional context sentences
    :param label_to_id: Relation name to relation index mapping
    :param limit_documents: Limit the number of documents for which input examples should be created
    :param textual_context_information: Additional textual context information to be used
    :param embedding_configurations: Configured embedded context information
    :param structure_encoding_configurations: Configured molecular context information

    :return: List of examples
    """
    embedding_configurations = embedding_configurations if embedding_configurations is not None else []
    structure_encoding_configurations = structure_encoding_configurations if structure_encoding_configurations is not None else []

    documents = [document for split in splits for document in dataset[split]]
    if limit_documents:
        documents = documents[:limit_documents]

    str_enc_to_tokenizer = {}
    for str_encoding_config in structure_encoding_configurations:
        if str_encoding_config.encoder_type == "molbert":
            smiles_tokenizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(
                max_length=str_encoding_config.max_length,
                permute=False
            )
        elif str_encoding_config.encoder_type == "chemberta":
            smiles_tokenizer = RobertaTokenizer.from_pretrained(str_encoding_config.model_path)
        else:
            raise NotImplementedError(f"Unsupported encoder type: {str_encoding_config.encoder_type}")

        str_enc_to_tokenizer[str_encoding_config.id] = smiles_tokenizer

    examples = []

    num_truncated_pairs = 0
    num_intersentence_pairs = 0
    printed_instances = 0

    for document in tqdm(documents, desc="tokenize", total=len(documents)):
        entity_id_to_entity = {}
        for entity in document["entities"]:
            entity_id_to_entity[entity["id"]] = entity

        pair_to_relations = defaultdict(set)
        for relation in document["relations"]:
            pair_to_relations[(relation["arg1_id"], relation["arg2_id"])].add(relation["type"])

        document_text = " ".join([
            text
            for passage in document["passages"]
            for text in passage["text"]
        ])
        sentences = split_into_sentences(document_text)

        if pair_generation_strategy == PAIR_GENERATION_STRATEGY_ALL:
            pairs = [
                (head_entity, tail_entity)
                for head_entity in document["entities"]
                for tail_entity in document["entities"]
                if head_entity["id"] != tail_entity["id"]
            ]

        elif pair_generation_strategy == PAIR_GENERATION_STRATEGY_DDI:
            pairs = [
                (document["entities"][i], document["entities"][i+j+1])
                for i in range(len(document["entities"]))
                for j in range(len(document["entities"]) - i - 1)
            ]

        else:
            raise NotImplementedError(f"Unsupported strategy {pair_generation_strategy}")

        for (head_entity, tail_entity) in pairs:
            type_pair = (head_entity["type"].lower(), tail_entity["type"].lower())
            if type_pair not in pair_types:
                continue    # Entity types of head and tail doesn't match types of pairs under investigation!

            pair_id = (head_entity["id"], tail_entity["id"])

            head_sentence = get_sentence(head_entity, sentences)
            tail_sentence = get_sentence(tail_entity, sentences)

            if head_sentence is None or tail_sentence is None:
                continue

            # We ignore pairs that are too far away from each other
            sentence_distance = abs(head_sentence[0] - tail_sentence[0])
            if sentence_distance > max_sentence_distance:
                if len(pair_to_relations.get(pair_id, [])) > 0:
                    num_intersentence_pairs += 1
                continue

            # Prepend prompt (if configured)
            prompt = ""
            prompt_length = 0
            if input_prompt:
                prompt = input_prompt\
                    .replace("[head]", head_entity["text"][0])\
                    .replace("[tail]", tail_entity["text"][0])

                prompt_encoded = tokenizer.encode_plus(
                    text=prompt,
                    truncation="longest_first",
                    max_length=max_length,
                )
                prompt_length = len(prompt_encoded.input_ids)

            start_idx = min(head_sentence[1], tail_sentence[1])
            end_idx = max(head_sentence[2], tail_sentence[2])

            example_sentence = insert_pair_markers(
                head_entity=head_entity,
                tail_entity=tail_entity,
                text=document_text[start_idx:end_idx],
                text_offset=start_idx,
                entity_marker=EntityMarker.SPECIAL_TOKEN,
                blind_entities=blind_entities
            )

            if num_context_sentences > 0:
                example_sentence = augment_with_context_sentence(
                    pair_text=example_sentence,
                    start_sentence_id=min(head_sentence[0], tail_sentence[0]),
                    end_sentence_id=max(head_sentence[0], tail_sentence[0]),
                    sentences=sentences,
                    max_length=max_length-prompt_length,
                    num_context_sentences=num_context_sentences,
                    tokenizer=tokenizer
                )

            example_sentence = prompt + example_sentence

            head_mention_id = document["document_id"] + "_" + head_entity["id"]
            head_id = mention_mapping.get(head_mention_id, "")

            tail_mention_id = document["document_id"] + "_" + tail_entity["id"]
            tail_id = mention_mapping.get(tail_mention_id, "")

            if textual_context_information:
                head_text = textual_context_information.get(head_id, "")
                tail_text = textual_context_information.get(tail_id, "")

                if add_text_prompt and len(head_text) > 0:
                    head_type = ENTITY_TYPE_TO_DEFAULT_NAME[head_entity["type"].lower()]
                    head_text = f"{head_type} {head_entity['text'][0]} can be described by: {head_text}"

                if add_text_prompt and len(tail_text) > 0:
                    tail_type = ENTITY_TYPE_TO_DEFAULT_NAME[tail_entity["type"].lower()]
                    tail_text = f"{tail_type} {tail_entity['text'][0]} can be described by: {tail_text}"

                example_sentence = f"{example_sentence}[SEP]{head_text}[SEP]{tail_text}"

            printed_instances += 1
            if printed_instances <= 8:
                print(f"Example {printed_instances}:\n{example_sentence}\n")

            features_text = tokenizer.encode_plus(
                text=example_sentence,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt"
            )

            features = {
                "input_ids": features_text.input_ids.squeeze(),
                "attention_mask": features_text.attention_mask.squeeze()
            }

            if "token_type_ids" in features_text:
                features["token_type_ids"] = torch.tensor([0] * features_text.input_ids.shape[1], dtype=torch.int8) # + [1] * len(features_side.input_ids)

            if entity_marker == EntityMarker.SPECIAL_TOKEN:
                try:
                    assert "HEAD-S" in tokenizer.decode(features["input_ids"])
                    assert "HEAD-E" in tokenizer.decode(features["input_ids"])
                    assert "TAIL-S" in tokenizer.decode(features["input_ids"])
                    assert "TAIL-E" in tokenizer.decode(features["input_ids"])
                except AssertionError:
                    num_truncated_pairs += 1
                    continue

            for emb_conf in embedding_configurations:
                id_to_index = emb_conf.entity_id_to_embedding_index

                if emb_conf.target == "both":
                    features[f"{emb_conf.id}_head_index"] = id_to_index.get(head_id, -1) + 1
                    features[f"{emb_conf.id}_tail_index"] = id_to_index.get(tail_id, -1) + 1
                elif emb_conf.target == "head":
                    features[f"{emb_conf.id}_{emb_conf.target}_index"] = id_to_index.get(head_id, -1) + 1
                elif emb_conf.target == "tail":
                    features[f"{emb_conf.id}_{emb_conf.target}_index"] = id_to_index.get(tail_id, -1) + 1
                else:
                    raise NotImplementedError()

            for str_enc_config in structure_encoding_configurations:
                smiles_tokenizer = str_enc_to_tokenizer[str_enc_config.id]

                if str_enc_config.encoder_type == "molbert":
                    if str_enc_config.target == "head" or str_enc_config.target == "both":
                        head_smiles = str_enc_config.entity_id_to_smiles.get(head_id, "[SEP]")
                        input_ids, valid = smiles_tokenizer.transform([head_smiles])
                        features[f"{str_enc_config.id}_head_input_ids"] = input_ids
                        features[f"{str_enc_config.id}_head_valid"] = valid

                    if str_enc_config.target == "tail" or str_enc_config.target == "both":
                        tail_smiles = str_enc_config.entity_id_to_smiles.get(tail_id, "[SEP]")
                        input_ids, valid = smiles_tokenizer.transform([tail_smiles])
                        features[f"{str_enc_config.id}_tail_input_ids"] = input_ids
                        features[f"{str_enc_config.id}_tail_valid"] = valid

                elif str_enc_config.encoder_type == "chemberta":
                    if str_enc_config.target == "head" or str_enc_config.target == "both":
                        head_smiles = str_enc_config.entity_id_to_smiles.get(head_id, "<unk>")
                        output = smiles_tokenizer.encode_plus(
                            head_smiles,
                            max_length=str_enc_config.max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=True
                        )

                        features[f"{str_enc_config.id}_head_input_ids"] = output.data["input_ids"]
                        features[f"{str_enc_config.id}_head_attention_mask"] = output.data["attention_mask"]

                    if str_enc_config.target == "tail" or str_enc_config.target == "both":
                        tail_smiles = str_enc_config.entity_id_to_smiles.get(tail_id, "<unk>")
                        output = smiles_tokenizer.encode_plus(
                            tail_smiles,
                            max_length=str_enc_config.max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=True
                        )
                        features[f"{str_enc_config.id}_tail_input_ids"] = output.data["input_ids"]
                        features[f"{str_enc_config.id}_tail_attention_mask"] = output.data["attention_mask"]

            features["labels"] = torch.tensor([0 for i in range(len(label_to_id))], dtype=torch.float)
            for label in pair_to_relations[pair_id]:
                features["labels"][label_to_id[label]] = 1.0

            if use_norel_class and features["labels"].sum() == 0:
                features["labels"][0] = 1.0

            examples.append({
                "document_id": document["id"],
                "head_id": head_entity["id"],
                "tail_id": tail_entity["id"],
                "relations": pair_to_relations[pair_id],
                "features": features
            })

    log.warning(f"  Truncated pairs: {num_truncated_pairs} ({num_truncated_pairs/len(examples)})")
    log.warning(f"  Intersentence pairs: {num_intersentence_pairs} ({num_intersentence_pairs/len(examples)})")

    return examples


class BigBioRelationClassificationDataset(Dataset):
    """ Class for handling BigBio relation data set"""

    def __init__(
            self,
            data: DatasetDict,
            mention_mapping: Dict[str, str],
            splits: Union[str, List[str]],
            tokenizer: PreTrainedTokenizer,
            pair_generation_strategy: str,
            pair_types: List[Tuple[str, str]],
            entity_marker: EntityMarker,
            max_sentence_distance: int,
            max_length: int,
            input_prompt: str,
            add_text_prompt: bool,
            num_context_sentences: int,
            blind_entities: bool,
            label_to_id: Dict[str, int],
            use_norel_class: bool,
            limit_documents: Optional[int] = None,
            entity_side_information: Dict[str, str] = None,
            embedding_configurations: List[EmbeddingConfiguration] = None,
            structure_encoding_configurations: List[MoleculeStructureEncoderConfiguration] = None,
    ):
        if isinstance(splits, str):
            splits = [splits]

        self.examples = generate_examples(
            dataset=data,
            mention_mapping=mention_mapping,
            splits=splits,
            tokenizer=tokenizer,
            pair_generation_strategy=pair_generation_strategy,
            pair_types=pair_types,
            entity_marker=entity_marker,
            max_sentence_distance=max_sentence_distance,
            max_length=max_length,
            input_prompt=input_prompt,
            add_text_prompt=add_text_prompt,
            num_context_sentences=num_context_sentences,
            blind_entities=blind_entities,
            use_norel_class=use_norel_class,
            label_to_id=label_to_id,
            limit_documents=limit_documents,
            textual_context_information=entity_side_information,
            embedding_configurations=embedding_configurations,
            structure_encoding_configurations=structure_encoding_configurations
        )

    def __getitem__(self, item):
        return self.examples[item].copy()

    def __len__(self):
        return len(self.examples)

    @classmethod
    def create(
            cls,
            data: DatasetDict,
            mention_mapping: Dict[str, str],
            splits: List[str],
            tokenizer: PreTrainedTokenizer,
            pair_generation_strategy: str,
            entity_marker: EntityMarker,
            max_sentence_distance: int,
            max_length: int,
            input_prompt: str,
            add_text_prompt: bool,
            num_context_sentences: int,
            blind_entities: bool,
            use_no_relation_class: bool,
            limit_documents: Optional[int] = None,
            entity_side_information: Optional[Dict[str, str]] = None,
            embedding_configurations: Optional[List[EmbeddingConfiguration]] = None,
            structure_encoding_configurations: Optional[List[MoleculeStructureEncoderConfiguration]] = None
    ) -> Tuple[Dict[str, Dataset], Dict[str, int], List[Tuple]]:

        # Build relation type to index mapping
        reltype_to_id = build_label_dictionary(data, use_no_relation_class)
        log.info(f" Label dictionary: {reltype_to_id}")

        # Build the list of valid entity type pairs
        rel_pair_types = build_relation_pair_types(data)
        log.info(f" Relation types: {rel_pair_types}")

        # Build a dataset for each split
        datasets = {}
        for split in splits:
            dataset = BigBioRelationClassificationDataset(
                data=data,
                mention_mapping=mention_mapping,
                splits=[split],
                tokenizer=tokenizer,
                pair_generation_strategy=pair_generation_strategy,
                pair_types=rel_pair_types,
                entity_marker=entity_marker,
                max_sentence_distance=max_sentence_distance,
                max_length=max_length,
                input_prompt=input_prompt,
                add_text_prompt=add_text_prompt,
                num_context_sentences=num_context_sentences,
                blind_entities=blind_entities,
                label_to_id=reltype_to_id,
                use_norel_class=use_no_relation_class,
                limit_documents=limit_documents,
                entity_side_information=entity_side_information,
                embedding_configurations=embedding_configurations,
                structure_encoding_configurations=structure_encoding_configurations,
            )
            log.info(f"  Size split {split}: {len(dataset)}")
            datasets[split] = dataset

        return datasets, reltype_to_id, rel_pair_types
