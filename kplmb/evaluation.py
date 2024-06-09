import numpy as np
import pandas as pd
import torch
import torchmetrics

from collections import defaultdict
from datasets import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple

from pandas import DataFrame
from pytorch_lightning import Callback, Trainer, LightningModule

from kplmb.constants import NO_RELATION_CLASS


class EvaluationResult:

    def __init__(
            self,
            precision: float,
            recall: float,
            f1_score: float,
            class_results: Tuple[str, float, float, float]
    ):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.class_results = class_results


def evaluate_document_level_by_file(
        dataset: Dataset,
        prediction_file: Path,
        relation_to_id: Dict[str, int],
        limit_documents: Optional[int] = None
) -> EvaluationResult:
    prediction_data = pd.read_csv(prediction_file, sep="\t")

    return evaluate_document_level(
        dataset=dataset,
        prediction_data=prediction_data,
        relation_to_id=relation_to_id,
        limit_documents=limit_documents
    )


def evaluate_document_level(
        dataset: Dataset,
        prediction_data: DataFrame,
        relation_to_id: Dict[str, int],
        limit_documents: Optional[int] = None
) -> EvaluationResult:
    num_classes = len(relation_to_id) - 1 if NO_RELATION_CLASS in relation_to_id else len(relation_to_id)

    # Read goldstandard and form them to prediction vectors
    goldstandard = defaultdict(dict)
    num_documents = 0

    doc_to_entity_mention_map = {}

    for document in dataset:
        doc_id = str(document["id"])

        # Build mapping from mention_id to db_id
        entity_id_to_db_id = defaultdict(list)
        for entity in document["entities"]:
            for db_entry in entity["normalized"]:
                entity_id_to_db_id[entity["id"]].append(db_entry["db_id"])

        doc_to_entity_mention_map[document["id"]] = entity_id_to_db_id

        for relation in document["relations"]:
            relation_idx = relation_to_id[relation["type"]]
            if NO_RELATION_CLASS in relation_to_id:
                relation_idx -= 1

            # Reconstruct document-level pairs by using the db_ids instead of mention ids
            pair_ids = [
                "##".join([head_id, tail_id])
                for head_id in entity_id_to_db_id[relation["arg1_id"]]
                for tail_id in entity_id_to_db_id[relation["arg2_id"]]
            ]

            for pair_id in pair_ids:
                if pair_id in goldstandard[doc_id]:
                    gold_vector = goldstandard[doc_id][pair_id]
                else:
                    gold_vector = np.zeros(num_classes)

                gold_vector[relation_idx] = 1.0
                goldstandard[doc_id][pair_id] = gold_vector

        num_documents += 1
        if limit_documents is not None and num_documents == limit_documents:
            break

    # Read predictions and form them to prediction vectors
    prediction = defaultdict(dict)
    for id, row in prediction_data.iterrows():
        if not pd.notna(row["pred_label"]):
            continue

        # Get mapping of mention_ids to db_ids
        doc_id = str(row["doc_id"])
        entity_id_to_db_id = doc_to_entity_mention_map[doc_id]

        # Reconstruct db_id pairs
        pair_ids = [
            "##".join([head_id, tail_id])
            for head_id in entity_id_to_db_id[str(row["head_id"])]
            for tail_id in entity_id_to_db_id[str(row["tail_id"])]
        ]

        predicted_idx = [int(float(id)) for id in str(row["pred_label"]).split(",")]

        # Set prediction for all db_ids pairs
        for pair_id in pair_ids:
            if pair_id in prediction[doc_id]:
                pred_vector = prediction[doc_id][pair_id]
            else:
                pred_vector = np.zeros(num_classes)

            pred_vector[predicted_idx] = 1.0
            prediction[doc_id][pair_id] = pred_vector

    return compute_evaluation_result(dataset, num_classes, relation_to_id, goldstandard, prediction)


def evaluate_by_file(
        dataset: Dataset,
        prediction_file: Path,
        relation_to_id: Dict[str, int],
        limit_documents: Optional[int] = None
) -> EvaluationResult:
    prediction_data = pd.read_csv(prediction_file, sep="\t")

    return evaluate(
        dataset=dataset,
        prediction_data=prediction_data,
        relation_to_id=relation_to_id,
        limit_documents=limit_documents
    )


def evaluate(
        dataset: Dataset,
        prediction_data: DataFrame,
        relation_to_id: Dict[str, int],
        limit_documents: Optional[int] = None
) -> EvaluationResult:
    num_classes = len(relation_to_id) - 1 if NO_RELATION_CLASS in relation_to_id else len(relation_to_id)

    # Read predictions and form them to prediction vectors
    prediction = defaultdict(dict)
    for id, row in prediction_data.iterrows():
        prediction_vector = np.zeros(num_classes)

        if pd.notna(row["pred_label"]):
            predicted_idx = [int(float(id)) for id in str(row["pred_label"]).split(",")]
            prediction_vector[predicted_idx] = 1.0

        pair_key = "##".join([str(row["head_id"]), str(row["tail_id"])])
        prediction[str(row["doc_id"])][pair_key] = prediction_vector

    # Read goldstandard and form them to prediction vectors
    goldstandard = defaultdict(dict)
    num_documents = 0

    for document in dataset:
        doc_id = str(document["id"])
        for relation in document["relations"]:
            relation_idx = relation_to_id[relation["type"]]
            if NO_RELATION_CLASS in relation_to_id:
                relation_idx -= 1

            pair_key = "##".join([str(relation["arg1_id"]), str(relation["arg2_id"])])
            if pair_key in goldstandard[doc_id]:
                gold_vector = goldstandard[doc_id][pair_key]
            else:
                gold_vector = np.zeros(num_classes)

            gold_vector[relation_idx] = 1.0
            goldstandard[doc_id][pair_key] = gold_vector

        num_documents += 1
        if limit_documents is not None and num_documents == limit_documents:
            break

    return compute_evaluation_result(dataset, num_classes, relation_to_id, goldstandard, prediction)


def compute_evaluation_result(
        dataset: Dataset,
        num_classes: int,
        relation_to_id: Dict[str, int],
        goldstandard: Dict[str, Dict],
        prediction: Dict[str, Dict]
) -> EvaluationResult:
    gold_vectors = []
    prediction_vectors = []

    for document in dataset:
        doc_id = document["id"]

        doc_goldstandard = goldstandard[doc_id]
        doc_prediction = prediction[doc_id]

        # First, go through all goldstandard relations and search
        # for the respective prediction for this pair
        found_pair_ids = set()
        for pair_id, gold_vector in doc_goldstandard.items():
            gold_vectors.append(gold_vector)

            if pair_id in doc_prediction:
                prediction_vectors.append(doc_prediction[pair_id])
                found_pair_ids.add(pair_id)
            else:
                # This might happen due to a pair of entities not occurring in a sentence
                prediction_vectors.append(np.zeros(num_classes))

        # Second, go through all predictions (not part of the goldstandard) and
        # add their prediction vectors
        for pair_id, pred_vector in doc_prediction.items():
            if pair_id in found_pair_ids:
                continue

            prediction_vectors.append(pred_vector)
            gold_vectors.append(np.zeros(num_classes))

    prediction_vectors = torch.tensor(np.array(prediction_vectors))
    gold_vectors = torch.tensor(np.array(gold_vectors))

    if num_classes > 1:
        precision_metric = torchmetrics.Precision(task="multilabel", num_labels=num_classes, average="micro")
        recall_metric = torchmetrics.Recall(task="multilabel", num_labels=num_classes, average="micro")
        f1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average="micro")
    else:
        precision_metric = torchmetrics.Precision(task="binary")
        recall_metric = torchmetrics.Recall(task="binary")
        f1_metric = torchmetrics.F1Score(task="binary")

    precision = precision_metric(prediction_vectors, gold_vectors)
    print(f"Precision: {precision}")

    recall = recall_metric(prediction_vectors, gold_vectors)
    print(f"Recall: {recall}")

    f1_score = f1_metric(prediction_vectors, gold_vectors)
    print(f"F1 score: {f1_score}")

    label_list = [
        (idx, label) for label, idx in relation_to_id.items()
        if label != NO_RELATION_CLASS
    ]
    label_list = sorted(label_list, key=lambda t: t[0])

    cls_precision_metric = torchmetrics.Precision(task="binary")
    cls_recall_metric = torchmetrics.Recall(task="binary")
    cls_f1_metric = torchmetrics.F1Score(task="binary")

    class_results = []
    for (idx, label) in label_list:
        if NO_RELATION_CLASS in relation_to_id:
            idx = idx - 1

        cls_prediction = prediction_vectors[:, idx:idx + 1]
        cls_gold = gold_vectors[:, idx:idx + 1]

        cls_precision: float = cls_precision_metric(cls_prediction, cls_gold)
        cls_recall: float = cls_recall_metric(cls_prediction, cls_gold)
        cls_f1_score: float = cls_f1_metric(cls_prediction, cls_gold)

        class_results.append((label, cls_precision, cls_recall, cls_f1_score))

        print(f"{label}")
        print(f"  Precision: {cls_precision}")
        print(f"  Recall: {cls_recall}")
        print(f"  F1 score: {cls_f1_score}")

    return EvaluationResult(
        precision=precision.cpu().item(),
        recall=recall.cpu().item(),
        f1_score=f1_score.cpu().item(),
        class_results=class_results
    )


class EvaluationCallback(Callback):

    def __init__(
            self,
            metric_name: str,
            dataset: Dataset,
            relation_to_id: Dict[str, int],
            is_document_level: bool,
            limit_documents: Optional[int] = None
    ):
        self.metric_name = metric_name
        self.dataset = dataset
        self.relation_to_id = relation_to_id
        self.is_document_level = is_document_level
        self.limit_documents = limit_documents

        self.eval_func = evaluate_document_level if self.is_document_level else evaluate

        self.best_f1 = 0.0

        self.tsv_output_file = Path(metric_name.replace("/", "_") + ".tsv")
        self.pkl_output_file = Path(metric_name.replace("/", "_") + ".pkl")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.best_f1 = -1.0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        prediction_data = pl_module.aggregate_predictions(pl_module.val_outputs)

        result = self.eval_func(
            dataset=self.dataset,
            prediction_data=prediction_data,
            relation_to_id=self.relation_to_id,
            limit_documents=self.limit_documents
        )

        pl_module.log(self.metric_name, result.f1_score, prog_bar=False)

        if result.f1_score > self.best_f1:
            self.best_f1 = result.f1_score
            pl_module.log(self.metric_name + "_best", result.f1_score, prog_bar=False)

            prediction_data.to_csv(self.tsv_output_file, sep="\t")
            pl_module.save_logits(pl_module.val_outputs, self.pkl_output_file)
