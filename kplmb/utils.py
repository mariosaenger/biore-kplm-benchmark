import logging
import pytorch_lightning as pl
import os

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger, Logger
from pytorch_lightning.utilities import rank_zero_only
from typing import List, cast, Optional
from torch import Tensor

from kplmb.evaluation import EvaluationResult


TYPE_TO_DEFAULT_NAME = {
    "CHEMICAL": "Chemical",
    "GENE-N": "Gene",
    "GENE-Y": "Gene",
    #
    "BRAND": "Chemical",
    "DRUG": "Chemical",
    "GROUP": "Chemical",
    "DRUG_N": "Chemical",
    #
    "protein": "Gene",
    "compound": "Chemical",
    #
}

logging.basicConfig(level=logging.INFO)


ID_FIELDS = [
    "batch_size",
    "debug",
    "data.id",
    "model.id",
    "model.lr",
    "model.entity_marker",
    "model.use_cls_token",
    "model.use_start_tokens",
    "model.use_end_tokens",
    "model.weight_decay",
    "model.use_norel_class",
    "model.blind_entities",
    "model.input_prompt",
    "model.max_length",
    "model.num_context_sentences",
    "trainer.accumulate_grad_batches",
    "trainer.gradient_clip_val"
]


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def init(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[Logger],
):
    for lg in logger:
        if isinstance(lg, WandbLogger):
            lg.experiment.define_metric(f"train/f1", summary="max")
            lg.experiment.define_metric(f"val/f1", summary="max")
            lg.experiment.define_metric(f"val/f1_epoch", summary="max")


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb
            wandb.finish()


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["batch_size"] = config["batch_size"]
    hparams["seed"] = config["seed"]
    hparams["data"] = config["data"]
    hparams["out_dir"] = os.path.abspath(os.getcwd())

    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    if "side_info" in config:
        hparams["side_info"] = config["side_info"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def build_experiment_id(config: DictConfig) -> str:
    id = ""

    for id_field in ID_FIELDS:
        value = config
        for part in id_field.split("."):
            value = value[part]
        id += f"|{id_field}={value}"


    sides_infos = (
        ",".join([side_info_key for side_info_key,_ in config["side_info"].items()])
        if "side_info" in config else ""
    )
    id += f"|sides_infos={sides_infos}"

    return id


def log_results(
        result: EvaluationResult,
        eval_prefix: str,
        logger: List[Logger],
        num_steps: Optional[int] = 1
):
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb_lg = cast(WandbLogger, lg)
            metrics = {
                f"{eval_prefix}/precision": result.precision,
                f"{eval_prefix}/recall": result.recall,
                f"{eval_prefix}/f1": result.f1_score,
            }

            for (label, precision, recall, f1_score) in result.class_results:
                metrics.update({
                    f"{eval_prefix}/cl_{label}/precision": precision,
                    f"{eval_prefix}/cl_{label}/recall": recall,
                    f"{eval_prefix}/cl_{label}/f1": f1_score,
                })

            wandb_lg.log_metrics(metrics) #, num_steps=num_steps)


def secure_two_dimensions(tensor: Tensor) -> Tensor:
    if len(tensor.shape) == 2:
        return tensor

    if len(tensor.shape) == 1:
        return tensor.view(1, -1)

    raise NotImplementedError(f"Given tensor has shape: {tensor.shape}")


def get_dataset(name: str) -> DatasetDict:
    if name == "ddi":
        data = load_dataset("bigbio/ddi_corpus", "ddi_corpus_re_bigbio_kb")

        train_val = data["train"].train_test_split(
            test_size=101,
            seed=221
        )

        data = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": data["test"]
        })

    elif name == "chemprot":
        data = load_dataset("bigbio/chemprot", "chemprot_shared_task_eval_bigbio_kb")

    elif name == "cpi":
        cpi_data = load_dataset("bigbio/cpi", "cpi_bigbio_kb")

        train_test = cpi_data["train"].train_test_split(
            test_size=300,
            seed=130
        )

        train_val = train_test["train"].train_test_split(
            test_size=300,
            seed=130
        )

        data = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_test["test"]
        })

    elif name == "cdg":
        data = load_dataset("bigbio/chem_dis_gene", "chem_dis_gene_curated_gene_dis_bigbio_kb")

        train_test = data["train"].train_test_split(
            test_size=123,
            seed=128
        )

        train_val = train_test["train"].train_test_split(
            test_size=80,
            seed=128
        )

        data = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_test["test"]
        })
    elif name == "bc5cdr":
        return load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb")

    else:
        raise AssertionError()

    return data


