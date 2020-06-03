import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

import util
from base_model import Model
from classifier import Classifier
from dependency_parser import DependencyParser
from enumeration import Task
from pl_util.early_stopping import EarlyStopping
from pl_util.logging import Logging
from tagger import Tagger


def main(hparams):
    if hparams.cache_dataset:
        if not hparams.cache_path:
            hparams.cache_path = os.path.join(os.path.expanduser("~"), ".cache/clnlp")
        os.makedirs(hparams.cache_path, exist_ok=True)

    model = {
        Task.conllner: Tagger,
        Task.wikiner: Tagger,
        Task.udpos: Tagger,
        Task.xnli: Classifier,
        Task.pawsx: Classifier,
        Task.mldoc: Classifier,
        Task.langid: Classifier,
        Task.parsing: DependencyParser,
    }[hparams.task](hparams)

    os.makedirs(
        os.path.join(hparams.default_save_path, hparams.exp_name), exist_ok=True
    )
    logger = pl.loggers.TensorBoardLogger(
        hparams.default_save_path, name=hparams.exp_name, version=None
    )

    early_stopping = EarlyStopping(
        monitor=model.selection_criterion,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=True,
        mode=model.comparsion,
        strict=True,
    )

    base_dir = os.path.join(
        hparams.default_save_path,
        hparams.exp_name,
        f"version_{logger.version}" if logger.version is not None else "",
    )
    model.base_dir = base_dir
    filepath = os.path.join(
        base_dir, "ckpts", "ckpts_{epoch}-{%s:.3f}" % model.selection_criterion,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=model.selection_criterion,
        verbose=True,
        save_top_k=hparams.save_top_k,
        mode=model.comparsion,
        period=0,
        prefix="",
    )
    logging_callback = Logging(base_dir)

    trainer = pl.Trainer(
        logger=logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        callbacks=[logging_callback],
        default_save_path=hparams.default_save_path,
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        overfit_pct=hparams.overfit_pct,
        track_grad_norm=hparams.track_grad_norm,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        max_steps=hparams.max_steps,
        min_steps=hparams.min_steps,
        val_check_interval=int(hparams.val_check_interval)
        if hparams.val_check_interval > 1
        else hparams.val_check_interval,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
    )
    trainer.fit(model)

    if hparams.tst_langs:
        best_model = {v: k for k, v in checkpoint_callback.best_k_models.items()}[
            checkpoint_callback.best
        ]
        assert "select" not in trainer.callback_metrics
        trainer.callback_metrics["select"] = checkpoint_callback.best
        model = model.load_from_checkpoint(best_model)
        trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--min_delta", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--cache_dataset", default=False, type=util.str2bool)
    parser.add_argument("--cache_path", default="", type=str)
    ############################################################################
    parser.add_argument("--default_save_path", default="./", type=str)
    parser.add_argument("--gradient_clip_val", default=0, type=float)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--overfit_pct", default=0.0, type=float)
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=util.str2bool)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--amp_level", default="O1", type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    ############################################################################
    parser = Model.add_model_specific_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    parser = DependencyParser.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)
