import os
from argparse import ArgumentParser

import pytorch_lightning as pl

import util
from aligner import Aligner
from base_model import Model
from classifier import Classifier
from dependency_parser import DependencyParser
from enumeration import Task
from tagger import Tagger


def main(hparams):
    if hparams.cache_dataset:
        if not hparams.cache_path:
            hparams.cache_path = os.path.join(os.path.expanduser("~"), ".cache/clnlp")
        os.makedirs(hparams.cache_path, exist_ok=True)

    ModelClass = {
        Task.conllner: Tagger,
        Task.wikiner: Tagger,
        Task.udpos: Tagger,
        Task.xnli: Classifier,
        Task.pawsx: Classifier,
        Task.mldoc: Classifier,
        Task.langid: Classifier,
        Task.parsing: DependencyParser,
        Task.alignment: Aligner,
    }[hparams.task]
    if hparams.do_train:
        model = ModelClass(hparams)
    else:
        assert os.path.isfile(hparams.checkpoint)
        model = ModelClass.load_from_checkpoint(hparams.checkpoint)

    os.makedirs(
        os.path.join(hparams.default_save_path, hparams.exp_name), exist_ok=True
    )
    logger = pl.loggers.TensorBoardLogger(
        hparams.default_save_path, name=hparams.exp_name, version=None
    )

    early_stopping = pl.callbacks.EarlyStopping(
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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(base_dir, "ckpts"),
        filename="ckpts_{epoch}-{%s:.3f}" % model.selection_criterion,
        monitor=model.selection_criterion,
        verbose=True,
        save_last=hparams.save_last,
        save_top_k=hparams.save_top_k,
        mode=model.comparsion,
    )
    logging_callback = util.Logging(base_dir)
    lr_logger = pl.callbacks.LearningRateMonitor()
    callbacks = [early_stopping, checkpoint_callback, logging_callback, lr_logger]
    if isinstance(model, Aligner) and hparams.aligner_sim == "linear":
        callbacks.append(util.MappingCheckpoint(base_dir))

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=hparams.default_save_path,
        gradient_clip_val=hparams.gradient_clip_val,
        num_nodes=hparams.num_nodes,
        gpus=hparams.gpus,
        auto_select_gpus=True,
        overfit_batches=hparams.overfit_batches,
        track_grad_norm=hparams.track_grad_norm,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        max_steps=hparams.max_steps,
        min_steps=hparams.min_steps,
        val_check_interval=int(hparams.val_check_interval)
        if hparams.val_check_interval > 1
        else hparams.val_check_interval,
        log_every_n_steps=hparams.log_every_n_steps,
        accelerator=hparams.accelerator,
        precision=hparams.precision,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        replace_sampler_ddp=True,
        terminate_on_nan=True,
        amp_backend=hparams.amp_backend,
        amp_level=hparams.amp_level,
    )
    if hparams.do_train:
        trainer.fit(model)

    if hparams.do_test and hparams.tst_langs:
        if hparams.do_train:
            assert "select" not in trainer.callback_metrics
            trainer.callback_metrics["select"] = checkpoint_callback.best_model_score
            trainer.test(ckpt_path="best")
        else:
            trainer.test(model=model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--min_delta", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--save_last", default=False, type=util.str2bool)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--do_train", default=True, type=util.str2bool)
    parser.add_argument("--do_test", default=True, type=util.str2bool)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--cache_dataset", default=False, type=util.str2bool)
    parser.add_argument("--cache_path", default="", type=str)
    ############################################################################
    parser.add_argument("--default_save_path", default="./", type=str)
    parser.add_argument("--gradient_clip_val", default=0, type=float)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--overfit_batches", default=0.0, type=float)
    parser.add_argument("--track_grad_norm", default=-1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=util.str2bool)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--log_every_n_steps", default=10, type=int)
    parser.add_argument("--accelerator", default=None, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--amp_backend", default="native", type=str)
    # only used for non-native amp
    parser.add_argument("--amp_level", default="01", type=str)
    ############################################################################
    parser = Model.add_model_specific_args(parser)
    parser = Tagger.add_model_specific_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    parser = DependencyParser.add_model_specific_args(parser)
    parser = Aligner.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)
