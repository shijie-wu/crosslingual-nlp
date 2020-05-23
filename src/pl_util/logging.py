import os
import json

from pytorch_lightning.callbacks.base import Callback


class Logging(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.filename = os.path.join(save_dir, "results.jsonl")

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pl_module.reset_metrics()

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        with open(self.filename, "a") as fp:
            logs = {
                k: v
                for k, v in trainer.callback_metrics.items()
                if k.startswith("val_")
            }
            logs["step"] = trainer.global_step
            print(json.dumps(logs), file=fp)

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pl_module.reset_metrics()

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        with open(self.filename, "a") as fp:
            logs = {
                k: v
                for k, v in trainer.callback_metrics.items()
                if k.startswith("tst_") or k == "select"
            }
            assert "select" in logs
            print(json.dumps(logs), file=fp)
