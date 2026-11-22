from pytorch_lightning.callbacks import Callback, TQDMProgressBar

from evaluation import knn_by_layer


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar.

    The validation bar is drawn on a second line, and Colab can't redraw a second
    line in place, so every update would be printed as a new line.
    Validation metrics still appear at the end of the main epoch bar.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class KNNMonitor(Callback):
    """Log validation k-NN accuracy of the encoder every few epochs.

    This is the model-selection metric of the assignment: hyperparameters are
    chosen by representation quality (k-NN), not by reconstruction error. The bank
    is the train subset the datamodule serves, the queries are the 10k validation
    hold-out. No labels reach the encoder: they are only used for the neighbor vote.

    Both candidate layers are measured and logged as val_knn_top1_<layer>. The layer chosen
    as h (config feature_layer) is also logged under the plain names val_knn_top1 / val_knn_top5,
    and that is the number the search selects on.
    """

    def __init__(self, every_n_epochs=5, feature_layer="conv"):
        self.every_n_epochs = every_n_epochs
        self.feature_layer = feature_layer

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        is_last_epoch = epoch == trainer.max_epochs
        if epoch % self.every_n_epochs != 0 and not is_last_epoch:
            return

        accuracy_by_layer = knn_by_layer(pl_module.encoder, trainer.datamodule, pl_module.device)
        for layer, accuracy in accuracy_by_layer.items():
            pl_module.log(f"val_knn_top1_{layer}", accuracy["top1"])
        chosen = accuracy_by_layer[self.feature_layer]
        pl_module.log("val_knn_top1", chosen["top1"], prog_bar=True)
        pl_module.log("val_knn_top5", chosen["top5"])
