from pytorch_lightning.callbacks import Callback, TQDMProgressBar

from evaluation import extract_features, knn_accuracy


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
    """

    def __init__(self, every_n_epochs=5):
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        is_last_epoch = epoch == trainer.max_epochs
        if epoch % self.every_n_epochs != 0 and not is_last_epoch:
            return

        datamodule = trainer.datamodule
        bank_features, bank_labels = extract_features(pl_module.encoder, datamodule.train_eval_dataloader(), pl_module.device)
        val_features, val_labels = extract_features(pl_module.encoder, datamodule.val_dataloader(), pl_module.device)
        accuracy = knn_accuracy(bank_features, bank_labels, val_features, val_labels)
        pl_module.log("val_knn_top1", accuracy["top1"], prog_bar=True)
        pl_module.log("val_knn_top5", accuracy["top5"])
