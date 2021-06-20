import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the pretrained models are saved
from transformers_multihead_attn_tutorial.model.anomaly_predictor import AnomalyPredictor
from transformers_multihead_attn_tutorial.model.reverse_predictor import ReversePredictor

CHECKPOINT_PATH = "../saved_models/tutorial6"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def train_anomaly(**kwargs):
    """
       :param kwargs: additional kwargs => model_name,  train_anom_loader, val_anom_loader, test_anom_loader
       :return: model, result
       """
    # Create ReversePredictor kwargs
    model_kwargs = {}
    for key in kwargs.keys():
        if key not in ["model_name", "train_anom_loader", "val_anom_loader", "test_anom_loader"]:
            model_kwargs[key] = kwargs[key]
    train_anom_loader = kwargs['train_anom_loader']
    val_anom_loader = kwargs['val_anom_loader']
    test_anom_loader = kwargs['test_anom_loader']
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=100,
                         gradient_clip_val=2,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = AnomalyPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = AnomalyPredictor(max_iters=trainer.max_epochs*len(train_anom_loader), **model_kwargs)
        trainer.fit(model, train_anom_loader, val_anom_loader)
        model = AnomalyPredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=train_anom_loader, verbose=False)
    val_result = trainer.test(model, test_dataloaders=val_anom_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_anom_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"], "train_acc": train_result[0]["test_acc"]}

    model = model.to(device)
    return model, result

