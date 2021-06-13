import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the pretrained models are saved
from transformers_multihead_attn_tutorial.model.reverse_predictor import ReversePredictor

CHECKPOINT_PATH = "../saved_models/tutorial6"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def train_reverse(**kwargs):
    """
    :param kwargs: additional kwargs => model_name,  train_loader, val_loader, test_loader
    :return: model, result
    """
    # Create ReversePredictor kwargs
    model_kwargs = {}
    for key in kwargs.keys():
        if key not in ["model_name", "train_loader", "val_loader", "test_loader"]:
            model_kwargs[key] = kwargs[key]
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, kwargs['model_name'])
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=10,
                         gradient_clip_val=5,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, kwargs['model_name'] + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor(max_iters=trainer.max_epochs * len(kwargs['train_loader']), **model_kwargs)
        trainer.fit(model, kwargs['train_loader'], kwargs['val_loader'])

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=kwargs['val_loader'], verbose=False)
    test_result = trainer.test(model, test_dataloaders=kwargs['test_loader'], verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result
