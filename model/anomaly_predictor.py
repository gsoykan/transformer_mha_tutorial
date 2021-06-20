import torch.nn.functional as F

from transformers_multihead_attn_tutorial.model.transformer_predictor import TransformerPredictor


class AnomalyPredictor(TransformerPredictor):

    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch
        preds = self.forward(img_sets,
                             add_positional_encoding=False)  # No positional encodings as it is a set, not a sequence!
        preds = preds.squeeze(dim=-1)  # Shape: [Batch_size, set_size]
        loss = F.cross_entropy(preds, labels)  # Softmax/CE over set dimension
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
