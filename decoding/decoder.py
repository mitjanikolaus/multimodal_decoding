import os

import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl


DECODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/decoders/")


class Decoder(pl.LightningModule):

    def __init__(self, input_size, output_size, learning_rate):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.learning_rate = learning_rate
        self.loss_contrastive = ContrastiveLoss()
        self.loss_mse = nn.MSELoss() #TODO l2 regularization? with wd on optimizer?

    def forward(self, x):
        x = self.fc(x)
        return x

    def loss(self, preds, targets):
        return self.loss_contrastive(preds, targets) #TODO

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, estimates, candidates):
        inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
        # Normalize inside the einsum to avoid creating a copy of candidates
        scores = torch.einsum("bc,oc,o->bo", estimates, candidates, inv_norms)
        target = torch.arange(len(scores), device=estimates.device)
        loss = F.cross_entropy(scores, target)
        return loss

        # equivalent:
        # I_e = torch.nn.functional.normalize(candidates, dim=1)
        # T_e = estimates #torch.nn.functional.normalize(estimates, dim=1)
        # logits = T_e @ I_e.T #TODO* np.exp(t) # temperature
        #
        # loss = F.cross_entropy(logits, target)
        # # loss_t = F.cross_entropy(logits.T, target)
        # # loss_alt = (loss_i + loss_t) / 2

