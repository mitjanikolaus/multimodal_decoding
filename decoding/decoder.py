import os

import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl

from analyses.ridge_regression_decoding import get_distance_matrix, dist_mat_to_pairwise_acc, CAPTION, IMAGE
from decoding.data import Standardize
from utils import ACC_CAPTIONS, ACC_IMAGES


def pairwise_accuracy(latents, predictions, metric="cosine", standardize_predictions=True,
                      standardize_targets=False):
    if standardize_predictions:
        preds_standardize = Standardize(predictions.mean(axis=0), predictions.std(axis=0))
        predictions = preds_standardize(predictions)
    if standardize_targets:
        latens_standardize = Standardize(latents.mean(axis=0), latents.std(axis=0))
        latents = latens_standardize(latents)

    dist_mat = get_distance_matrix(predictions, latents, metric)
    return dist_mat_to_pairwise_acc(dist_mat)


def test_set_pairwise_acc_scores(latents, predictions, stim_types, metric="cosine", standardize_predictions=True,
                                 standardize_targets=False):
    results = {}
    for modality, acc_metric_name in zip([CAPTION, IMAGE], [ACC_CAPTIONS, ACC_IMAGES]):
        preds_mod = predictions[stim_types == modality]
        latents_mod = latents[stim_types == modality]

        print(latents_mod.shape)
        print(preds_mod.shape)
        results[acc_metric_name] = pairwise_accuracy(latents_mod, preds_mod, metric, standardize_predictions,
                                                     standardize_targets)

    return results


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
        return self.loss_contrastive(preds, targets)


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
        acc = pairwise_accuracy(y.cpu(), preds.cpu())

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_pairwise_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, stim_types, _ = batch
        preds = self(x)
        loss = self.loss(preds, y)
        results = test_set_pairwise_acc_scores(y.cpu(), preds.cpu(), stim_types)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(results, on_step=True, on_epoch=True, logger=True)

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

