import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics.functional import pairwise_cosine_similarity

from analyses.ridge_regression_decoding import CAPTION, IMAGE
from decoding.data import Standardize
from utils import ACC_CAPTIONS, ACC_IMAGES


def dist_mat_to_pairwise_acc(dist_mat):
    diag = dist_mat.diagonal().reshape(-1, 1)
    comp_mat = diag < dist_mat
    corrects = comp_mat.sum()
    # subtract the number of elements of the diagonal as these values are always "False" (not smaller than themselves)
    score = corrects / (dist_mat.numel() - diag.numel())
    return score


def get_distance_matrix(predictions, originals, metric='cosine'):
    if metric != "cosine":
        raise NotImplementedError()
    return 1 - pairwise_cosine_similarity(predictions, originals)


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

        results[acc_metric_name] = pairwise_accuracy(latents_mod, preds_mod, metric, standardize_predictions,
                                                     standardize_targets)

    return results


class Decoder(pl.LightningModule):

    def __init__(self, input_size, output_size, learning_rate, weight_decay, batch_size, mse_loss_weight):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.learning_rate = learning_rate
        self.loss_contrastive = ContrastiveLoss()
        self.loss_mse = nn.MSELoss()
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.mse_loss_weight = mse_loss_weight

        self.test_outputs = {}

    def forward(self, x):
        x = self.fc(x)
        return x

    def loss(self, preds, targets):
        contrastive_loss = self.loss_contrastive(preds, targets)
        mse_loss = self.loss_mse(preds, targets)
        loss = (1-self.mse_loss_weight) * contrastive_loss + self.mse_loss_weight * mse_loss
        return loss, contrastive_loss, mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss, contrastive_loss, mse_loss = self.loss(preds, y)

        # self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log('train_loss_contrastive', contrastive_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log('train_loss_mse', mse_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss, contrastive_loss, mse_loss = self.loss(preds, y)

        acc = pairwise_accuracy(y, preds)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_pairwise_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, targets, stim_types, _ = batch
        preds = self(x)
        loss, _, _ = self.loss(preds, targets)

        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        if "preds" not in self.test_outputs:
            # first test batch
            self.test_outputs["preds"] = preds
            self.test_outputs["targets"] = targets
            self.test_outputs["stim_types"] = stim_types
        else:
            self.test_outputs["preds"] = torch.concatenate((self.test_outputs["preds"], preds))
            self.test_outputs["targets"] = torch.concatenate((self.test_outputs["targets"], targets))
            self.test_outputs["stim_types"].extend(stim_types)

        return loss, preds

    def on_test_epoch_start(self):
        self.test_outputs = {}

    def on_test_epoch_end(self):
        targets = self.test_outputs["targets"]
        preds = self.test_outputs["preds"]
        stim_types = np.array(self.test_outputs["stim_types"])

        results = test_set_pairwise_acc_scores(targets, preds, stim_types)
        self.log_dict(results)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, estimates, candidates):
        # inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
        # # Normalize inside the einsum to avoid creating a copy of candidates
        # scores = torch.einsum("bc,oc,o->bo", estimates, candidates, inv_norms)
        # target = torch.arange(len(scores), device=estimates.device)
        # loss = F.cross_entropy(scores, target)

        # equivalent:
        candidates = torch.nn.functional.normalize(candidates, dim=1)
        # estimates = torch.nn.functional.normalize(estimates, dim=1)

        logits = estimates @ candidates.T #TODO* np.exp(t) # temperature

        target = torch.arange(candidates.shape[0], device=estimates.device)
        loss = F.cross_entropy(logits, target)
        # loss_2 = F.cross_entropy(logits.T, target)
        # loss = (loss_1 + loss_2) / 2

        return loss

