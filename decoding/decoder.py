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
        self.loss_contrastive = ClipLoss() #TODO
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


class ClipLoss(torch.nn.Module):
    def __init__(self, linear=None, twin=True, pool=False, tmin=None, tmax=None,
                 tmin_train=None, tmax_train=None, dset_args=None, center=False):
        super().__init__()
        self.linear = None
        self.pool = pool
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.tmin = tmin
        self.tmax = tmax
        self.tmin_train = tmin_train
        self.tmax_train = tmax_train
        self.dset_args = dset_args

    def trim_samples(self, estimates, candidates):
        """Given estimates that is [B1, C, T] and candidates
        which is [B2, C, T], return estimates_trim of size [B1, C, T']
        and candidates_trim of size [B2, C, T'], such that T'
        corresponds to the samples between [self.tmin, self.tmax]
        """
        if self.training and (self.tmin_train is not None or self.tmax_train is not None):
            tmin, tmax = self.tmin_train, self.tmax_train
        else:
            tmin, tmax = self.tmin, self.tmax
        if (tmin is not None) or (tmax is not None):
            assert self.dset_args is not None
            assert self.dset_args.tmin is not None
            dset_tmin = self.dset_args.tmin
        if tmin is None:
            trim_min = 0
        else:
            assert tmin >= dset_tmin, 'clip.tmin should be above dset.tmin'
            trim_min = int((-dset_tmin + tmin) * self.dset_args.sample_rate)
        if tmax is None:
            trim_max = estimates.shape[-1]
        else:
            trim_max = int((-dset_tmin + tmax) * self.dset_args.sample_rate)
        estimates_trim = estimates[..., trim_min:trim_max]
        candidates_trim = candidates[..., trim_min:trim_max]
        return estimates_trim, candidates_trim

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of scores of matching.
        """
        estimates, candidates = self.trim_samples(estimates, candidates)
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        # We normalize inside the einsum, to avoid creating a copy
        # of candidates, which can be pretty big.
        scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
        """
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate):
        """Warning: estimate and candidate are not symmetrical.
        If estimate of shape [B, C, T] and candidate of size [B', C, T]
        with B'>=B, the first B samples of candidate are targets, while
        the remaining B'-B samples of candidate are only used as negatives.
        """
        # assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)


class CustomLoss(torch.nn.Module):
    def __init__(self, weight_mse=0.5):
        super().__init__()
        self.weight_mse = weight_mse
        loss_mse = nn.MSELoss()

    def forward(self, estimate, output):
        assert estimate.shape[1] == self.used_features.output_dimension and \
               output.shape[1] == self.used_features.dimension, \
               "Invalid features dim received. Are you using the correct " \
               "features for the loss?"

        loss = 0
        for feature in self.used_features.values():
            feature_name = feature.name
            feature_slice = self.used_features.get_slice(feature_name)
            feature_slice_model_output = self.used_features.get_slice(
                feature_name, model_output=True)

            feature_estimate = estimate[:, feature_slice_model_output]
            feature_output = output[:, feature_slice]

            if feature.categorical:
                # Classificaion loss
                assert feature_slice.stop - feature_slice.start == 1, \
                    "Supporting only single categorical cross entropy for now."
                assert feature.output_dimension > output[:, feature_slice.start].max(), \
                    f"feature output_dim is {feature.output_dimension} while output contains " \
                    f"categories up to {output[:, feature_slice.start].max()}"
                weights = self.scaler.get_categorical_feature_weights(feature_name).to(output) \
                    if self.scaler else None

                # Classes probabilities dim goes last, so feature_estimate shape is
                # [batch, seq-len, num-classes]
                feature_estimate = feature_estimate.transpose(1, 2)
                feature_output = feature_output.transpose(1, 2)

                # loss += F.cross_entropy(
                #     feature_estimate[feature_mask].reshape(
                #         -1, feature_slice_model_output.stop - feature_slice_model_output.start),
                #     feature_output.long()[mask.transpose(1, 2)],
                #     weights
                # )
            else:
                # Regression loss
                # loss += F.mse_loss(
                #     feature_estimate[feature_mask], feature_output[feature_mask])
                ...

        return loss