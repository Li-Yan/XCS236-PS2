import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        # 1. encoding the input
        qm, qv = self.enc(x)
        # 2. sample z given the Mean and Variance
        z = ut.sample_gaussian(qm,qv)

        # 3. expand z_prior_m and z_prior_v to the same shape as qm and qv
        pm = self.z_prior_m.expand(qm.shape)
        pv = self.z_prior_v.expand(qv.shape)
        # 4. calculate KL divergence to prior
        kl = torch.mean(ut.kl_normal(qm, qv, pm, pv))

        # 5. decode z and tries to reconstruct the original input
        x_logits = self.dec(z)
        # 6. calculates the reconstruction loss
        rec = -1.0 * torch.mean(ut.log_bernoulli_with_logits(x, x_logits))

        nelbo = kl + rec

        return nelbo, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###
        # 1. encoding the input
        qm, qv = self.enc(x)
        # 2. duplicates qm,qv iw times along a new dimension.
        multi_qm = ut.duplicate(qm, iw)
        multi_qv = ut.duplicate(qv, iw)
        # 3. sample z given the Mean and Variance
        z = ut.sample_gaussian(multi_qm, multi_qv)

        # 4. calcuate KL
        pm = self.z_prior_m.expand(qm.shape)
        pv = self.z_prior_v.expand(qv.shape)
        kl = torch.mean(ut.kl_normal(qm, qv, pm, pv))

        # 5. duplicates the input data x iw times
        multi_x = ut.duplicate(x, iw)
        # 6. calculate rec
        x_logits = self.dec(z)
        recs = ut.log_bernoulli_with_logits(multi_x, x_logits)
        rec = -1.0 * torch.mean(recs)

        # 7. calculate z_posteriors
        x_posteriors = recs
        z_posteriors = ut.log_normal(z, multi_qm, multi_qv)

        # 7. expands Mean and Variance of z_prior to match the shape of multi_qm and multi_qv
        multi_pm = self.z_prior_m.expand(multi_qm.shape)
        multi_pv = self.z_prior_v.expand(multi_qv.shape)
        # 8. calculate z_priors
        z_priors = ut.log_normal(z, multi_pm, multi_pv)

        # 9. calculate niwae
        log_ratios = z_priors + x_posteriors - z_posteriors
        niwae = -1.0 * torch.mean(ut.log_mean_exp(log_ratios.reshape(iw, x.shape[0]), 0))

        return niwae, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
