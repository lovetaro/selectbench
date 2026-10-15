import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion


class ConditionalDDPM(EnVariationalDiffusion):
    """
    Conditional Diffusion Module.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.dynamics.update_pocket_coords

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice
        negligible in the loss. However, you compute it so that you see it when
        you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = \
            mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part.
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h

    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig,
                                           net_out_lig, gamma_0, epsilon=1e-10):

        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution
        # N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        squared_error = (eps_lig_x - net_lig_x) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch(squared_error, ligand['mask'])
        )

        # Compute delta indicator masks.
        # un-normalize
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]

        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1,
                                keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])

        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def sample_p_xh_given_z0(self, z0_lig, xh0_pocket, lig_mask, pocket_mask,
                             batch_size, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _ = self.dynamics(
            z0_lig, xh0_pocket, t_zeros, lig_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask, fix_noise)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])

        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        # h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_normal(self, *args):
        raise NotImplementedError("Has been replaced by sample_normal_zero_com()")

    def sample_normal_zero_com(self, mu_lig, xh0_pocket, sigma, lig_mask,
                               pocket_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        out_lig = mu_lig + sigma[lig_mask] * eps_lig

        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        out_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(out_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return out_lig, xh_pocket

    def noised_representation(self, xh_lig, xh0_pocket, lig_mask, pocket_mask,
                              gamma_t):
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        z_t_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(z_t_lig[:, :self.n_dims],
                                   xh_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return z_t_lig, xh_pocket, eps_lig

    def log_pN(self, N_lig, N_pocket):
        """
        Prior on the sample size for computing
        log p(x,h,N) = log p(x,h|N) + log p(N), where log p(x,h|N) is the
        model's output
        Args:
            N: array of sample sizes
        Returns:
            log p(N)
        """
        log_pN = self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def forward(self, ligand, pocket, return_info=False):
        """
        Computes the loss and NLL terms
        """
        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        # Likelihood change due to normalization
        # if self.vnode_idx is not None:
        #     delta_log_px = self.delta_log_px(ligand['size'] - ligand['num_virtual_atoms'] + pocket['size'])
        # else:
        delta_log_px = self.delta_log_px(ligand['size'])

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float()
        s_int = t_int - 1  # previous timestep

        # Masks: important to compute log p(x | z0).
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)

        # Neural net prediction.
        net_out_lig, _ = self.dynamics(
            z_t_lig, xh_pocket, t, ligand['mask'], pocket['mask'])

        # For LJ loss term
        # xh_lig_hat does not need to be zero-centered as it is only used for
        # computing relative distances
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                                  ligand['mask'])

        # Compute the L2 error.
        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand['mask'])

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t_lig.size() == SNR_weight.size()

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand['size'], device=error_t_lig.device)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1).
        # Should be close to zero.
        kl_prior = self.kl_prior(xh0_lig, ligand['mask'], ligand['size'])

        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)

            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                              t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

            # apply t_is_zero mask
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()

        else:
            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            z_0_lig, xh_pocket, eps_0_lig = \
                self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                           pocket['mask'], gamma_0)

            net_out_0_lig, _ = self.dynamics(
                z_0_lig, xh_pocket, t_zeros, ligand['mask'], pocket['mask'])

            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        # sample size prior
        log_pN = self.log_pN(ligand['size'], pocket['size'])

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'],
                dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'],
                dim=0).mean(),
        }
        loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), SNR_weight,
                      loss_0_x_ligand, torch.tensor(0.0), loss_0_h,
                      neg_log_constants, kl_prior, log_pN,
                      t_int.squeeze(), xh_lig_hat)
        return (*loss_terms, info) if return_info else loss_terms
    
    def partially_noised_ligand(self, ligand, pocket, noising_steps):
        """
        Partially noises a ligand to be later denoised.
        """

        # Inflate timestep into an array
        t_int = torch.ones(size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float() * noising_steps

        # Normalize t to [0, 1].
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)
            
        return z_t_lig, xh_pocket, eps_t_lig

    def diversify(self, ligand, pocket, noising_steps):
        """
        Diversifies a set of ligands via noise-denoising
        """

        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        z_lig, xh_pocket, _ = self.partially_noised_ligand(ligand, pocket, noising_steps)

        timesteps = self.T
        n_samples = len(pocket['size'])
        device = pocket['x'].device

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = ligand['mask']

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.

        for s in reversed(range(0, noising_steps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig.detach(), xh_pocket.detach(), lig_mask, pocket['mask'])

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Overwrite last frame with the resulting x and h.
        out_lig = torch.cat([x_lig, h_lig], dim=1)
        out_pocket = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig, out_pocket, lig_mask, pocket['mask']


    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh

    def sample_p_zt_given_zs(self, zs_lig, xh0_pocket, ligand_mask, pocket_mask,
                             gamma_t, gamma_s, fix_noise=False):
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        zt_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma_t_given_s, ligand_mask, pocket_mask,
            fix_noise)

        return zt_lig, xh0_pocket

    def sample_p_zs_given_zt(self, s, t, zt_lig, xh0_pocket, ligand_mask,
                             pocket_mask, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # Neural net prediction.
        eps_t_lig, _ = self.dynamics(
            zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)

        # Compute mu for p(zs | zt).
        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * \
                 eps_t_lig

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand_mask, pocket_mask, fix_noise)

        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)

        return zs_lig, xh0_pocket

    def sample_combined_position_feature_noise(self, lig_indices, xh0_pocket,
                                               pocket_indices):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise
        for z_h.
        """
        raise NotImplementedError("Use sample_normal_zero_com() instead.")

    def sample(self, *args):
        raise NotImplementedError("Conditional model does not support sampling "
                                  "without given pocket.")

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):
        """
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        n_samples = len(pocket['size'])
        device = pocket['x'].device

        _, pocket = self.normalize(pocket=pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)

        # Sample from Normal distribution in the pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask'])

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'])

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = \
                    self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket['mask'])

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask']

    @torch.no_grad()
    def compose_sample_given_pocket(
            self, pocket_on, pocket_off, num_nodes_lig,
            guidance_weight=1.0, reverse_method='DDPM', inv_temp=1.0,
            beta_on=None, beta_off=None,
            resample=False, ess_thresh=0.0, resample_thresh=0.6,
            dw_clip=0.0, prob_floor=0.0, resample_v=1,
            rescale_prior=True,
            return_frames=1, timesteps=None):
        """Selective sampling: product-of-experts guidance over one on-target
        and K≥1 off-target pockets (Prop. D.5 in fkc.pdf).

        ``pocket_off`` may be a single pocket dict OR a list of K pocket dicts.
        With K off-targets, the scalar ``beta_off`` is split equally — each
        off-target receives weight ``beta_off / K`` — so passing
        (β_on=7, β_off=−2) with K=3 yields per-target exponents
        (+7, −2/3, −2/3, −2/3). Defaults (β_on, β_off) = (1+w, −w) reproduce
        standard CFG.

        At each step ``self.dynamics`` is called K+1 times (once per pocket);
        the combined noise prediction is

            eps_lig_guided = β_on · eps_on + Σ_k (β_off/K) · eps_off^k.

        reverse_method='DDPM' — spiritual CFG with the standard posterior step.
        reverse_method='SDE'  — Itô SDE step + FKC importance weights (dw_traj),
                                 generalized to N targets via Prop. D.5.

        Each off-target pocket must already be superimposed onto pocket_on in
        3D (selectbench case_prep handles this for the 1:1 case; for K>1 the
        same superposition must be done for every off PDB upstream).

        Returns (xh_lig, xh_pocket, lig_mask, pocket_on_mask, dw_traj).
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        if beta_on is None:
            beta_on = 1.0 + guidance_weight
        if beta_off is None:
            beta_off = -guidance_weight

        # Normalise off-target argument to a list, then split beta_off equally.
        pockets_off = pocket_off if isinstance(pocket_off, (list, tuple)) \
            else [pocket_off]
        K = len(pockets_off)
        assert K >= 1
        betas_off = [beta_off / K] * K

        n_samples = len(pocket_on['size'])
        device = pocket_on['x'].device

        _, pocket_on = self.normalize(pocket=pocket_on)
        pockets_off = [self.normalize(pocket=p)[1] for p in pockets_off]

        # Align all pockets on the on-target CoM so the same zt_lig makes
        # sense in every dynamics call. Each pocket_off must already be
        # roughly structurally aligned onto pocket_on (selectbench case_prep
        # handles this by superimposing each off PDB onto the on PDB).
        com_on = scatter_mean(pocket_on['x'], pocket_on['mask'], dim=0)
        pocket_on['x'] = pocket_on['x'] - com_on[pocket_on['mask']]
        for p_off in pockets_off:
            p_off['x'] = p_off['x'] - com_on[p_off['mask']]

        xh0_pocket_on = torch.cat([pocket_on['x'], pocket_on['one_hot']], dim=1)
        xh0_pockets_off = [
            torch.cat([p['x'], p['one_hot']], dim=1) for p in pockets_off
        ]
        pocket_off_masks = [p['mask'] for p in pockets_off]

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)

        # Initial ligand ~ N(pocket_on_CoM, 1), then ligand-zero-CoM projection
        # (same as sample_given_pocket but pocket_on CoM is now 0).
        mu_lig_x = torch.zeros((n_samples, self.n_dims), device=device)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket_on['size']).unsqueeze(1)

        z_lig, xh_pocket_on = self.sample_normal_zero_com(
            mu_lig, xh0_pocket_on, sigma, lig_mask, pocket_on['mask'])
        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # Tempered-prior rescaling (SDE path only). The composed SDE drift is
        # inv_temp·g²·(β_on·s_on + Σ β_off·s_off), so at t=T the effective score
        # field is s = inv_temp·(β_on+β_off) times the standard one — consistent
        # with starting from N(0, I/s) rather than N(0, I). Rescale the position
        # channels only (the zero-CoM Gaussian prior); scalar division preserves
        # zero-CoM. h channels and the DDPM path are unchanged. No-op for
        # standard CFG (β_on+β_off = 1, inv_temp = 1 ⇒ s ≈ 1).
        # Disable with rescale_prior=False to start from the plain N(0, I) prior.
        if reverse_method == 'SDE' and rescale_prior:
            s_prior = inv_temp * (beta_on + sum(betas_off))
            if s_prior > 0 and abs(s_prior - 1.0) > 1e-8:
                z_lig[:, :self.n_dims] = \
                    z_lig[:, :self.n_dims] / float(s_prior ** 0.5)

        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket_on.size(),
                                 device=device)

        # SMC log-weights (SDE only). Equal atoms-per-sample assumed.
        log_w = torch.zeros(n_samples, device=device)
        n_atoms_per_sample = z_lig.shape[0] // n_samples
        dw_traj = []

        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket_on, dw_step = self._compose_step(
                s_array, t_array, z_lig, xh_pocket_on, xh0_pockets_off,
                lig_mask, pocket_on['mask'], pocket_off_masks,
                reverse_method, inv_temp, beta_on, betas_off)

            if dw_step is not None:
                if dw_clip > 0:
                    dw_step = torch.clamp(dw_step, -dw_clip, dw_clip)
                dw_traj.append(dw_step.cpu())
                log_w = log_w + dw_step
                t_frac = (s + 1) / timesteps
                if resample and t_frac >= (1 - resample_thresh):
                    do_resample = True
                    if ess_thresh > 0:
                        log_w_norm = log_w - torch.logsumexp(log_w, dim=0)
                        ess = torch.exp(-torch.logsumexp(2 * log_w_norm, dim=0)).item()
                        do_resample = ess < ess_thresh
                    if do_resample:
                        w = torch.softmax(log_w, dim=0)
                        if prob_floor > 0:
                            w = torch.clamp(w, min=prob_floor)
                            w = w / w.sum()
                        u = (torch.rand(1, device=device)
                             + torch.arange(n_samples, device=device).float() / n_samples) % 1
                        cdf = torch.cumsum(w, dim=0)
                        new_ids = torch.bucketize(u, cdf).clamp(0, n_samples - 1)
                        if resample_v:
                            z_lig = torch.cat(
                                [z_lig[k * n_atoms_per_sample:(k + 1) * n_atoms_per_sample].clone()
                                 for k in new_ids.tolist()], dim=0)
                        else:
                            new_pos = torch.cat(
                                [z_lig[k * n_atoms_per_sample:(k + 1) * n_atoms_per_sample,
                                       :self.n_dims].clone()
                                 for k in new_ids.tolist()], dim=0)
                            z_lig = torch.cat([new_pos, z_lig[:, self.n_dims:]], dim=1)
                        log_w = torch.zeros(n_samples, device=device)

            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = self.unnormalize_z(z_lig, xh_pocket_on)

        # Finally sample p(x, h | z_0)
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket_on, lig_mask, pocket_on['mask'], n_samples)
        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket_on['mask'])

        # Translate back to the original on-target pocket frame
        x_lig = x_lig + com_on[lig_mask]
        x_pocket = x_pocket + com_on[pocket_on['mask']]

        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket_on['mask'], dw_traj

    def _compose_step(self, s, t, zt_lig, xh0_pocket_on, xh0_pockets_off,
                      lig_mask, pocket_on_mask, pocket_off_masks,
                      reverse_method, inv_temp, beta_on, betas_off):
        """One denoising step with product-of-experts guidance over one
        on-target pocket and K off-target pockets (Prop. D.5).

        For K=1 this reduces to the original on/off CFG step bit-for-bit:
        the noise-combination and FKC-weight expressions are written in the
        same operation order as the prior 2-target form so a single off-target
        run is unaffected by this generalization.
        """
        K = len(xh0_pockets_off)
        assert K >= 1
        assert len(pocket_off_masks) == K
        assert len(betas_off) == K

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # K+1 eps predictions (pocket is clean and fixed in every call — matches
        # how ConditionalDDPM was trained). On-target first to match the prior
        # call order for K=1.
        eps_lig_on, _ = self.dynamics(
            zt_lig, xh0_pocket_on, t, lig_mask, pocket_on_mask)
        eps_lig_offs = [
            self.dynamics(zt_lig, xh0_off, t, lig_mask, off_mask)[0]
            for xh0_off, off_mask in zip(xh0_pockets_off, pocket_off_masks)
        ]

        # eps_lig_guided = β_on·eps_on + Σ_k β_off^k·eps_off^k
        # For K=1 this reduces to (β_on·eps_on) + (β_off·eps_off), matching
        # the prior expression's evaluation order.
        eps_lig_guided = beta_on * eps_lig_on
        for b_k, eps_k in zip(betas_off, eps_lig_offs):
            eps_lig_guided = eps_lig_guided + b_k * eps_k
        dw_lig = None

        if reverse_method == 'DDPM':
            mu_lig = zt_lig / alpha_t_given_s[lig_mask] - \
                     (sigma2_t_given_s / alpha_t_given_s / sigma_t)[lig_mask] * \
                     eps_lig_guided
            sigma_step = sigma_t_given_s * sigma_s / sigma_t
            zs_lig, zs_pocket_on = self.sample_normal_zero_com(
                mu_lig, xh0_pocket_on, sigma_step, lig_mask, pocket_on_mask)

        elif reverse_method == 'SDE':
            dt = 1.0 / self.T
            gamma_prime = self._gamma_derivative(t)
            sigma_t_sq = self.inflate_batch_array(torch.sigmoid(gamma_t), zt_lig)
            gamma_prime_atom = self.inflate_batch_array(gamma_prime, zt_lig)

            g_sq = sigma_t_sq[lig_mask] * gamma_prime_atom[lig_mask]
            g = torch.sqrt(g_sq)
            dlog_alpha_dt = -0.5 * sigma_t_sq[lig_mask] * gamma_prime_atom[lig_mask]
            f_x = dlog_alpha_dt * zt_lig

            score_on = -eps_lig_on / sigma_t[lig_mask]
            score_offs = [-eps_k / sigma_t[lig_mask] for eps_k in eps_lig_offs]
            score_guided = beta_on * score_on
            for b_k, s_k in zip(betas_off, score_offs):
                score_guided = score_guided + b_k * s_k

            eps_sde = torch.randn_like(zt_lig)
            zs_lig = zt_lig + dt * (-f_x + inv_temp * g_sq * score_guided) \
                     + g * (dt ** 0.5) * eps_sde

            # Re-project ligand to zero-CoM; shift pocket by the same vector
            # to preserve relative geometry (ConditionalDDPM convention).
            zs_lig_pos, zs_pocket_pos = self.remove_mean_batch(
                zs_lig[:, :self.n_dims], xh0_pocket_on[:, :self.n_dims].clone(),
                lig_mask, pocket_on_mask)
            zs_lig = torch.cat([zs_lig_pos, zs_lig[:, self.n_dims:]], dim=1)
            zs_pocket_on = torch.cat(
                [zs_pocket_pos, xh0_pocket_on[:, self.n_dims:]], dim=1)

            # FKC log-weight, Prop D.5 (Target Score Product SDE):
            #   dw = [n·dlogα/dt·(Σ_i a_i − 1)
            #         + g²·(Σ_{i<j} a_i·a_j·⟨s_i,s_j⟩
            #               + Σ_i a_i(a_i−1)/2·‖s_i‖²)] · dt
            # where i ∈ {on, off^1, …, off^K} and a_i = inv_temp·β_i.
            #
            # Accumulation order (cross terms first, then self-norms; on-target
            # self-norm immediately after the on-target's cross terms) is chosen
            # so that for K=1 every multiply/add lines up with the prior
            # 2-target formula and produces bit-identical dw.
            ndim = self.n_dims
            a_on = inv_temp * beta_on
            a_offs = [inv_temp * b for b in betas_off]
            a_sum = a_on
            for a_k in a_offs:
                a_sum = a_sum + a_k

            dlog_a = dlog_alpha_dt[:, :1]
            g_sq_s = g_sq[:, :1]

            cross_on_offs = [
                (score_on[:, :ndim] * s_k[:, :ndim]).sum(-1, keepdim=True)
                for s_k in score_offs
            ]

            # Cross terms: on×off first (matches prior code for K=1)…
            quad = a_on * a_offs[0] * cross_on_offs[0]
            for k in range(1, K):
                quad = quad + a_on * a_offs[k] * cross_on_offs[k]
            # …then off×off cross terms (only present for K≥2).
            for j in range(K):
                for k in range(j + 1, K):
                    cjk = (score_offs[j][:, :ndim]
                           * score_offs[k][:, :ndim]).sum(-1, keepdim=True)
                    quad = quad + a_offs[j] * a_offs[k] * cjk
            # Self-norms: on, then each off (matches prior code's order for K=1).
            quad = quad + a_on * (a_on - 1) * 0.5 \
                   * (score_on[:, :ndim] ** 2).sum(-1, keepdim=True)
            for a_k, s_k in zip(a_offs, score_offs):
                quad = quad + a_k * (a_k - 1) * 0.5 \
                       * (s_k[:, :ndim] ** 2).sum(-1, keepdim=True)

            dw_per_atom = (
                ndim * dlog_a * (a_sum - 1) + g_sq_s * quad
            ) * dt
            dw_lig = scatter_add(dw_per_atom.squeeze(-1), lig_mask, dim=0)
        else:
            raise ValueError(reverse_method)

        return zs_lig, zs_pocket_on, dw_lig

    @torch.no_grad()
    def inpaint(self, ligand, pocket, lig_fixed, resamplings=1, return_frames=1,
                timesteps=None, center='ligand'):
        """
        Draw samples from the generative model while fixing parts of the input.
        Optionally, return intermediate states for visualization purposes.
        Inspired by Algorithm 1 in:
        Lugmayr, Andreas, et al.
        "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition. 2022.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        n_samples = len(ligand['size'])
        device = pocket['x'].device

        # Normalize
        ligand, pocket = self.normalize(ligand, pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        com_pocket_0 = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        xh0_ligand = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh_ligand = xh0_ligand.clone()

        # Center initial system, subtract COM of known parts
        if center == 'ligand':
            mean_known = scatter_mean(ligand['x'][lig_fixed.bool().view(-1)],
                                      ligand['mask'][lig_fixed.bool().view(-1)],
                                      dim=0)
        elif center == 'pocket':
            mean_known = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        else:
            raise NotImplementedError(
                f"Centering option {center} not implemented")

        # Sample from Normal distribution in the ligand center
        mu_lig_x = mean_known
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[ligand['mask']]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand['mask'], pocket['mask'])

        # Output tensors
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Iteratively sample with resampling iterations
        for s in reversed(range(0, timesteps)):

            # resampling iterations
            for u in range(resamplings):

                # Denoise one time step: t -> s
                s_array = torch.full((n_samples, 1), fill_value=s,
                                     device=device)
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                gamma_t = self.gamma(t_array)
                gamma_s = self.gamma(s_array)

                # sample inpainted part
                z_lig_unknown, xh_pocket = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, xh_pocket, ligand['mask'],
                    pocket['mask'])

                # sample known nodes from the input
                com_pocket = scatter_mean(xh_pocket[:, :self.n_dims],
                                          pocket['mask'], dim=0)
                xh_ligand[:, :self.n_dims] = \
                    ligand['x'] + (com_pocket - com_pocket_0)[ligand['mask']]
                z_lig_known, xh_pocket, _ = self.noised_representation(
                    xh_ligand, xh_pocket, ligand['mask'], pocket['mask'],
                    gamma_s)

                # move center of mass of the noised part to the center of mass
                # of the corresponding denoised part before combining them
                # -> the resulting system should be COM-free
                com_noised = scatter_mean(
                    z_lig_known[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                com_denoised = scatter_mean(
                    z_lig_unknown[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                dx = com_denoised - com_noised
                z_lig_known[:, :self.n_dims] = z_lig_known[:, :self.n_dims] + dx[ligand['mask']]
                xh_pocket[:, :self.n_dims] = xh_pocket[:, :self.n_dims] + dx[pocket['mask']]

                # combine
                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (
                            1 - lig_fixed)

                if u < resamplings - 1:
                    # Noise the sample
                    z_lig, xh_pocket = self.sample_p_zt_given_zs(
                        z_lig, xh_pocket, ligand['mask'], pocket['mask'],
                        gamma_t, gamma_s)

                # save frame at the end of a resampling cycle
                if u == resamplings - 1:
                    if (s * return_frames) % timesteps == 0:
                        idx = (s * return_frames) // timesteps

                        out_lig[idx], out_pocket[idx] = \
                            self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, ligand['mask'], pocket['mask'], n_samples)

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), ligand['mask'], \
               pocket['mask']

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):

        # Just subtract the center of mass of the sampled part
        mean = scatter_mean(x_lig, lig_indices, dim=0)

        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        return x_lig, x_pocket


# ------------------------------------------------------------------------------
# The same model without subspace-trick
# ------------------------------------------------------------------------------
class SimpleConditionalDDPM(ConditionalDDPM):
    """
    Simpler conditional diffusion module without subspace-trick.
    - rotational equivariance is guaranteed by construction
    - translationally equivariant likelihood is achieved by first mapping
      samples to a space where the context is COM-free and evaluating the
      likelihood there
    - molecule generation is equivariant because we can first sample in the
      space where the context is COM-free and translate the whole system back to
      the original position of the context later
    """
    def subspace_dimensionality(self, input_size):
        """ Override because we don't use the linear subspace anymore. """
        return input_size * self.n_dims

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):
        """ Hacky way of removing the centering steps without changing too much
        code. """
        return x_lig, x_pocket

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        return

    def forward(self, ligand, pocket, return_info=False):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        ligand['x'] = ligand['x'] - pocket_com[ligand['mask']]
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).forward(
            ligand, pocket, return_info)

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).sample_given_pocket(
            pocket, num_nodes_lig, return_frames, timesteps)
