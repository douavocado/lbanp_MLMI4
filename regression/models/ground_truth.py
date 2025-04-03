import torch
import torch.nn as nn
from attrdict import AttrDict
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class GaussianProcess(nn.Module):
    """
    Gaussian Process model with squared exponential kernel
    """
    def __init__(self, dim_x, dim_y):
        super(GaussianProcess, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

    def predict(self, xc, yc, xt, cov):
       # cov is the full covariance matrix of all combined xt and xc points
        # mean for the joint is zero vector
        # want to calculate the conditional mean cov conditioned on (xc, yc) values for xc
        # Combine context and target points
        
        # Get dimensions
        batch_size = xc.shape[0]
        n_context = xc.shape[1]
        n_target = xt.shape[1]
        
        # Extract the relevant parts of the covariance matrix
        # cov is the full covariance matrix [B, Nc+Nt, Nc+Nt]
        K_cc = cov[:, :n_context, :n_context]  # [B, Nc, Nc]
        K_ct = cov[:, :n_context, n_context:]  # [B, Nc, Nt]
        K_tc = cov[:, n_context:, :n_context]  # [B, Nt, Nc]
        K_tt = cov[:, n_context:, n_context:]  # [B, Nt, Nt]
        
        # Add small jitter to diagonal for numerical stability
        jitter = 1e-6 * torch.eye(n_context, device=xc.device).unsqueeze(0).repeat(batch_size, 1, 1)
        K_cc = K_cc + jitter
        K_tt = K_tt + 1e-6 * torch.eye(n_target, device=xt.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Compute conditional mean: K_tc @ K_cc^(-1) @ yc
        # First solve the system K_cc @ alpha = yc for alpha
        yc_flat = yc.squeeze(-1)  # [B, Nc]
        
        # Use torch.linalg.solve for better numerical stability
        alpha = torch.linalg.solve(K_cc, yc_flat.unsqueeze(-1)).squeeze(-1)  # [B, Nc]
        
        # Compute conditional mean
        cond_mean = torch.bmm(K_tc, alpha.unsqueeze(-1)).squeeze(-1)  # [B, Nt]
        
        # Compute conditional covariance: K_tt - K_tc @ K_cc^(-1) @ K_ct
        # Use the same approach with torch.linalg.solve
        conditional_cov = K_tt - torch.bmm(K_tc, torch.linalg.solve(K_cc, K_ct))  # [B, Nt, Nt]
        
        # Add small jitter to diagonal for numerical stability
        #jitter = 1e-6 * torch.eye(n_target, device=xt.device).unsqueeze(0).repeat(batch_size, 1, 1)
        #conditional_cov = conditional_cov + jitter
        # progressively add more jitter until we get a valid multivariate normal distribution
        # Ensure the covariance matrix is positive definite
        # Start with a small jitter and increase if needed
        jitter_scale = 1e-6
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Try to create the distribution with current jitter
                _ = MultivariateNormal(cond_mean, covariance_matrix=conditional_cov)
                # If successful, break the loop
                break
            except RuntimeError:
                # If failed, increase jitter and try again
                jitter_scale *= 10
                jitter = jitter_scale * torch.eye(n_target, device=xt.device).unsqueeze(0).repeat(batch_size, 1, 1)
                conditional_cov = conditional_cov + jitter
                if attempt == max_attempts - 1:
                    # If we've reached max attempts, use diagonal covariance as fallback
                    conditional_cov = torch.diag_embed(torch.diagonal(conditional_cov, dim1=1, dim2=2))
        return MultivariateNormal(cond_mean, covariance_matrix=conditional_cov)


    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        
        # Get length and scale from batch
        length = batch.length # shape (batch_size,)
        scale = batch.scale # shape (batch_size,)

        # Make predictions
        pred_tar = self.predict(batch.xc, batch.yc, batch.xt, batch.cov)

        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt.squeeze(-1)).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt.squeeze(-1))
        outs.loss = -outs.tar_ll

        return outs
