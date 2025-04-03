import torch
from torch.distributions import Normal
from einops import rearrange, repeat

import itertools
import math
import sys
import os
from data.gp import RBFKernelNonRandom

def ar_sample(model, xc, yc, xt, num_samples=1, seed=None, smoothed=False, batch_size_targets=1):
    """
    Performs autoregressive sampling using an LBANP model.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        num_samples: Number of independent autoregressive samples to generate
        seed: Random seed for reproducibility
        smoothed: Whether to outputs smoothed samples too
        batch_size_targets: Number of target points to process at once in each autoregressive step
        
    Returns:
        sampled_y: Sampled y values [num_samples, batch_size, num_target_points, dim_y]
        sampled_y_noiseless: Noiseless y values [batch_size, num_target_points, dim_y]
        pred_dists: list of torch.distribution.Normal objects for the autoregressively sampled y
        smoothed_pred_dists: list of torch.distribution.Normal objects for the autoregressively smoothed sampled y
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    
    # Ensure batch_size_targets is valid
    batch_size_targets = min(batch_size_targets, num_targets)
    
    # Initialize containers for results
    use_x = repeat(xt.clone(), 'b n d -> s b n d', s=num_samples)
    sampled_y = torch.zeros(num_samples, batch_size, num_targets, dim_y, device=xt.device)
    
    # Generate multiple independent autoregressive samples
    # Start with the original context for each sample
    current_xc = repeat(xc.clone(), 'b n d -> (s b) n d', s=num_samples)
    current_yc = repeat(yc.clone(), 'b n d -> (s b) n d', s=num_samples)
    
    # Create random permutations for each sample
    permutations = []
    for s in range(num_samples):
        perm = torch.randperm(num_targets)
        permutations.append(perm)
    
    # Track original indices for each sample to restore original order later
    original_indices = torch.zeros(num_samples, num_targets, dtype=torch.long)
    for s, perm in enumerate(permutations):
        for i, p in enumerate(perm):
            original_indices[s, p] = i
    
    pred_dists = [None] * num_targets  # Initialize with None placeholders
    
    # Process target points in batches according to random order for each sample
    num_batches = (num_targets + batch_size_targets - 1) // batch_size_targets
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size_targets
        end_idx = min((batch_idx + 1) * batch_size_targets, num_targets)
        batch_size_current = end_idx - start_idx
        
        # For each sample, get the current batch of points in its permuted order
        current_xt_list = []
        batch_indices = []  # Store the actual indices for each sample
        
        for s in range(num_samples):
            # Get the indices of the current batch in the permuted order for this sample
            sample_indices = permutations[s][start_idx:end_idx]
            batch_indices.append(sample_indices)
            
            # Gather the corresponding target points
            sample_xt = use_x[s:s+1, :, sample_indices, :]
            current_xt_list.append(sample_xt)
        
        # Concatenate across samples and reshape
        current_xt = torch.cat(current_xt_list, dim=0)
        current_xt = rearrange(current_xt, 's b n d -> (s b) n d')
        
        # Get prediction distribution for the current batch of targets
        pred_dist = model.predict(current_xc, current_yc, current_xt)
        
        # Sample from the distribution
        current_yt = pred_dist.sample()
        
        # Store the sampled values and prediction distributions in the correct original positions
        for s in range(num_samples):
            for i, idx in enumerate(batch_indices[s]):
                pos = original_indices[s, idx]
                if start_idx <= pos < end_idx:  # This is a position in the current batch in original order
                    # Store the prediction distribution for this position
                    if pred_dists[pos] is None:
                        # Extract the distribution for this specific target point
                        mean = pred_dist.mean[s*batch_size:(s+1)*batch_size, i:i+1, :]
                        stddev = pred_dist.stddev[s*batch_size:(s+1)*batch_size, i:i+1, :]
                        pred_dists[pos] = Normal(mean, stddev)
                
                # Extract and store the sampled value for this sample
                s_idx = s * batch_size
                e_idx = (s + 1) * batch_size
                sampled_y[s, :, idx:idx+1, :] = rearrange(
                    current_yt[s_idx:e_idx, i:i+1, :], 
                    'b n d -> b n d'
                )
        
        # Add the new points to the context
        current_xc = torch.cat([current_xc, current_xt], dim=1)
        current_yc = torch.cat([current_yc, current_yt], dim=1)
    
    if not smoothed:
        return sampled_y, None, pred_dists, None
        
    # Get noiseless predictions using the final context for all target points in original order
    smoothed_pred_dists = model.predict(current_xc, current_yc, rearrange(use_x, 's b n d -> (s b) n d'))
    sampled_y_noiseless = rearrange(smoothed_pred_dists.mean, '(s b) n d -> s b n d', s=num_samples)
    smoothed_pred_dists = [
        Normal(
            smoothed_pred_dists.mean[:, i:i+1, :], 
            smoothed_pred_dists.stddev[:, i:i+1, :]
        ) 
        for i in range(use_x.size(2))
    ]
    
    return sampled_y, sampled_y_noiseless, pred_dists, smoothed_pred_dists

def no_cheat_ar_log_likelihood(model, xc, yc, xt, yt, num_samples=20, seed=None, smooth=False, covariance_est="scm", nu_p=2, batch_size_targets=1):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling WITHOUT feeding target labels into context set.
    In this sense the model does not cheat.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        num_samples: Number of samples to take for the AR sampling
        seed: Random seed for reproducibility
        smooth: Whether to use smooth or non smooth samples for covariance matrix estimation/likelihood calculation.
        covariance_est: The method for estimating covariance matrix from samples. One of [scm, shrinkage, bayesian]
        nu_p: The degrees of freedom for the inverse wishart prior for bayesian covariance estimation
        batch_size_targets: Number of target points to process at once in each autoregressive step
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    sampled_y, sampled_y_noiseless, pred_dists, smoothed_pred_dists = ar_sample(model, xc, yc, xt, num_samples=num_samples, seed=seed, smoothed=smooth, batch_size_targets=batch_size_targets)
    if smooth:
        use_y = sampled_y_noiseless.clone() # shape (s, b, n, d) where s is num samples, b is batch size, n is target size, d is data dimension
    else:
        use_y = sampled_y.clone()
    
    # Estimate mean across samples, or can use the model prediction
    # for scm, we use empirical mean
    mean_y = torch.mean(use_y, dim=0)  # shape (b, n, d)
    batch_size, num_targets, dim_y = mean_y.shape

    if covariance_est == "scm":
        # just take sample covariance matrix (the unbiased version)
        # uses torch.cov to compute sample covariance matrix for every batch and every data dimension
        
        log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
        for b in range(batch_size):
            # Reshape data for this batch to (num_samples, num_targets*dim_y)
            # First, extract data for this batch
            batch_data = use_y[:, b]  # shape (num_samples, num_targets, dim_y)
            
            # Reshape to (num_samples, num_targets*dim_y)
            batch_data_flat = batch_data.reshape(num_samples, -1)
            
            # Compute sample covariance matrix using torch.cov
            # torch.cov expects input of shape (features, observations)
            # so we transpose batch_data_flat
            cov_matrix = torch.cov(batch_data_flat.T)  # shape (num_targets*dim_y, num_targets*dim_y)
            
            # Get mean for this batch
            mean_vector = mean_y[b].reshape(-1)  # Flatten to (num_targets*dim_y)
            
            # Ensure covariance matrix is symmetric and positive definite
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            
            # Add small diagonal term for numerical stability
            cov_matrix = cov_matrix + 1e-12 * torch.eye(cov_matrix.shape[0], device=use_y.device)
            
            try:
                # Create multivariate normal distribution
                mvn_dist = torch.distributions.MultivariateNormal(mean_vector, cov_matrix)
                
                # Calculate log probability of ground truth
                yt_flat = yt[b].reshape(-1)  # Flatten to (num_targets*dim_y)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # normalize by number of targets
            except:
                # Handle numerical issues
                print("Warning: Covariance matrix is not positive definite. Using fallback method of only taking diagonal entries")
                # Fallback to diagonal covariance as approximation
                diag_cov = torch.diag(cov_matrix)
                mvn_dist = torch.distributions.MultivariateNormal(
                    mean_vector, 
                    torch.diag(torch.clamp(diag_cov, min=1e-6))
                )
                yt_flat = yt[b].reshape(-1)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # normalize by number of targets
        
        return log_likelihood

    elif covariance_est == "bayesian":
        # use bayesian covariance estimation with inverse wishart prior with scale matrix a diagonal matrix with variances given by
        # model prediction on full target set (var_y)
        # Get model prediction on full target set to estimate variances
        pred_dist = model.predict(xc, yc, xt)
        var_y = pred_dist.variance  # Shape: [batch_size, num_targets, dim_y]
        
        if nu_p is None:
            nu_p = 1
        # Parameters for inverse Wishart prior
        nu = num_samples + dim_y + nu_p  # Degrees of freedom (nu > dim_y + 1)
        
        # Calculate log likelihood with Bayesian covariance estimation
        log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
        for b in range(batch_size):
            # Reshape data for this batch to (num_samples, num_targets*dim_y)
            batch_data = use_y[:, b]  # shape (num_samples, num_targets, dim_y)
            
            # Reshape to (num_samples, num_targets*dim_y)
            batch_data_flat = batch_data.reshape(num_samples, -1)
            
            # Compute sample covariance matrix using torch.cov
            # torch.cov expects input of shape (features, observations)
            sample_cov = torch.cov(batch_data_flat.T)  # shape (num_targets*dim_y, num_targets*dim_y)
            
            # Create diagonal scale matrix from model's predicted variances
            psi = torch.diag_embed(var_y[b].reshape(-1))  # Diagonal matrix of variances
            
            # Bayesian estimate: weighted combination of sample covariance and prior
            alpha = (num_samples - 1) / (num_samples - 1 + nu)
            bayesian_cov = alpha * sample_cov + (1 - alpha) * psi
            
            # Ensure symmetry and positive definiteness
            bayesian_cov = 0.5 * (bayesian_cov + bayesian_cov.T)
            bayesian_cov = bayesian_cov + 1e-12 * torch.eye(bayesian_cov.shape[0], device=use_y.device)
            
            try:
                # Create multivariate normal distribution with Bayesian covariance
                mvn_dist = torch.distributions.MultivariateNormal(mean_y[b].reshape(-1), bayesian_cov)
                
                # Calculate log probability of ground truth
                yt_flat = yt[b].reshape(-1)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # Normalize by number of targets
            except:
                # Handle numerical issues
                print("Warning: Bayesian covariance matrix is not positive definite. Using fallback method.")
                # Fallback to diagonal covariance
                diag_cov = torch.diag(bayesian_cov)
                mvn_dist = torch.distributions.MultivariateNormal(
                    mean_y[b].reshape(-1), 
                    torch.diag(torch.clamp(diag_cov, min=1e-6))
                )
                yt_flat = yt[b].reshape(-1)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets
        
        return log_likelihood

    elif covariance_est == "shrinkage":
        # use ledoit-wolf shrinkage estimator
        # Implement Ledoit-Wolf shrinkage estimator for covariance matrix
        # This method combines the sample covariance matrix with a structured estimator
        # to improve estimation when number of samples is limited
        
        # Estimate mean across samples
        mean_y = torch.mean(use_y, dim=0)  # shape (b, n, d)
        
        # Center the data
        centered_y = use_y - mean_y.unsqueeze(0)  # shape (s, b, n, d)
        
        # Calculate log likelihood with shrinkage covariance estimation
        log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
        for b in range(batch_size):
            # Reshape centered data for this batch to (num_samples, num_targets*dim_y)
            X = centered_y[:, b].reshape(num_samples, -1)  # shape (s, n*d)
            
            # Sample covariance matrix
            n = X.shape[0]
            sample_cov = torch.matmul(X.t(), X) / (n - 1)  # shape (n*d, n*d)
            
            # Target for shrinkage: diagonal matrix with average variance
            target = torch.diag(torch.diag(sample_cov))
            
            # Calculate optimal shrinkage intensity
            # Formula based on Ledoit-Wolf method
            var_sample_cov = torch.mean((torch.matmul(X.t(), X) / n - sample_cov) ** 2)
            
            # Calculate Frobenius norm of difference between sample cov and target
            cov_minus_target = sample_cov - target
            norm_squared = torch.sum(cov_minus_target ** 2)
            
            # Optimal shrinkage intensity (capped between 0 and 1)
            shrinkage = torch.clamp(var_sample_cov / norm_squared, 0.0, 1.0)
            
            # Apply shrinkage
            shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
            
            # Ensure symmetry and positive definiteness
            shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.t())
            shrunk_cov = shrunk_cov + 1e-6 * torch.eye(shrunk_cov.shape[0], device=use_y.device)
            
            try:
                # Create multivariate normal distribution with shrunk covariance
                mvn_dist = torch.distributions.MultivariateNormal(mean_y[b].reshape(-1), shrunk_cov)
                
                # Calculate log probability of ground truth
                yt_flat = yt[b].reshape(-1)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # Normalize by number of targets
            except:
                # Handle numerical issues
                print("Warning: Shrunk covariance matrix is not positive definite. Using fallback method.")
                # Fallback to diagonal covariance
                diag_cov = torch.diag(shrunk_cov)
                mvn_dist = torch.distributions.MultivariateNormal(
                    mean_y[b].reshape(-1), 
                    torch.diag(torch.clamp(diag_cov, min=1e-6))
                )
                yt_flat = yt[b].reshape(-1)
                log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets
        
        return log_likelihood
    else:
        raise NotImplementedError(f"{covariance_est} covariance estimation is not implemented for no_cheat_ar_log_likelihood")



def ar_log_likelihood_all_perm(model, xc, yc, xt, yt, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        seed: Random seed for reproducibility
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    num_samples = math.factorial(num_targets)

    # Storage for log likelihoods from all permutations
    all_permutation_log_likelihoods = torch.zeros(num_samples, batch_size, device=xt.device)
    
    if get_normal:
        # Get the log-likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    
    # Generate all permutations at once
    permutations = list(itertools.permutations(range(num_targets)))
    
    # Create expanded batches for all permutations
    expanded_xc = repeat(xc, 'b n d -> (s b) n d', s=num_samples)
    expanded_yc = repeat(yc, 'b n d -> (s b) n d', s=num_samples)
    
    # Track log-likelihoods for each permutation and store original ordering
    log_likelihoods_by_perm = torch.zeros(num_samples, batch_size, num_targets, device=xt.device)
    
    # Process one target point at a time across all permutations simultaneously
    for i in range(num_targets):
        # Gather the i-th target point from each permutation
        current_xt_list = []
        current_yt_list = []
        
        for s in range(num_samples):
            perm_idx = permutations[s][i]
            current_xt_list.append(xt[:, perm_idx:perm_idx+1, :])
            current_yt_list.append(yt[:, perm_idx:perm_idx+1, :])
        
        # Stack into batches [num_samples*batch_size, 1, dim]
        current_xt = torch.cat(current_xt_list, dim=0)
        current_yt = torch.cat(current_yt_list, dim=0)
        
        # Get prediction distribution for all current targets at once
        pred_dist = model.predict(expanded_xc, expanded_yc, current_xt)
        
        # Calculate log probabilities
        log_probs = pred_dist.log_prob(current_yt).sum(-1)  # [num_samples*batch_size, 1]
        
        # Reshape back to [num_samples, batch_size]
        log_probs = log_probs.view(num_samples, batch_size, 1).squeeze(-1)
        
        # Store the log-likelihoods in the original order for each permutation
        for s in range(num_samples):
            perm_idx = permutations[s][i]
            log_likelihoods_by_perm[s, :, perm_idx] = log_probs[s]
        
        # Update context sets for each permutation
        expanded_xc = torch.cat([expanded_xc, current_xt], dim=1)
        expanded_yc = torch.cat([expanded_yc, current_yt], dim=1)
    
    # Average log-likelihoods across target points for each permutation
    all_permutation_log_likelihoods = torch.mean(log_likelihoods_by_perm, dim=-1)
    
    # Average in likelihood space rather than log likelihood space
    log_likelihood = torch.logsumexp(all_permutation_log_likelihoods, dim=0) - torch.log(torch.tensor(num_samples, device=all_permutation_log_likelihoods.device))
    
    if get_normal:
        print("Context size: ", xc.shape[1])
        print("Target size: ", xt.shape[1])
        print("Mean log-likelihood difference: ", torch.mean(log_likelihood - torch.mean(normal_log_likelihoods, dim=-1)))
        
    return log_likelihood


def ar_log_likelihood(model, xc, yc, xt, yt, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        seed: Random seed for reproducibility
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    
    # Initialize containers for results
    use_x = xt.clone()
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
    # Generate multiple independent autoregressive samples
    # Start with the original context for each sample
    current_xc = xc.clone()
    current_yc = yc.clone()

    if get_normal:
        # get the log_likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    
    # Process one target point at a time
    for i in range(num_targets):
        # Extract the current target point
        current_xt = use_x[:, i:i+1, :]
        current_yt = yt[:, i:i+1, :]
        
        # Get prediction distribution for the current target
        pred_dist = model.predict(current_xc, current_yc, current_xt)
        
        # Get log likelihood of the target point
        log_likelihood = pred_dist.log_prob(current_yt)
        
        # Store the log likelihood
        log_likelihoods[:, i] = log_likelihood[:,0,:].sum(-1)
        
        # Add the new point to the context
        current_xc = torch.cat([current_xc, current_xt], dim=1)
        current_yc = torch.cat([current_yc, current_yt], dim=1)
    
    log_likelihood = torch.mean(log_likelihoods, dim=-1)
    if get_normal:
        # print("context size: ", xc.shape[1])
        # print("target size: ", xt.shape[1])
        diff = log_likelihoods - normal_log_likelihoods
        # print("diff: ", diff[0])
        # also return dictionary containing the number of context points used and corresponding differences
        out_dic = {
            "log_likelihood": log_likelihoods.sum() / xt.shape[1],
            "normal_log_likelihood": normal_log_likelihoods.sum() / xt.shape[1],
            "diff": diff.sum() / xt.shape[1],
            "context_size": xc.shape[1],
            "target_size": xt.shape[1]
        }
        return log_likelihood, out_dic
        
    return log_likelihood


def ar_log_likelihood_mc(model, xc, yc, xt, yt, num_samples=5, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling.
    Uses multiple samples of different permutations of the target points to estimate the log likelihood.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        num_samples: Number of different permutations to sample
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood for comparison
        
    Returns:
        log_likelihood: Average log likelihood across different permutations
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    
    # Storage for log likelihoods from all permutations
    all_permutation_log_likelihoods = torch.zeros(num_samples, batch_size, device=xt.device)
    
    if get_normal:
        # Get the log-likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    # Generate all permutations at once
    permutations = [torch.randperm(num_targets) for _ in range(num_samples)]
    
    # Create expanded batches for all permutations
    expanded_xc = repeat(xc, 'b n d -> (s b) n d', s=num_samples)
    expanded_yc = repeat(yc, 'b n d -> (s b) n d', s=num_samples)
    
    # Track log-likelihoods for each permutation and store original ordering
    log_likelihoods_by_perm = torch.zeros(num_samples, batch_size, num_targets, device=xt.device)
    
    # Process one target point at a time across all permutations simultaneously
    for i in range(num_targets):
        # Gather the i-th target point from each permutation
        current_xt_list = []
        current_yt_list = []
        
        for s in range(num_samples):
            perm_idx = permutations[s][i]
            current_xt_list.append(xt[:, perm_idx:perm_idx+1, :])
            current_yt_list.append(yt[:, perm_idx:perm_idx+1, :])
        
        # Stack into batches [num_samples*batch_size, 1, dim]
        current_xt = torch.cat(current_xt_list, dim=0)
        current_yt = torch.cat(current_yt_list, dim=0)
        
        # Get prediction distribution for all current targets at once
        pred_dist = model.predict(expanded_xc, expanded_yc, current_xt)
        
        # Calculate log probabilities
        log_probs = pred_dist.log_prob(current_yt).sum(-1)  # [num_samples*batch_size, 1]
        
        # Reshape back to [num_samples, batch_size]
        log_probs = log_probs.view(num_samples, batch_size, 1).squeeze(-1)
        
        # Store the log-likelihoods in the original order for each permutation
        for s in range(num_samples):
            perm_idx = permutations[s][i]
            log_likelihoods_by_perm[s, :, perm_idx] = log_probs[s]
        
        # Update context sets for each permutation
        expanded_xc = torch.cat([expanded_xc, current_xt], dim=1)
        expanded_yc = torch.cat([expanded_yc, current_yt], dim=1)
    
    # Average log-likelihoods across target points for each permutation
    all_permutation_log_likelihoods = torch.mean(log_likelihoods_by_perm, dim=-1)
    
    # Average in likelihood space rather than log likelihood space
    log_likelihood = torch.logsumexp(all_permutation_log_likelihoods, dim=0) - torch.log(torch.tensor(num_samples, device=all_permutation_log_likelihoods.device))
    
    if get_normal:
        print("Context size: ", xc.shape[1])
        print("Target size: ", xt.shape[1])
        print("Mean log-likelihood difference: ", torch.mean(log_likelihood - torch.mean(normal_log_likelihoods, dim=-1)))
        
    return log_likelihood

def ar_log_likelihood_batch(model, xc, yc, xt, yt, ar_batches=2, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling
    with a specified number of autoregressive batches.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        ar_batches: Number of autoregressive steps to take (target points will be partitioned into this many batches)
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood for comparison
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    
    # Ensure ar_batches is valid
    ar_batches = min(ar_batches, num_targets)
    
    # Calculate targets per batch, handling uneven divisions
    targets_per_batch = [num_targets // ar_batches + (1 if i < num_targets % ar_batches else 0) 
                         for i in range(ar_batches)]
    
    # Initialize containers for results
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
    # Start with the original context for each batch
    current_xc = xc.clone()
    current_yc = yc.clone()

    if get_normal:
        # Get the log-likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    # Process targets in batches
    target_idx = 0
    for batch_idx in range(ar_batches):
        batch_size_current = targets_per_batch[batch_idx]
        
        # Get the current batch of target points
        batch_start = target_idx
        batch_end = target_idx + batch_size_current
        
        current_xt = xt[:, batch_start:batch_end, :]
        current_yt = yt[:, batch_start:batch_end, :]
        
        # Get prediction distribution for the current batch of targets
        pred_dist = model.predict(current_xc, current_yc, current_xt)
        
        # Get log likelihood of each target point in the batch
        log_probs = pred_dist.log_prob(current_yt)  # [batch_size, batch_size_current, dim_y]
        
        # Sum over output dimensions and store the log likelihoods
        log_likelihoods[:, batch_start:batch_end] = log_probs.sum(-1)
        
        # Add all points from this batch to the context before processing the next batch
        current_xc = torch.cat([current_xc, current_xt], dim=1)
        current_yc = torch.cat([current_yc, current_yt], dim=1)
        
        # Update target index for next batch
        target_idx = batch_end
    
    # Average log-likelihoods across target points
    log_likelihood = torch.mean(log_likelihoods, dim=-1)
    
    if get_normal:
        print("Context size: ", xc.shape[1])
        print("Target size: ", xt.shape[1])
        print("Number of AR batches: ", ar_batches)
        print("Targets per batch: ", targets_per_batch)
        diff = log_likelihoods - normal_log_likelihoods
        print("Mean log-likelihood difference: ", torch.mean(diff, dim=-1))
    
    return log_likelihood

def ar_log_likelihood_batch_clustered(model, xc, yc, xt, yt, ar_batches=2, num_clusters=2, seed=None, get_normal=False, max_kmeans_iter=10):
    """
    Computes the log likelihood using clustered batches with balanced cluster distribution.
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)

    batch_size, num_targets = xt.shape[:2]
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)

    # Parallel K-means clustering for all batches
    flat_xt = xt.reshape(batch_size * num_targets, -1)
    batch_indices = torch.repeat_interleave(torch.arange(batch_size, device=xt.device), num_targets)
    
    # Initialize centroids randomly for each batch in the dataset
    all_centroids = []
    for b in range(batch_size):
        batch_mask = (batch_indices == b)
        batch_xt_flat = flat_xt[batch_mask]
        
        # Select random initial centroids for this batch
        torch.manual_seed(seed if seed is not None else 42 + b)
        indices = torch.randperm(num_targets)[:num_clusters]
        centroids = batch_xt_flat[indices]
        all_centroids.append(centroids)
    
    # Create a tensor to store all cluster assignments
    all_cluster_assignments = torch.zeros(batch_size * num_targets, dtype=torch.long, device=xt.device)
    
    # Perform K-means clustering for all batches with dissimilarity maximisation
    for _ in range(max_kmeans_iter):
        new_assignments = torch.zeros_like(all_cluster_assignments)
        
        # Process each batch separately
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            batch_xt_flat = flat_xt[batch_mask]
            
            # Compute distances to centroids for this batch
            distances = torch.cdist(batch_xt_flat, all_centroids[b])
            
            # MODIFIED: Assign points to FARTHEST centroid to maximise intra-cluster dissimilarity
            batch_assignments = torch.argmax(distances, dim=1)
            
            # Store assignments for this batch
            new_assignments[batch_mask] = batch_assignments
            
        # Check for convergence across all batches
        if torch.all(all_cluster_assignments == new_assignments):
            break
            
        all_cluster_assignments = new_assignments
        
        # Update centroids to maintain maximum spread
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            batch_xt_flat = flat_xt[batch_mask]
            batch_assignments = all_cluster_assignments[batch_mask]
            
            for c in range(num_clusters):
                cluster_mask = (batch_assignments == c)
                if cluster_mask.any():
                    cluster_points = batch_xt_flat[cluster_mask]
                    # MODIFIED: Use farthest point from cluster mean as new centroid
                    cluster_mean = cluster_points.mean(dim=0)
                    dists_to_mean = torch.norm(cluster_points - cluster_mean, dim=1)
                    farthest_idx = torch.argmax(dists_to_mean)
                    all_centroids[b][c] = cluster_points[farthest_idx]
    
    # Reshape cluster assignments back to [batch_size, num_targets]
    cluster_assignments = all_cluster_assignments.reshape(batch_size, num_targets)
    
    all_ar_batch_indices = []
    # Create AR batches with maximum inter-cluster diversity
    for b in range(batch_size):
        batch_clusters = cluster_assignments[b].tolist()
        # Get cluster indices sorted by variance (descending)
        cluster_variances = []
        for c in range(num_clusters):
            cluster_points = xt[b, cluster_assignments[b] == c]
            if len(cluster_points) > 0:
                cluster_variances.append(torch.var(cluster_points).item())
            else:
                cluster_variances.append(0)
        cluster_order = torch.argsort(torch.tensor(cluster_variances), descending=True).tolist()

        # Distribute highest variance clusters first across AR batches
        ar_batch_indices = [[] for _ in range(ar_batches)]
        for c in cluster_order:
            c_indices = torch.where(cluster_assignments[b] == c)[0].tolist()
            # Add one point to each batch in round-robin fashion
            for i, idx in enumerate(c_indices):
                ar_batch_indices[i % ar_batches].append(idx)
        
        all_ar_batch_indices.append(ar_batch_indices)

    # Process batches in variance order
    for b in range(batch_size):
        batch_xc = xc[b:b+1].clone()
        batch_yc = yc[b:b+1].clone()
        remaining_batches = list(range(ar_batches))

        while remaining_batches:
            # Calculate variances for remaining batches
            variances = []
            for batch_idx in remaining_batches:
                batch_points = all_ar_batch_indices[b][batch_idx]
                if not batch_points:
                    variances.append(float('inf'))
                    continue
                
                batch_xt = xt[b:b+1, batch_points]
                pred_dist = model.predict(batch_xc, batch_yc, batch_xt)
                variances.append(pred_dist.variance.mean().item())

            # Select batch with lowest variance
            min_var_idx = torch.argmin(torch.tensor(variances)).item()
            selected_batch = remaining_batches.pop(min_var_idx)
            batch_points = all_ar_batch_indices[b][selected_batch]

            if batch_points:
                # Process batch and update context
                batch_xt = xt[b:b+1, batch_points]
                batch_yt = yt[b:b+1, batch_points]
                pred_dist = model.predict(batch_xc, batch_yc, batch_xt)
                log_probs = pred_dist.log_prob(batch_yt).sum(-1)
                
                for i, idx in enumerate(batch_points):
                    log_likelihoods[b, idx] = log_probs[0, i]
                
                batch_xc = torch.cat([batch_xc, batch_xt], dim=1)
                batch_yc = torch.cat([batch_yc, batch_yt], dim=1)

    return torch.mean(log_likelihoods, dim=-1)

def ar_log_likelihood_smart(model, xc, yc, xt, yt, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        seed: Random seed for reproducibility
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_y = yc.shape[-1]
    
    # Single target case - revert to simpler method
    if num_targets == 1:
        return ar_log_likelihood(model, xc, yc, xt, yt, seed, get_normal)
    
    # Initialize containers for results
    use_x = xt.clone()
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
    # Each batch will have its own set of non-used indices
    batch_non_used_indices = [list(range(num_targets)) for _ in range(batch_size)]
    
    # Separate context and target for each batch
    batch_current_xc = [xc[b:b+1].clone() for b in range(batch_size)]
    batch_current_yc = [yc[b:b+1].clone() for b in range(batch_size)]

    if get_normal:
        # get the log_likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    # Process one target point at a time for each batch independently
    for i in range(num_targets):
        for b in range(batch_size):
            # Skip if this batch has no more target points to process
            if not batch_non_used_indices[b]:
                continue
                
            # Extract current target points for this batch
            indices = batch_non_used_indices[b]
            current_xt = use_x[b:b+1, indices, :]
            current_yt = yt[b:b+1, indices, :]
            
            # Get prediction distribution for the current target
            pred_dist = model.predict(batch_current_xc[b], batch_current_yc[b], current_xt)
            
            # Find the index of the target point with the lowest variance
            variances = pred_dist.variance.sum(dim=-1).squeeze(-1)  # Sum over output dimensions
            min_var_idx = torch.argmin(variances, dim=-1)  # Get index of minimum variance
            
            if min_var_idx.dim() == 0:
                selected_idx = min_var_idx.item()
            else:
                selected_idx = min_var_idx[0].item()
            
            # Get the log probability for the target point of the one with lowest variance
            current_yt = current_yt[:, selected_idx:selected_idx+1, :]
            relevant_pred_dist = torch.distributions.Normal(pred_dist.mean[:, selected_idx:selected_idx+1, :], pred_dist.scale[:, selected_idx:selected_idx+1, :])
            log_probs = relevant_pred_dist.log_prob(current_yt).sum(-1)  # Sum over output dimensions
            
            # Store the log likelihood in the correct position
            actual_idx = indices[selected_idx]
            log_likelihoods[b, actual_idx] = log_probs[0, 0]
            
            # Add the new chosen point to the context for this batch
            batch_current_xc[b] = torch.cat([batch_current_xc[b], current_xt[:, selected_idx:selected_idx+1, :]], dim=1)
            batch_current_yc[b] = torch.cat([batch_current_yc[b], current_yt], dim=1)

            # Remove the selected index from non_used_indices for this batch
            batch_non_used_indices[b].pop(selected_idx)
    
    log_likelihood = torch.mean(log_likelihoods, dim=-1)
    if get_normal:
        print("context size: ", xc.shape[1])
        print("target size: ", xt.shape[1])
        diff = log_likelihoods - normal_log_likelihoods
        print("diff: ", diff[0])
    
    return log_likelihood

def ar_log_likelihood_smart_batch_clustered(model, xc, yc, xt, yt, ar_batches=2, num_clusters=5, seed=None, get_normal=False, max_kmeans_iter=10):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling
    with target points clustered into K-means clusters, and distributed across autoregressive batches
    to ensure each batch contains a balanced mix of points from different clusters.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        ar_batches: Number of autoregressive batches to process
        num_clusters: Number of K-means clusters to create (can be different from ar_batches)
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood for comparison
        max_kmeans_iter: Maximum number of K-means iterations
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_x = xt.shape[-1]
    dim_y = yc.shape[-1]
    
    # Ensure ar_batches and num_clusters are valid
    ar_batches = min(ar_batches, num_targets)
    num_clusters = min(num_clusters, num_targets)
    
    # Initialize containers for results
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
    if get_normal:
        # Get the log-likelihoods for all the targets at once for comparison
        normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
    
    # Perform K-means clustering for all batches in parallel
    # Reshape xt for clustering: [batch_size, num_targets, dim_x] -> [batch_size * num_targets, dim_x]
    flat_xt = xt.reshape(batch_size * num_targets, -1)
    
    # Create batch indices to keep track of which batch each point belongs to
    batch_indices = torch.repeat_interleave(torch.arange(batch_size, device=xt.device), num_targets)
    
    # Initialize centroids randomly for each batch in the dataset
    all_centroids = []
    for b in range(batch_size):
        batch_mask = (batch_indices == b)
        batch_xt_flat = flat_xt[batch_mask]
        
        # Select random initial centroids for this batch
        torch.manual_seed(seed if seed is not None else 42 + b)
        indices = torch.randperm(num_targets)[:num_clusters]
        centroids = batch_xt_flat[indices]
        all_centroids.append(centroids)
    
    # Create a tensor to store all cluster assignments
    all_cluster_assignments = torch.zeros(batch_size * num_targets, dtype=torch.long, device=xt.device)
    
    # Perform K-means clustering for all batches with dissimilarity maximisation
    for _ in range(max_kmeans_iter):
        new_assignments = torch.zeros_like(all_cluster_assignments)
        
        # Process each batch separately
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            batch_xt_flat = flat_xt[batch_mask]
            
            # Compute distances to centroids for this batch
            distances = torch.cdist(batch_xt_flat, all_centroids[b])
            
            # MODIFIED: Assign points to FARTHEST centroid to maximise intra-cluster dissimilarity
            batch_assignments = torch.argmax(distances, dim=1)
            
            # Store assignments for this batch
            new_assignments[batch_mask] = batch_assignments
            
        # Check for convergence across all batches
        if torch.all(all_cluster_assignments == new_assignments):
            break
            
        all_cluster_assignments = new_assignments
        
        # Update centroids to maintain maximum spread
        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            batch_xt_flat = flat_xt[batch_mask]
            batch_assignments = all_cluster_assignments[batch_mask]
            
            for c in range(num_clusters):
                cluster_mask = (batch_assignments == c)
                if cluster_mask.any():
                    cluster_points = batch_xt_flat[cluster_mask]
                    # MODIFIED: Use farthest point from cluster mean as new centroid
                    cluster_mean = cluster_points.mean(dim=0)
                    dists_to_mean = torch.norm(cluster_points - cluster_mean, dim=1)
                    farthest_idx = torch.argmax(dists_to_mean)
                    all_centroids[b][c] = cluster_points[farthest_idx]
    
    # Reshape cluster assignments back to [batch_size, num_targets]
    cluster_assignments = all_cluster_assignments.reshape(batch_size, num_targets)
    
    # Create autoregressive batches with balanced cluster representation for each batch
    all_ar_batch_indices = []
    
    for b in range(batch_size):
        # Create lists of indices for each cluster in this batch
        batch_cluster_indices = [torch.where(cluster_assignments[b] == c)[0].tolist() for c in range(num_clusters)]
        
        # Shuffle each cluster's indices to avoid spatial patterns
        for c in range(num_clusters):
            if batch_cluster_indices[c]:
                torch.manual_seed(seed if seed is not None else 42 + b + c)
                batch_cluster_indices[c] = [batch_cluster_indices[c][i] for i in torch.randperm(len(batch_cluster_indices[c])).tolist()]
        
        # Create ar_batches with balanced cluster representation
        batch_ar_indices = [[] for _ in range(ar_batches)]
        
        # Distribute points from each cluster to each AR batch as evenly as possible
        for c in range(num_clusters):
            if not batch_cluster_indices[c]:
                continue
                
            # Calculate target number of points from this cluster for each AR batch
            points_per_batch = [int(len(batch_cluster_indices[c]) / ar_batches) for _ in range(ar_batches)]
            
            # Distribute remainder
            remainder = len(batch_cluster_indices[c]) - sum(points_per_batch)
            for i in range(remainder):
                points_per_batch[i % ar_batches] += 1
            
            # Assign points to batches
            start_idx = 0
            for batch_idx, num_points in enumerate(points_per_batch):
                end_idx = start_idx + num_points
                batch_ar_indices[batch_idx].extend(batch_cluster_indices[c][start_idx:end_idx])
                start_idx = end_idx
        
        all_ar_batch_indices.append(batch_ar_indices)
    
    # Now process each batch in the dataset
    for b in range(batch_size):
        # Start with original context for this batch
        batch_xc = xc[b:b+1].clone()
        batch_yc = yc[b:b+1].clone()
        
        # Get the AR batches for this batch
        ar_batch_indices = all_ar_batch_indices[b]
        
        # Process AR batches in order of lowest average variance
        remaining_batches = list(range(ar_batches))
        
        while remaining_batches:
            batch_variances = []
            
            # Compute variances for all remaining batches
            for batch_idx in remaining_batches:
                if not ar_batch_indices[batch_idx]:
                    batch_variances.append(float('inf'))
                    continue
                
                # Extract target points for this AR batch
                batch_point_indices = ar_batch_indices[batch_idx]
                ar_batch_xt = xt[b:b+1, batch_point_indices]
                
                # Get prediction distribution for this batch
                pred_dist = model.predict(batch_xc, batch_yc, ar_batch_xt)
                
                # Compute average variance for this batch
                avg_variance = pred_dist.variance.mean()
                batch_variances.append(avg_variance.item())
            
            # Find batch with minimum average variance
            max_var_idx = torch.argmax(torch.tensor(batch_variances)).item()
            selected_batch_idx = remaining_batches[max_var_idx]
            
            # Remove this batch from remaining batches
            remaining_batches.remove(selected_batch_idx)
            
            # Get indices of points in this batch
            batch_point_indices = ar_batch_indices[selected_batch_idx]
            
            if not batch_point_indices:
                continue
                
            # Extract target points for this batch
            ar_batch_xt = xt[b:b+1, batch_point_indices]
            ar_batch_yt = yt[b:b+1, batch_point_indices]
            
            # Get prediction distribution for this batch
            pred_dist = model.predict(batch_xc, batch_yc, ar_batch_xt)
            
            # Get log likelihood of each target point in the batch
            log_probs = pred_dist.log_prob(ar_batch_yt).sum(-1)  # [1, batch_size]
            
            # Store the log likelihoods
            for i, idx in enumerate(batch_point_indices):
                log_likelihoods[b, idx] = log_probs[0, i]
            
            # Add all points from this batch to the context before processing the next batch
            batch_xc = torch.cat([batch_xc, ar_batch_xt], dim=1)
            batch_yc = torch.cat([batch_yc, ar_batch_yt], dim=1)
    
    # Average log-likelihoods across target points
    log_likelihood = torch.mean(log_likelihoods, dim=-1)
    
    if get_normal:
        print("Context size: ", xc.shape[1])
        print("Target size: ", xt.shape[1])
        print("Number of K-means clusters: ", num_clusters)
        print("Number of AR batches: ", ar_batches)
        diff = log_likelihoods - normal_log_likelihoods
        print("Mean log-likelihood difference: ", torch.mean(diff, dim=-1))
    
    return log_likelihood

def ar_log_likelihood_fast_smart(model, xc, yc, xt, yt, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points using autoregressive sampling.
    The order of target point processing is determined once at the beginning:
    - If the context set `xc` is empty, a random order is used.
    - Otherwise, points are sorted based on their maximum Euclidean distance
      to any point in the *original* context `xc` (descending order, furthest first).
    The context used for prediction is dynamically updated at each step.
    **This version parallelises computation across the batch dimension in the main loop.**

    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood for comparison

    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)

    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_x = xt.shape[-1] # Get dim_x
    dim_y = yc.shape[-1]

    # Single target case - use simpler method (no ordering needed)
    if num_targets <= 1:
        # Note: Using ar_log_likelihood which processes one by one anyway.
        # If num_targets is 0, it should handle gracefully.
        # If num_targets is 1, the order doesn't matter.
        return ar_log_likelihood(model, xc, yc, xt, yt, seed, get_normal)

    # Initialize containers
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)

    # Determine processing order for each batch item
    target_order = torch.zeros(batch_size, num_targets, dtype=torch.long, device=xt.device)

    for b in range(batch_size):
        if xc.shape[1] == 0:
            # Empty context: use random order
            target_order[b] = torch.randperm(num_targets, device=xt.device)
        else:
            # Context exists: order based on sum of square rooted distances to adjacent context points
            context_x_b = xc[b]  # [num_context, dim_x]
            target_x_b = xt[b]   # [num_targets, dim_x]

            # Compute pairwise Euclidean distances square rooted: [num_targets, num_context]
            distances_square_rooted_b = torch.cdist(target_x_b, context_x_b, p=2) ** 0.1

            # For each target point, compute sum of squared distances to the two nearest context points
            # First sort distances for each target point
            sorted_distances_b, _ = torch.sort(distances_square_rooted_b, dim=1)
            
            # Take sum of two smallest distances (or just one if there's only one context point)
            if context_x_b.shape[0] >= 2:
                adjacent_sum_b = sorted_distances_b[:, 0] + sorted_distances_b[:, 1]
            else:
                adjacent_sum_b = sorted_distances_b[:, 0]

            # Get indices sorted by sum of square rooted distances to adjacent points (descending)
            target_order[b] = torch.argsort(adjacent_sum_b, descending=True)

    # Initial context (batched tensor)
    current_xc = xc.clone() # Shape [batch_size, num_context, dim_x]
    current_yc = yc.clone() # Shape [batch_size, num_context, dim_y]


    if get_normal:
        # Calculate normal log likelihood for comparison if requested
        with torch.no_grad(): # Ensure no gradients computed here
             normal_pred_dist = model.predict(xc, yc, xt)
             normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)

    # Process targets sequentially according to the pre-determined order (parallelised over batch)
    for i in range(num_targets):
        # Get the actual indices for ALL batches for the i-th step
        actual_indices = target_order[:, i] # Shape [batch_size]

        # Gather the corresponding xt and yt values for all batches
        # Indices need shape [batch_size, 1, 1] expanded for gather
        indices_for_gather_x = actual_indices.view(batch_size, 1, 1).expand(-1, -1, dim_x)
        current_xt = torch.gather(xt, 1, indices_for_gather_x) # Shape [batch_size, 1, dim_x]

        indices_for_gather_y = actual_indices.view(batch_size, 1, 1).expand(-1, -1, dim_y)
        current_yt = torch.gather(yt, 1, indices_for_gather_y) # Shape [batch_size, 1, dim_y]

        # Predict using the *current* parallel context (for the entire batch)
        with torch.no_grad():
            pred_dist = model.predict(current_xc, current_yc, current_xt) # Operates on [batch_size, ...] tensors

        # Calculate log probability for these specific target points
        # Shape [batch_size, 1], sum over dim_y
        log_probs = pred_dist.log_prob(current_yt).sum(dim=-1)

        # Store log likelihoods at the correct 'actual_idx' for each batch.
        # Use scatter_ for in-place update.
        # unsqueeze actual_indices to match log_probs shape for scatter
        log_likelihoods.scatter_(1, actual_indices.unsqueeze(1), log_probs)

        # Update the parallel context
        current_xc = torch.cat([current_xc, current_xt], dim=1)
        current_yc = torch.cat([current_yc, current_yt], dim=1)


    # Average log-likelihoods across all target points for each batch item
    log_likelihood = torch.mean(log_likelihoods, dim=-1)

    if get_normal:
        # do comparison with normal ar method where we keep original order
        # Initialize containers
        original_order_log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
        # Generate multiple independent autoregressive samples
        # Start with the original context for each sample
        original_order_xc = xc.clone()
        original_order_yc = yc.clone()
    
        # Process one target point at a time
        for i in range(num_targets):
            # Extract the current target point
            current_xt = xt[:, i:i+1, :]
            current_yt = yt[:, i:i+1, :]
            
            # Get prediction distribution for the current target
            pred_dist = model.predict(original_order_xc, original_order_yc, current_xt)
            
            # Get log likelihood of the target point
            log_likelihood = pred_dist.log_prob(current_yt)
            
            # Store the log likelihood
            original_order_log_likelihoods[:, i] = log_likelihood[:,0,:].sum(-1)
            
            # Add the new point to the context
            original_order_xc = torch.cat([original_order_xc, current_xt], dim=1)
            original_order_yc = torch.cat([original_order_yc, current_yt], dim=1)

        # Print comparison information if requested
        print("Context size:", xc.shape[1])
        print("Target size:", xt.shape[1])
        # Ensure normal_log_likelihoods is defined before calculating diff
        if 'normal_log_likelihoods' in locals():
             diff = log_likelihoods - normal_log_likelihoods
             # Avoid printing potentially large tensors, print summary stats instead
             print(f"Mean log-likelihood difference (across batch): {torch.mean(diff).item():.4f}")
             print(f"Mean Absolute Diff per point (Batch 0): {torch.mean(torch.abs(diff[0])).item():.4f}")
             # do comparison with original order
             ar_diff = log_likelihoods - original_order_log_likelihoods
             print(f"Mean log-likelihood difference (AR original order): {torch.mean(ar_diff).item():.4f}")
             print(f"Mean Absolute Diff per point (AR original order): {torch.mean(torch.abs(ar_diff[0])).item():.4f}")
             print(f"Number of original context points: {xc.shape[1]}")
             print(f"Number of target points: {xt.shape[1]}")

             # Find cases where smart AR does worse than original AR by more than 0.3
             # Check difference per batch element, not averaged
             ll_smart = torch.mean(log_likelihoods, dim=1)  # Average over target points for each batch
             ll_original = torch.mean(original_order_log_likelihoods, dim=1)  # Same for original
             
             diff_per_batch = ll_smart - ll_original
             worse_indices = torch.where(diff_per_batch < -0.3)[0]
             
             if len(worse_indices) > 0:
                print(f"Found {len(worse_indices)} cases where smart AR performs worse by >0.3 log likelihood")
                 
                # Import visualization libraries if needed
                try:
                     import matplotlib.pyplot as plt
                     import numpy as np
                     import os
                     
                     # Create save directory
                     save_dir = os.path.join(os.getcwd(), "ar_plots")
                     os.makedirs(save_dir, exist_ok=True)
                except ImportError:
                     print("Matplotlib not available for visualization")
                     has_matplotlib = False
                 
                # Only plot for 1D input/output for simplicity
                if xt.shape[-1] == 1 and yt.shape[-1] == 1 and num_targets < 6:
                    # Plot at most 3 bad cases to avoid too many plots
                    for i, bad_idx in enumerate(worse_indices[:min(3, len(worse_indices))]):
                        # Get data for this batch item
                        this_xc = xc[bad_idx].cpu().numpy()
                        this_yc = yc[bad_idx].cpu().numpy()
                        this_xt = xt[bad_idx].cpu().numpy()
                        this_yt = yt[bad_idx].cpu().numpy()
                        
                        # Get the ordering used by each method
                        smart_order = target_order[bad_idx].cpu().numpy()
                        
                        # Extract individual point likelihoods for this batch item
                        smart_ll_points = log_likelihoods[bad_idx].cpu().numpy()
                        orig_ll_points = original_order_log_likelihoods[bad_idx].cpu().numpy()
                        point_ll_diff = smart_ll_points - orig_ll_points
                        
                        # Create figure
                        plt.figure(figsize=(12, 8))
                        
                        # Plot context points in blue
                        plt.scatter(this_xc, this_yc, color='blue', s=100, label='Context', zorder=10)
                        
                        # Plot target points with color based on likelihood difference
                        # Create a colormap: red=worse (negative diff), white=neutral, green=better (positive diff)
                        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
                        
                        # Normalize the colormap - set limits to make the colormap more visible
                        vmax = max(0.5, np.abs(point_ll_diff).max())
                        vmin = -vmax
                        norm = plt.Normalize(vmin, vmax)
                        
                        # Plot target points with colors from the colormap
                        scatter = plt.scatter(this_xt, this_yt, c=point_ll_diff, cmap=cmap, norm=norm, 
                                            s=80, label='Target', zorder=5, edgecolor='black')
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Smart LL - Original LL (per point)')
                        
                        # Label target points with their original and smart order
                        for t_idx in range(len(this_xt)):
                            # Find position in smart order
                            smart_pos = np.where(smart_order == t_idx)[0][0]
                            
                            # Include the point's log likelihood difference in the label
                            point_diff = point_ll_diff[t_idx]
                            
                            # Place text labels with ordering information and likelihood diff
                            plt.annotate(f"O:{t_idx}\nS:{smart_pos}\nΔ:{point_diff:.2f}",
                                    xy=(this_xt[t_idx], this_yt[t_idx]),
                                    xytext=(10, 0), 
                                    textcoords='offset points',
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))
                        
                        # Optional: Add arrows showing smart ordering path
                        for i in range(len(smart_order) - 1):
                            from_idx = smart_order[i]
                            to_idx = smart_order[i+1]
                            plt.annotate("",
                                    xy=(this_xt[to_idx], this_yt[to_idx]),
                                    xytext=(this_xt[from_idx], this_yt[from_idx]),
                                    arrowprops=dict(arrowstyle="->", lw=1.5, color='green', alpha=0.6))
                        
                        # Title with performance info
                        plt.title(f"Case where Smart AR performs worse (batch {bad_idx})\n" +
                                f"Smart LL: {ll_smart[bad_idx].item():.4f}, " +
                                f"Original LL: {ll_original[bad_idx].item():.4f}, " +
                                f"Diff: {diff_per_batch[bad_idx].item():.4f}")
                        
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Save plot
                        try:
                            plt.savefig(os.path.join(save_dir, f"ar_comparison_batch{bad_idx}.png"))
                            print(f"Plot saved to {os.path.join(save_dir, f'ar_comparison_batch{bad_idx}.png')}")
                        except Exception as e:
                            print(f"Could not save plot: {e}")
                            
                        plt.close()

        else:
            print("Normal log likelihoods not calculated (perhaps num_targets <= 1).")

        # Also find cases where smart AR does BETTER than original AR by more than 0.3
        better_indices = torch.where(diff_per_batch > 0.3)[0]
        
        if len(better_indices) > 0:
            print(f"Found {len(better_indices)} cases where smart AR performs better by >0.3 log likelihood")
            
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            # Only plot for 1D input/output for simplicity
            if xt.shape[-1] == 1 and yt.shape[-1] == 1 and num_targets < 6:
                # Plot at most 3 good cases to avoid too many plots
                for i, good_idx in enumerate(better_indices[:min(3, len(better_indices))]):
                    # Get data for this batch item
                    this_xc = xc[good_idx].cpu().numpy()
                    this_yc = yc[good_idx].cpu().numpy()
                    this_xt = xt[good_idx].cpu().numpy()
                    this_yt = yt[good_idx].cpu().numpy()
                    
                    # Get the ordering used by each method
                    smart_order = target_order[good_idx].cpu().numpy()
                    
                    # Extract individual point likelihoods for this batch item
                    smart_ll_points = log_likelihoods[good_idx].cpu().numpy()
                    orig_ll_points = original_order_log_likelihoods[good_idx].cpu().numpy()
                    point_ll_diff = smart_ll_points - orig_ll_points
                    
                    # Create figure
                    plt.figure(figsize=(12, 8))
                    
                    # Plot context points in blue
                    plt.scatter(this_xc, this_yc, color='blue', s=100, label='Context', zorder=10)
                    
                    # Plot target points with color based on likelihood difference
                    # Create a colormap: red=worse (negative diff), white=neutral, green=better (positive diff)
                    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
                    
                    # Normalize the colormap - set limits to make the colormap more visible
                    vmax = max(0.5, np.abs(point_ll_diff).max())
                    vmin = -vmax
                    norm = plt.Normalize(vmin, vmax)
                    
                    # Plot target points with colors from the colormap
                    scatter = plt.scatter(this_xt, this_yt, c=point_ll_diff, cmap=cmap, norm=norm, 
                                        s=80, label='Target', zorder=5, edgecolor='black')
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('Smart LL - Original LL (per point)')
                    
                    # Label target points with their original and smart order
                    for t_idx in range(len(this_xt)):
                        # Find position in smart order
                        smart_pos = np.where(smart_order == t_idx)[0][0]
                        
                        # Include the point's log likelihood difference in the label
                        point_diff = point_ll_diff[t_idx]
                        
                        # Place text labels with ordering information and likelihood diff
                        plt.annotate(f"O:{t_idx}\nS:{smart_pos}\nΔ:{point_diff:.2f}",
                                    xy=(this_xt[t_idx], this_yt[t_idx]),
                                    xytext=(10, 0), 
                                    textcoords='offset points',
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))
                    
                    # Optional: Add arrows showing smart ordering path
                    for i in range(len(smart_order) - 1):
                        from_idx = smart_order[i]
                        to_idx = smart_order[i+1]
                        plt.annotate("",
                                    xy=(this_xt[to_idx], this_yt[to_idx]),
                                    xytext=(this_xt[from_idx], this_yt[from_idx]),
                                    arrowprops=dict(arrowstyle="->", lw=1.5, color='green', alpha=0.6))
                    
                    # Title with performance info
                    plt.title(f"Case where Smart AR performs better (batch {good_idx})\n" +
                            f"Smart LL: {ll_smart[good_idx].item():.4f}, " +
                            f"Original LL: {ll_original[good_idx].item():.4f}, " +
                            f"Diff: {diff_per_batch[good_idx].item():.4f}")
                    
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save plot
                    try:
                        plt.savefig(os.path.join(save_dir, f"ar_better_batch{good_idx}.png"))
                        print(f"Plot saved to {os.path.join(save_dir, f'ar_better_batch{good_idx}.png')}")
                    except Exception as e:
                        print(f"Could not save plot: {e}")
                        
                    plt.close()
            else:
                print("Visualization only available for 1D input/output data")

    return log_likelihood

def ar_log_likelihood_smart_simple(model, xc, yc, xt, yt, seed=None, get_normal=False):
    """
    Computes the log likelihood of the target points given the context using autoregressive sampling.
    Unlike ar_log_likelihood_smart, this function determines the order of target points only once
    at the beginning (selecting points with smallest variance first).
    This implementation processes all batches in parallel to speed up computation.
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood for comparison
        
    Returns:
        log_likelihood: Log likelihood of the target points given the context
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_x = xt.shape[-1]
    dim_y = yc.shape[-1]
    
    # Single target case - revert to simpler method
    if num_targets == 1:
        return ar_log_likelihood(model, xc, yc, xt, yt, seed, get_normal)
    
    # Initialize containers for results
    log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
    # Determine ordering at the beginning for all batches
    target_order = torch.zeros(batch_size, num_targets, dtype=torch.long, device=xt.device)
    
    # Get initial prediction distributions for all target points
    with torch.no_grad():
        pred_dist = model.predict(xc, yc, xt)
    
    # Calculate variance for each target point and determine ordering (smallest variance last)
    variances = pred_dist.variance.sum(dim=-1)  # Sum over output dimensions, shape [batch_size, num_targets]
    
    # Sort variances to get indices in ascending order (smallest variance last) for all batches
    for b in range(batch_size):
        target_order[b] = torch.argsort(variances[b], descending=True)
    
    # Create parallel context - start with original context for all batches
    current_xc = xc.clone()  # Shape [batch_size, num_context, dim_x]
    current_yc = yc.clone()  # Shape [batch_size, num_context, dim_y]
    
    # Process one target point at a time for all batches in parallel
    for i in range(num_targets):
        # Get current target indices for this step across all batches
        current_indices = target_order[:, i]  # Shape [batch_size]
        
        # Gather the current target x and y values for all batches
        # Need to reshape indices for gather operation
        gather_indices_x = current_indices.view(batch_size, 1, 1).expand(-1, -1, dim_x)
        current_xt = torch.gather(xt, 1, gather_indices_x)  # Shape [batch_size, 1, dim_x]
        
        gather_indices_y = current_indices.view(batch_size, 1, 1).expand(-1, -1, dim_y)
        current_yt = torch.gather(yt, 1, gather_indices_y)  # Shape [batch_size, 1, dim_y]
        
        # Get prediction distribution for all batches at once
        with torch.no_grad():
            pred_dist = model.predict(current_xc, current_yc, current_xt)
        
        # Calculate log probabilities
        log_probs = pred_dist.log_prob(current_yt).sum(dim=-1)  # Shape [batch_size, 1]
        
        # Store log likelihoods at the correct positions
        for b in range(batch_size):
            log_likelihoods[b, current_indices[b]] = log_probs[b, 0]
        
        # Add current targets to context for all batches
        current_xc = torch.cat([current_xc, current_xt], dim=1)
        current_yc = torch.cat([current_yc, current_yt], dim=1)
    
    # Average log-likelihoods across target points
    log_likelihood = torch.mean(log_likelihoods, dim=-1)
    
    if get_normal:
        # Calculate normal log likelihood for comparison if requested
        with torch.no_grad():
            normal_pred_dist = model.predict(xc, yc, xt)
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
        
        print("Context size:", xc.shape[1])
        print("Target size:", xt.shape[1])
        diff = log_likelihoods - normal_log_likelihoods
        print("Difference:", diff[0])
    
    return log_likelihood

def ar_log_likelihood_mc_smart(model, xc, yc, xt, yt, num_samples_factor=1/5, l=10.0, seed=None, get_normal=False):
    """
    Highly parallelised version of ar_log_likelihood_mc that computes permutations based on variance.
    
    This implementation:
    1. Identifies high variance points that exceed l standard deviations from the average
    2. Only permutes high variance points amongst themselves
    3. Slots high variance points into random positions
    4. Randomly permutes low variance points in the remaining positions
    
    Args:
        model: An instance of the LBANP model
        xc: Context x values [batch_size, num_context_points, dim_x]
        yc: Context y values [batch_size, num_context_points, dim_y]
        xt: Target x values [batch_size, num_target_points, dim_x]
        yt: Target y values [batch_size, num_target_points, dim_y]
        num_samples_factor: Number of permutations to sample
        l: Number of standard deviations for high variance identification
        seed: Random seed for reproducibility
        get_normal: Whether to compute and return the normal (non-autoregressive) log-likelihood
        
    Returns:
        log_likelihood: Average log likelihood across different permutations
    """
    model.eval()
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42)
    
    batch_size = xc.shape[0]
    num_targets = xt.shape[1]
    dim_x = xt.shape[-1]
    dim_y = yc.shape[-1]
    
    # Get original prediction variances for all target points
    with torch.no_grad():
        normal_pred_dist = model.predict(xc, yc, xt)
        point_variances = normal_pred_dist.variance.sum(dim=-1)  # [batch_size, num_targets]

    # Find high variance points for each batch
    high_var_masks = []
    num_high_var_points = []
    
    for b in range(batch_size):
        # Calculate threshold for high variance
        mean_var = point_variances[b].mean()
        std_var = point_variances[b].std()
        threshold = mean_var + l * std_var
        
        # Identify high variance points
        high_var_mask = point_variances[b] > threshold
        
        # Cap at maximum 5 high variance points
        if high_var_mask.sum() > 5:
            # Get indices of top 5 highest variance points
            _, top_indices = torch.topk(point_variances[b], 5)
            new_mask = torch.zeros_like(high_var_mask)
            new_mask[top_indices] = True
            high_var_mask = new_mask
        
        high_var_masks.append(high_var_mask)
        num_high_var_points.append(high_var_mask.sum().item())
    
    # Determine number of samples for each batch
    num_samples_per_batch = []
    
    for b in range(batch_size):
        n_high = num_high_var_points[b]
        
        if n_high <= 1:
            # If 0 or 1 high variance points, take only one permutation
            num_samples = 1
        else:
            # Use the num_samples_factor directly
            num_samples = math.factorial(n_high) #max(int(num_samples_factor * math.factorial(n_high-1)), 1)
        
        num_samples_per_batch.append(num_samples)
    
    # Find max number of samples across all batches for tensor allocation
    max_num_samples = max(num_samples_per_batch)
    
    # Generate permutations for each batch separately
    all_permutations = torch.zeros(batch_size, max_num_samples, num_targets, dtype=torch.long, device=xt.device)
    
    for b in range(batch_size):
        high_var_mask = high_var_masks[b]
        num_samples = num_samples_per_batch[b]
        
        # Split indices into high and low variance
        high_var_indices = torch.where(high_var_mask)[0]
        low_var_indices = torch.where(~high_var_mask)[0]
        
        n_high = len(high_var_indices)
        n_low = len(low_var_indices)
        # num_samples = math.factorial(n_high)
        all_perms = list(itertools.permutations(list(range(n_high))))
        
        for s in range(num_samples):
            # Use unique but deterministic seed for this batch and sample
            sample_seed = seed + b * 1000 + s if seed is not None else 42 + b * 1000 + s
            torch.manual_seed(sample_seed)
            
            # Create an array to hold the final permutation
            permutation = torch.zeros(num_targets, dtype=torch.long, device=xt.device)
            if n_high > 0:
                # get all permutations high variance indices
                high_var_perm = torch.tensor(all_perms[s], dtype=torch.long, device=xt.device)
                
                # Choose random slots for high variance points
                slot_indices = torch.randperm(num_targets)[:n_high]
                permutation[slot_indices] = high_var_perm
                
                # Create mask of remaining slots
                remaining_mask = torch.ones(num_targets, dtype=torch.bool, device=xt.device)
                remaining_mask[slot_indices] = False
                
                # Randomly permute low variance indices
                low_var_perm = low_var_indices[torch.randperm(n_low)]
                
                # Fill remaining slots with low variance points
                permutation[remaining_mask] = low_var_perm
            else:
                # If no high variance points, just do a random permutation
                permutation = torch.randperm(num_targets, device=xt.device)
            
            all_permutations[b, s] = permutation
    
    # Initialize storage for log likelihoods
    log_likelihoods = torch.zeros(batch_size, max_num_samples, num_targets, device=xt.device)
    
    # Process each batch separately due to different number of samples
    for b in range(batch_size):
        num_samples = num_samples_per_batch[b]
        
        # Skip if this batch needs only one permutation (optimization)
        if num_samples <= 1:
            # Single permutation case - process directly
            current_xc = xc[b:b+1].clone()
            current_yc = yc[b:b+1].clone()
            permutation = all_permutations[b, 0]
            
            for i in range(num_targets):
                idx = permutation[i].item()
                target_x = xt[b:b+1, idx:idx+1]
                target_y = yt[b:b+1, idx:idx+1]
                
                with torch.no_grad():
                    pred_dist = model.predict(current_xc, current_yc, target_x)
                
                log_prob = pred_dist.log_prob(target_y).sum(-1)
                log_likelihoods[b, 0, idx] = log_prob
                
                # Update context
                current_xc = torch.cat([current_xc, target_x], dim=1)
                current_yc = torch.cat([current_yc, target_y], dim=1)
            
            continue
        
        # Create expanded context for this batch - shape: [num_samples, num_context, dim]
        batch_xc = repeat(xc[b:b+1], 'b n d -> (s b) n d', s=num_samples)
        batch_yc = repeat(yc[b:b+1], 'b n d -> (s b) n d', s=num_samples)
        
        # Process target points one at a time
        for i in range(num_targets):
            # Get indices for current step across all samples
            current_indices = all_permutations[b, :num_samples, i]
            
            # Extract target points for each sample based on their permutation
            current_xt = torch.zeros(num_samples, 1, dim_x, device=xt.device)
            current_yt = torch.zeros(num_samples, 1, dim_y, device=yt.device)
            
            # Extract the correct target point for each sample
            for s in range(num_samples):
                idx = current_indices[s].item()
                current_xt[s, 0] = xt[b, idx]
                current_yt[s, 0] = yt[b, idx]
            
            # Predict
            with torch.no_grad():
                pred_dist = model.predict(batch_xc, batch_yc, current_xt)
            
            # Get log probabilities
            log_probs = pred_dist.log_prob(current_yt).sum(-1)  # [num_samples, 1]
            
            # Store log probabilities in original order
            for s in range(num_samples):
                idx = current_indices[s].item()
                log_likelihoods[b, s, idx] = log_probs[s, 0]
            
            # Update context
            batch_xc = torch.cat([batch_xc, current_xt], dim=1)
            batch_yc = torch.cat([batch_yc, current_yt], dim=1)
    
    # Average log likelihoods across target points for each valid permutation
    # Shape: [batch_size, max_num_samples]
    avg_log_likelihoods = torch.zeros(batch_size, max_num_samples, device=xt.device)
    
    for b in range(batch_size):
        num_samples = num_samples_per_batch[b]
        avg_log_likelihoods[b, :num_samples] = torch.mean(log_likelihoods[b, :num_samples], dim=-1)
        
        # Fill unused samples with large negative value to exclude from logsumexp
        if num_samples < max_num_samples:
            avg_log_likelihoods[b, num_samples:] = -1e20
    
    # Average in likelihood space for each batch using only valid samples
    log_likelihood = torch.zeros(batch_size, device=xt.device)
    
    for b in range(batch_size):
        num_samples = num_samples_per_batch[b]
        log_likelihood[b] = torch.logsumexp(avg_log_likelihoods[b, :num_samples], dim=0) - torch.log(torch.tensor(num_samples, device=xt.device))
    
    if get_normal:
        # Normal log likelihoods already calculated at the beginning
        normal_log_likelihoods = normal_pred_dist.log_prob(yt).sum(-1)
        normal_avg = torch.mean(normal_log_likelihoods, dim=-1)
        
        print(f"Context size: {xc.shape[1]}")
        print(f"Target size: {xt.shape[1]}")
        print(f"Mean log-likelihood difference: {torch.mean(log_likelihood - normal_avg).item():.4f}")
        
        # Print statistics about high variance points
        print(f"Number of high variance points per batch: {num_high_var_points}")
        print(f"Number of samples per batch: {num_samples_per_batch}")
    
    return log_likelihood

if __name__ == "__main__":
    # test ar_log_likelihood_smart
    from models.lbanp import LBANP
    import torch    
    model = LBANP(num_latents=8,
                dim_x= 1,
                dim_y= 1,
                d_model=64,
                emb_depth=4,
                dim_feedforward=128,
                nhead=4,
                dropout=0.0,
                num_layers=6)
    xc = torch.randn(10, 10, 1)
    yc = torch.randn(10, 10, 1)
    xt = torch.randn(10, 2, 1)
    yt = torch.randn(10, 2, 1)
    print(ar_log_likelihood_smart(model, xc, yc, xt, yt))