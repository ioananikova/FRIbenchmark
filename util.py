import torch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, AdditiveKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import matplotlib.pyplot as plt
import numpy as np


def get_and_fit_gp(X, Y, inp_transform, train=True):
    """Simple method for creating a GP with one output dimension.

    X is assumed to be in [0, 1]^d.
    """
    assert Y.ndim == 2 and Y.shape[-1] == 1
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
    covar_module = get_covar_module_with_dim_scaled_prior(ard_num_dims=X.shape[-1], use_rbf_kernel=False)
    gp = SingleTaskGP(X, Y, likelihood=likelihood, covar_module=covar_module, input_transform=inp_transform)
    
    if train:
        mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
        fit_gpytorch_mll(mll)
        
    return gp

def get_kernel(gp_models):
    kernels = []
    scale = 1/len(gp_models)
    for model in gp_models:
        k = ScaleKernel(model.covar_module)
        k.outputscale = scale  # Set weight directly as output scale
        kernels.append(k)
    
    return AdditiveKernel(*kernels)

def identify_samples_which_satisfy_constraints(X, constraints):
    """
    Takes in values (a1, ..., ak, o) and returns (a1, ..., ak, o)
    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        direction, value = constraints[model_index]
        successful[..., model_index] = (
            these_X <= value if direction == "lt" else these_X >= value
        )
    return successful

def calculate_probability_of_feasibility(X, gp_models, constraints):
    """
    Takes in values (a1, ..., ak) and returns (a1, ..., ak)
    probability of feasibility values, where o is the number of outputs.
    """
    gp_models.eval()
    with torch.no_grad():
        posterior = gp_models.posterior(X)
        mus = posterior.mean
        sigma2s = posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        thresholds = torch.tensor([threshold for _, threshold in constraints]).to(X)
        norm_cdf = dist.cdf(thresholds)
    
    prob_feas = torch.ones(X.shape[0], len(constraints)).to(X)
    for i, (direction, _) in enumerate(constraints):
        prob_feas[:, i] = (
            norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
        )
    return prob_feas

def plot_predictions(gp, bounds, problem_name, save_path, constraints=None):
    """
    Create scatter plots for 2D and 3D cases of X, or parallel coordinate plots for higher dimensions,
    with GP mean predictions, uncertainty (standard deviation), and probability of feasibility as color. 
    For multiple outputs, creates side-by-side mean, uncertainty, and PoF plots for each output.
    
    Args:
        gp: trained GP model or list of GP models
        bounds: list of lists - bounds for each dimension [[lower_bounds], [upper_bounds]]
        problem_name: str - name of the problem for plot title
        save_path: str - path where to save the plot
        constraints: list of constraints for PoF calculation (optional)
    """
    n_dims = bounds[0].shape[0]

    if n_dims == 2:
        # Generate a grid of points within the bounds
        n_points_per_dim = 100
        grid_axes = [np.linspace(bounds[0][dim], bounds[1][dim], n_points_per_dim) for dim in range(n_dims)]
        mesh = np.meshgrid(*grid_axes)
        X_np = np.vstack([m.flatten() for m in mesh]).T
        X = torch.tensor(X_np, dtype=torch.float32)
        
        # Get GP predictions
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X)
            mus = posterior.mean
            sigma2s = posterior.variance
            mean = mus.detach().cpu().numpy()
            variance = sigma2s.detach().cpu().numpy()
            std = np.sqrt(variance)
        
        # Handle single or multiple outputs
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)
        
        n_outputs = mean.shape[1]
        
        # Calculate PoF if constraints are provided
        pof = None
        if constraints is not None:
            pof = calculate_probability_of_feasibility(X, gp, constraints).detach().cpu().numpy()
        
        # Create subfigures with 3 columns per output (mean, std, and PoF)
        n_cols = 3 if pof is not None else 2
        fig, axes = plt.subplots(n_outputs, n_cols, figsize=(8 * n_cols, 8 * n_outputs))
        
        # Ensure axes is always 2D
        if n_outputs == 1:
            axes = axes.reshape(1, -1)
        
        for output_idx in range(n_outputs):
            # Mean plot
            ax_mean = axes[output_idx, 0]
            sc_mean = ax_mean.scatter(X_np[:, 0], X_np[:, 1], c=mean[:, output_idx], cmap='viridis', alpha=0.7)
            cbar_mean = plt.colorbar(sc_mean, ax=ax_mean)
            cbar_mean.set_label('Mean')
            ax_mean.set_xlabel('X1')
            ax_mean.set_ylabel('X2')
            ax_mean.set_title(f'{problem_name} - Output {output_idx + 1} (Mean)')
            ax_mean.grid(True, alpha=0.3)
            ax_mean.set_xlim(bounds[0][0], bounds[1][0])
            ax_mean.set_ylim(bounds[0][1], bounds[1][1])
            
            # Std plot
            ax_std = axes[output_idx, 1]
            sc_std = ax_std.scatter(X_np[:, 0], X_np[:, 1], c=std[:, output_idx], cmap='plasma', alpha=0.7)
            cbar_std = plt.colorbar(sc_std, ax=ax_std)
            cbar_std.set_label('Std Dev')
            ax_std.set_xlabel('X1')
            ax_std.set_ylabel('X2')
            ax_std.set_title(f'{problem_name} - Output {output_idx + 1} (Uncertainty)')
            ax_std.grid(True, alpha=0.3)
            ax_std.set_xlim(bounds[0][0], bounds[1][0])
            ax_std.set_ylim(bounds[0][1], bounds[1][1])
            
            # PoF plot
            if pof is not None:
                ax_pof = axes[output_idx, 2]
                sc_pof = ax_pof.scatter(X_np[:, 0], X_np[:, 1], c=pof[:, output_idx], cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)
                cbar_pof = plt.colorbar(sc_pof, ax=ax_pof)
                cbar_pof.set_label('Probability')
                ax_pof.set_xlabel('X1')
                ax_pof.set_ylabel('X2')
                ax_pof.set_title(f'{problem_name} - Output {output_idx + 1} (Probability of Feasibility)')
                ax_pof.grid(True, alpha=0.3)
                ax_pof.set_xlim(bounds[0][0], bounds[1][0])
                ax_pof.set_ylim(bounds[0][1], bounds[1][1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # print("Plotting is only implemented for 2D inputs.")
        pass
        
    
def plot_data(X, Y_feas_all, save_path, problem_name, bounds, num_init_points=0):
    """
    Create scatter plots for 2D and 3D cases of X, or parallel coordinate plots for higher dimensions,
    with different colors based on feasibility, and save to file.
    
    Args:
        X: torch.Tensor of shape (n_samples, n_dims) - input points
        Y_feas_all: torch.Tensor of shape (n_samples,) - feasibility indicator (0 or 1)
        save_path: str - path where to save the plot
        problem_name: str - name of the problem for plot title
        bounds: list of lists - bounds for each dimension [[lower_bounds], [upper_bounds]]
        num_init_points: int - number of initial points to mark differently
    """
    # Convert to numpy for matplotlib
    X_np = X.detach().cpu().numpy()
    Y_feas_np = Y_feas_all.detach().cpu().numpy()
    
    # Define colors for feasible (1) and infeasible (0) points
    colors = ['red', 'blue']  # red for infeasible (0), blue for feasible (1)
    labels = ['Infeasible', 'Feasible']
    
    n_dims = X_np.shape[1]
    
    if n_dims == 2:
        # 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for feasible_val in [0, 1]:
            mask = Y_feas_np == feasible_val
            if mask.any():
                # Split into initial and acquisition points
                init_mask = mask & (np.arange(len(X_np)) < num_init_points)
                acq_mask = mask & (np.arange(len(X_np)) >= num_init_points)
                
                # Plot acquisition points with circles
                if acq_mask.any():
                    ax.scatter(X_np[acq_mask, 0], X_np[acq_mask, 1], 
                              c=colors[feasible_val], 
                              label=labels[feasible_val],
                              alpha=0.7, s=50, marker='o')
                
                # Plot initial points with x markers
                if init_mask.any():
                    ax.scatter(X_np[init_mask, 0], X_np[init_mask, 1], 
                              c=colors[feasible_val], 
                              label=f'{labels[feasible_val]} (Initial)',
                              alpha=0.7, s=80, marker='x', linewidths=3)
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(problem_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set axis limits using bounds
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        
    elif n_dims == 3:
        # 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for feasible_val in [0, 1]:
            mask = Y_feas_np == feasible_val
            if mask.any():
                # Split into initial and acquisition points
                init_mask = mask & (np.arange(len(X_np)) < num_init_points)
                acq_mask = mask & (np.arange(len(X_np)) >= num_init_points)
                
                # Plot acquisition points with circles
                if acq_mask.any():
                    ax.scatter(X_np[acq_mask, 0], X_np[acq_mask, 1], X_np[acq_mask, 2],
                              c=colors[feasible_val], 
                              label=labels[feasible_val],
                              alpha=0.7, s=50, marker='o')
                
                # Plot initial points with x markers
                if init_mask.any():
                    ax.scatter(X_np[init_mask, 0], X_np[init_mask, 1], X_np[init_mask, 2],
                              c=colors[feasible_val], 
                              label=f'{labels[feasible_val]} (Initial)',
                              alpha=0.7, s=80, marker='x', linewidths=3)
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title(problem_name)
        ax.legend()
        
        # Set axis limits using bounds
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        ax.set_zlim(bounds[0][2], bounds[1][2])
        
    else:
        # Parallel coordinate plot for higher dimensional data
        fig, ax = plt.subplots(figsize=(max(10, n_dims * 1.5), 6))
        
        # Create x-axis positions for each dimension
        x_positions = np.arange(n_dims)
        
        # Normalize data to [0, 1] for each dimension using its specific bounds
        X_normalized = np.zeros_like(X_np)
        for dim in range(n_dims):
            dim_range = bounds[1][dim] - bounds[0][dim]
            if dim_range > 0:
                X_normalized[:, dim] = (X_np[:, dim] - bounds[0][dim]) / dim_range
            else:
                X_normalized[:, dim] = 0.5  # If no range, center it
        
        # Plot lines for each data point
        for feasible_val in [0, 1]:
            mask = Y_feas_np == feasible_val
            if mask.any():
                # Split into initial and acquisition points
                init_mask = mask & (np.arange(len(X_np)) < num_init_points)
                acq_mask = mask & (np.arange(len(X_np)) >= num_init_points)
                
                # Plot acquisition points with normal lines
                if acq_mask.any():
                    data_subset = X_normalized[acq_mask]
                    for i in range(data_subset.shape[0]):
                        ax.plot(x_positions, data_subset[i], 
                               color=colors[feasible_val], 
                               alpha=0.3, 
                               linewidth=1)
                
                # Plot initial points with thicker dashed lines
                if init_mask.any():
                    data_subset = X_normalized[init_mask]
                    for i in range(data_subset.shape[0]):
                        ax.plot(x_positions, data_subset[i], 
                               color=colors[feasible_val], 
                               alpha=0.7, 
                               linewidth=2,
                               linestyle='--')
                
                # Add dummy lines for legend
                if acq_mask.any():
                    ax.plot([], [], color=colors[feasible_val], 
                           label=labels[feasible_val], linewidth=2)
                if init_mask.any():
                    ax.plot([], [], color=colors[feasible_val], 
                           label=f'{labels[feasible_val]} (Initial)', 
                           linewidth=2, linestyle='--')
        
        # Customize the plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'X{i+1}' for i in range(n_dims)])
        ax.set_ylabel('Normalized Value')
        ax.set_title(problem_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to [0, 1] and show actual bound values
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Min', 'Max'])
        
        # Add vertical lines at each dimension and show actual bound values
        for x_pos in range(n_dims):
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            # Add actual bound values as text labels
            ax.text(x_pos, -0.05, f'{bounds[0][x_pos]:.2f}', 
                   ha='center', va='top', fontsize=8, rotation=45)
            ax.text(x_pos, 1.05, f'{bounds[1][x_pos]:.2f}', 
                   ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
