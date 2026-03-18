import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch import Tensor
from botorch.utils.transforms import  t_batch_mode_transform
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior

class ProbabilityBoundaryEntropy(AnalyticAcquisitionFunction):
    def __init__(self, model, constraints, bounds) -> None:
        """
        Custom acquisition function that combines probability of feasibility (PoF)
        with boundary entropy for each constraint.

        Args:
            model: A fitted GP model.
            bounds: A dictionary specifying the bounds for each constraint.
                    Format: {constraint_index: (lower_bound, upper_bound)}.
            multiplier: A scaling factor for the acquisition function.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        self.constraints = constraints
        self.bounds = bounds
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)

    def _estimate_probabilities_of_feasibility_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points)
        if isinstance(posterior, GaussianMixturePosterior):
            mus = posterior.mixture_mean
            sigma2s = posterior.mixture_variance
        else:
            mus = posterior.mean
            sigma2s = posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs, sigma2s
    
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X:Tensor) -> Tensor:
        # Compute the probability of feasibility (PoF) and variance (VAR) for each constraint
        p_feas, sigma2s = self._estimate_probabilities_of_feasibility_at_points(X)
        # Clip sigma2s to avoid log(0)
        sigma2s_clipped = torch.clamp(sigma2s, min=1e-9)
        acq = p_feas * (1 - p_feas) * sigma2s_clipped.prod(dim=-1)
        return acq.squeeze(-1)
    

class EntropyFeasibleAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model, constraints, bounds) -> None:
        """
        Custom acquisition function that combines entropy with feasibility.

        Args:
            model: A fitted GP model.
            bounds: A dictionary specifying the bounds for each constraint.
                    Format: {constraint_index: (lower_bound, upper_bound)}.
            multiplier: A scaling factor for the acquisition function.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        self.constraints = constraints
        self.bounds = bounds

        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        
        # Create alpha_k (lower thresholds) and beta_k (upper thresholds)
        alpha_k_list = []
        beta_k_list = []
        numerically_stable_inf = 1e10
        for direction, threshold in self.constraints:
            if direction == "gt":
                # Greater than: lower threshold is the value, upper is infinity
                alpha_k_list.append(threshold)
                beta_k_list.append(numerically_stable_inf)
            else:  # direction == "lt"
                # Less than: upper threshold is the value, lower is -infinity
                alpha_k_list.append(-numerically_stable_inf)
                beta_k_list.append(threshold)
        
        self.alpha_k = torch.tensor(alpha_k_list).to(bounds)
        self.beta_k = torch.tensor(beta_k_list).to(bounds)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X:Tensor) -> Tensor:

        posterior = self.model.posterior(X)
        if isinstance(posterior, GaussianMixturePosterior):
            mus = posterior.mixture_mean
            sigma2s = posterior.mixture_variance
        else:
            mus = posterior.mean
            sigma2s = posterior.variance
        total_entropy = torch.zeros(X.shape[0]).to(X)

        for k in range(len(self.constraints)):
            mu = mus[..., k]
            sigma = sigma2s[..., k].sqrt()
            sigma = torch.clamp(sigma, min=1e-9)
            dist = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(mu))
            upper_cdf = dist.cdf((self.beta_k[k] - mu) / sigma)
            lower_cdf = dist.cdf((self.alpha_k[k] - mu) / sigma)
            pf = torch.clamp(upper_cdf - lower_cdf, min=1e-12, max=1.0 - 1e-12)
            entropy = -(pf * torch.log(pf) + (1 - pf) * torch.log(1 - pf))
            total_entropy = total_entropy + entropy.squeeze(-1)
        
        return total_entropy