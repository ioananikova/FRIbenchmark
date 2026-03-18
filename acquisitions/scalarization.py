from typing import Optional
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch import Tensor
from botorch.utils.transforms import  t_batch_mode_transform
from gpytorch.kernels import ScaleKernel, RBFKernel

class ScalarizationBasedAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model, constraints, bounds, weights) -> None:
        """
        Custom acquisition function that combines probability of feasibility (PoF)
        with posterior variance for each constraint.

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
        self.weights = weights
        self.normalized = False
        self.reference_point = torch.ones(2)
        self.gamma = 1e-4

    def _estimate_probabilities_of_feasibility_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points)
        mus, sigma2s = posterior.mean, posterior.variance # type: ignore
        stddevs = sigma2s.sqrt()
        dist = torch.distributions.normal.Normal(mus, stddevs)
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )

        stddevs = stddevs.prod(dim=-1)
        return probs, stddevs
    
    @t_batch_mode_transform(expected_q=1) # type: ignore
    def forward(self, X:Tensor) -> Tensor: # type: ignore
        """
        Evaluate the custom acquisition function at the given points X.

        Args:
            X: A tensor of candidate points to evaluate.

        Returns:
            A tensor of acquisition function values.
        """
        # Compute the probability of feasibility (PoF) and variance (VAR) for each constraint
        p_feas, stddevs = self._estimate_probabilities_of_feasibility_at_points(X)

        if not self.normalized:
            # calculate normalization offsets and scales for sigmas
            # Normalize sigmas across constraints
            self.scales = stddevs.max(dim=0).values - stddevs.min(dim=0).values + 1e-9
            self.offsets = stddevs.min(dim=0).values
            self.normalized = True
        
        stddevs = (stddevs - self.offsets) / self.scales

        objectives = torch.stack([p_feas, stddevs], dim=-1)  # shape: (num_points, 2)

        # Scalarize using augmented Tchebycheff function
        weighted_diffs = self.weights * (objectives - self.reference_point)
        combined_acqf = torch.min(weighted_diffs, dim=-1).values + self.gamma * torch.sum(weighted_diffs, dim=-1)

        return combined_acqf.squeeze(-1)
