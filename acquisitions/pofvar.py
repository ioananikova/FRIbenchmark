import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch import Tensor
from botorch.utils.transforms import  t_batch_mode_transform

class PofVarAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model, constraints, bounds) -> None:
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

    def _estimate_probabilities_of_feasibility_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points)
        mus, sigma2s = posterior.mean, posterior.variance
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
        """
        Evaluate the custom acquisition function at the given points X.

        Args:
            X: A tensor of candidate points to evaluate.

        Returns:
            A tensor of acquisition function values.
        """
        # Compute the probability of feasibility (PoF) and variance (VAR) for each constraint
        p_feas, sigma2s = self._estimate_probabilities_of_feasibility_at_points(X)

        # Multiply the sigmas of all constraints to get one value per X entry
        combined_sigmas = sigma2s.prod(dim=-1)  # Multiply across constraints

        # Multiply the combined sigmas with the feasibility probability
        combined_acqf = (combined_sigmas * p_feas).prod(dim=-1)  # p_feas is already aggregated

        return combined_acqf
