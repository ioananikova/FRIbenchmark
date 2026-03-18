import torch
from torch.quasirandom import SobolEngine

from problems import ProblemType
from util import identify_samples_which_satisfy_constraints

def create_test_sets(num_points: int, bounds: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create a test set of points uniformly sampled within the given bounds.

    Args:
        num_points: Number of test points to generate.
        bounds: A tensor of shape (2, d) specifying the lower and upper bounds
                for each dimension.

    Returns:
        A tensor of shape (num_points, d) containing the generated test points.
    """
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    random_points = SobolEngine(dim, scramble=True, seed=1000).draw(num_points)
    scaled_points = lower_bounds + (upper_bounds - lower_bounds) * random_points
    return scaled_points

if __name__ == "__main__":

    # CHANGE HERE: specify the problem
    problem_name = "branin"


    num_points = 5000
    problem = ProblemType.get_problem(problem_name)
    bounds = torch.tensor(ProblemType[problem_name].get_bounds())
    dim = ProblemType[problem_name].num_dimensions()
    constraints = ProblemType[problem_name].get_constraints()
    test_set = create_test_sets(num_points=num_points, bounds=bounds, dim=dim)
    test_set_values = problem(test_set)
    test_set_feasibility = identify_samples_which_satisfy_constraints(test_set_values, constraints)
    all_constraints_satisfied = torch.all(test_set_feasibility, dim=-1)  # Check if all outputs satisfy constraints

    num_successful = torch.sum(all_constraints_satisfied).item()

    torch.save(
        {
            "X_test": test_set,
            "Y_test": test_set_values,
            "Y_test_feas": test_set_feasibility,
            "Y_test_all_feas": all_constraints_satisfied.int()
        },
        f"test_sets/test_set_{problem_name}_{num_successful}_{num_points}.pt",
    )
    