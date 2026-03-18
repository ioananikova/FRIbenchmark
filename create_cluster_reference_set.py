import torch
from torch.quasirandom import SobolEngine
from sklearn.cluster import KMeans, HDBSCAN

from problems import ProblemType
from util import identify_samples_which_satisfy_constraints
from pandas.plotting import parallel_coordinates
import pandas as pd

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
    random_points = SobolEngine(dim, scramble=True, seed=5000).draw(num_points)
    scaled_points = lower_bounds + (upper_bounds - lower_bounds) * random_points
    return scaled_points

def create_clusters(points: torch.Tensor, num_clusters: int) -> dict:
    """
    Create clusters of points using KMeans clustering.

    Args:
        points: A tensor of shape (n_points, d) containing the points to cluster.
        num_clusters: The number of clusters to create.

    Returns:
        A dictionary mapping cluster indices to tensors of points in that cluster.
    """
    clustering = HDBSCAN(min_cluster_size=2, allow_single_cluster=True).fit(points.numpy())
    # clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(points.numpy())
    labels = clustering.labels_
    print(f"Identified {len(set(labels))} - {(1 if -1 in labels else 0)} clusters using HDBSCAN.")
    print(f"Cluster sizes: {[torch.sum(torch.tensor(labels) == i).item() for i in set(labels)]}")
    
    clusters = {}
    for cluster_num in range(num_clusters):
        cluster_points = points[torch.tensor(labels) == cluster_num]
        clusters[cluster_num+1] = cluster_points
    return clusters

def filter_points(points: torch.Tensor, bounds: torch.Tensor, threshold: float = 0.01) -> tuple:
    """
    Filter points to remove those that are too close to each other.
    
    Algorithm:
    1. Scale points back to [0,1] range following the bounds
    2. Find nearest neighbor for each point and sort by distance (descending)
    3. Cycle through sorted points: if nearest neighbor is within threshold, remove the point
    4. Removed points are not considered in future neighbor searches
    
    Args:
        points: A tensor of shape (n_points, d) containing the points to filter.
        bounds: A tensor of shape (2, d) specifying the lower and upper bounds.
        threshold: Distance threshold for removing points.
    
    Returns:
        A tuple of (filtered_points, removed_points) where both are tensors in original (unscaled) coordinates.
    """
    # Step 1: Scale points to [0,1] range
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    scaled_points = (points - lower_bounds) / (upper_bounds - lower_bounds)
    
    # Step 2: Find nearest neighbor for each point and compute distances
    distances_to_nn = []
    nn_indices = []
    
    for i in range(scaled_points.shape[0]):
        # Compute distances to all other points
        diffs = scaled_points - scaled_points[i]
        dists = torch.norm(diffs, dim=1)
        # Set distance to self as infinity to ignore it
        dists[i] = float('inf')
        # Find nearest neighbor distance and index
        min_dist, min_idx = torch.min(dists, dim=0)
        distances_to_nn.append(min_dist.item())
        nn_indices.append(min_idx.item())
    
    distances_to_nn = torch.tensor(distances_to_nn)
    print(f"minimum nearest neighbor distance before filtering: {torch.min(distances_to_nn).item()}")
    print(f"maximum nearest neighbor distance before filtering: {torch.max(distances_to_nn).item()}")
    
    # Step 3: Sort points by nearest neighbor distance (descending order)
    sorted_indices = torch.argsort(distances_to_nn, descending=True)
    
    # Step 4: Cycle through sorted points and filter
    remaining_mask = torch.ones(points.shape[0], dtype=torch.bool)
    
    for idx in sorted_indices:
        if not remaining_mask[idx]:
            # Point already removed, skip
            continue
        
        # Find current nearest neighbor among remaining points
        remaining_indices = torch.where(remaining_mask)[0]
        if remaining_indices.shape[0] <= 1:
            # Only one point left, stop filtering
            break
        
        # Compute distances to remaining points
        diffs = scaled_points[remaining_indices] - scaled_points[idx]
        dists = torch.norm(diffs, dim=1)
        
        # Find nearest neighbor (excluding self)
        min_dist = torch.min(dists[dists > 0])
        
        # If nearest neighbor is within threshold, remove this point
        if min_dist < threshold:
            remaining_mask[idx] = False
    
    # Return filtered points and removed points in original coordinates
    print(f"Number of points before filtering: {points.shape[0]}")
    print(f"Number of points after filtering: {torch.sum(remaining_mask).item()}")
    filtered_points = points[remaining_mask]
    removed_points = points[~remaining_mask]
    return filtered_points, removed_points


if __name__ == "__main__":

    # CHANGE HERE: specify the problem
    problem_name = "speedreducer"
    num_clusters = 1 # need to change this only if you want to use kmeans instead of HDBSCAN. 
    #HDBSCAN can be inaccurate for some problems, so check the number with the ones given in the table of the paper.
    num_points = 100000

    
    problem = ProblemType.get_problem(problem_name)
    bounds = torch.tensor(ProblemType[problem_name].get_bounds())
    dim = ProblemType[problem_name].num_dimensions()
    constraints = ProblemType[problem_name].get_constraints()
    test_set = create_test_sets(num_points=num_points, bounds=bounds, dim=dim)
    test_set_values = problem(test_set)
    test_set_feasibility = identify_samples_which_satisfy_constraints(test_set_values, constraints)
    all_constraints_satisfied = torch.all(test_set_feasibility, dim=-1)  # Check if all outputs satisfy constraints
    feasible_inputs = test_set[all_constraints_satisfied.bool() == 1]

    num_successful = torch.sum(all_constraints_satisfied).item()

    clusters = create_clusters(feasible_inputs, num_clusters=num_clusters)

    num_successful_filtered = 0
    removed_points_by_cluster = {}
    for cluster_num in clusters:
        cluster_points = clusters[cluster_num]
        filtered_cluster_points, removed_cluster_points = filter_points(cluster_points, bounds, threshold=0.005)
        clusters[cluster_num] = filtered_cluster_points
        removed_points_by_cluster[cluster_num] = removed_cluster_points
        num_successful_filtered += filtered_cluster_points.shape[0]

    # Plot feasible inputs for 2D problems
    if dim == 2:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        for cluster_num, cluster_points in clusters.items():
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=2, label=f'Cluster {cluster_num}')
        
        # Plot removed points
        for cluster_num, removed_points in removed_points_by_cluster.items():
            if removed_points.shape[0] > 0:
                plt.scatter(removed_points[:, 0], removed_points[:, 1], s=2, marker='x', 
                           label=f'Removed (Cluster {cluster_num})')
        
        plt.xlim(bounds[0, 0].item(), bounds[1, 0].item())
        plt.ylim(bounds[0, 1].item(), bounds[1, 1].item())
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f'Feasible Regions: {problem_name} ({num_successful_filtered} feasible)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster_num, cluster_points in clusters.items():
            df = pd.DataFrame(cluster_points.numpy())
            df['Cluster'] = f'Cluster {cluster_num}'
            parallel_coordinates(df, 'Cluster', ax=ax, alpha=0.3)

        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Value')
        ax.set_title(f'Parallel Coordinates: {problem_name} ({num_successful_filtered} feasible)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    torch.save(
        {
            "X_ref_feas": feasible_inputs,
            "X_ref_clusters": clusters,
        },
        f"test_sets/reference_clusters_{problem_name}_{num_successful}_{num_successful_filtered}_{num_points}.pt",
    )