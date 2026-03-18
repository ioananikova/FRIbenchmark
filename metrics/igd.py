import torch


class IGD():
    def __init__(self, reference_set, reference_clusters:dict, input_transform) -> None:
        self.reference_set = reference_set
        self.reference_clusters = reference_clusters
        self.input_transform = input_transform
        self.reference_set_norm = self.input_transform(self.reference_set)
        self.reference_clusters_norm = {
            cluster_num: self.input_transform(cluster_reference_set)
            for cluster_num, cluster_reference_set in self.reference_clusters.items()
        }
        # Diagonal of the unit hypercube in d dimensions = the maximum possible IGD value in normalized space
        self.diagonal = torch.sqrt(torch.tensor(reference_set.shape[1]))

    def _calculate_igd(self, eval_set, reference_set):
        total_distance = 0.0
        for ref_point in reference_set:
            # Compute distances from the reference point to all points in the eval set
            distances = torch.norm(eval_set - ref_point.unsqueeze(0), dim=1)
            # Find the minimum distance (nearest neighbor)
            min_distance = torch.min(distances)
            total_distance += min_distance.item()
        # Average distance across all reference points
        igd_value = total_distance / len(reference_set)
        return igd_value

    def compute(self, eval_set, only_infeasible: bool = False) -> dict:
        """
        Compute the Inverted Generational Distance (IGD) metric.
        Args:
            eval_set: Tensor of shape (n_eval, d) representing the evaluated points.
        Returns:
            igd_values: dict The IGD metric values overall and per cluster.
        """
        eval_set_norm = self.input_transform(eval_set)
        # if only infeasible points are present, return infeasible igd + diagonal value
        if only_infeasible:
            igd_overall = self._calculate_igd(eval_set_norm, self.reference_set_norm)
            igd_values = {"IGD_overall": self.diagonal.item() + igd_overall}
            
            # same for clusters
            for cluster_num, cluster_reference_set in self.reference_clusters_norm.items():
                cluster_igd = self._calculate_igd(eval_set_norm, cluster_reference_set)
                igd_values[f"IGD_cluster_{cluster_num}"] = self.diagonal.item() + cluster_igd
            
            return igd_values
        
        # Overall IGD: for each reference point, find nearest neighbor in eval_set
        igd_overall = self._calculate_igd(eval_set_norm, self.reference_set_norm)
        igd_values = {"IGD_overall": igd_overall}

        # IGD per cluster: for each cluster, use cluster reference set as reference
        for cluster_num, cluster_reference_set in self.reference_clusters_norm.items():
            cluster_igd = self._calculate_igd(eval_set_norm, cluster_reference_set)
            igd_values[f"IGD_cluster_{cluster_num}"] = cluster_igd
        
        return igd_values
    
