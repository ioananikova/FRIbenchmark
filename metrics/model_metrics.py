import torch
from torchmetrics.functional import confusion_matrix, matthews_corrcoef, f1_score, precision, recall, specificity

from util import identify_samples_which_satisfy_constraints


class ModelMetrics():
    def __init__(self, X_test: torch.Tensor, Y_test: torch.Tensor, Y_test_feas: torch.Tensor, Y_test_all_feas: torch.Tensor, constraints) -> None:
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_test_feas = Y_test_feas
        self.Y_test_all_feas = Y_test_all_feas
        self.constraints = constraints
        self.num_constraints = len(constraints)

    def compute(self, model, tkwargs) -> dict:
        """
        Compute model metrics on the test set.
        Args:
            model: The trained model to evaluate.
            tkwargs: Tensor device and dtype information.
        Returns:
            metrics: dict The computed metrics.
        """
        model.eval()
        posterior = model.posterior(self.X_test)
        mus = posterior.mean
        Y_predicted = mus.to(**tkwargs)
        Y_predicted_feas = identify_samples_which_satisfy_constraints(Y_predicted, self.constraints)
        y_predicted_all_feas = torch.all(Y_predicted_feas, dim=-1).int().to(**tkwargs)

        metrics = self._compute_single(y_predicted_all_feas, self.Y_test_all_feas)

        for c in range(self.num_constraints):
                y_predicted_feas_c = Y_predicted_feas[:, c]
                y_test_feas_c = self.Y_test_feas[:, c]
                c_metrics = self._compute_single(y_predicted_feas_c, y_test_feas_c, constraint=c+1)
                metrics.update(c_metrics)

        return metrics
    
    def _compute_single(self, pred, true, constraint=0) -> dict:
        cm = confusion_matrix(pred, true, task="binary") # NOTE: [[TN, FP], [FN, TP]]
        mcc = matthews_corrcoef(pred, true, task="binary")
        f1 = f1_score(pred, true, task="binary")
        prec = precision(pred, true, task="binary")
        rec = recall(pred, true, task="binary")
        spec = specificity(pred, true, task="binary")
        informedness = rec + spec - 1.0
        if constraint == 0:
            metrics = {
                "TN": cm[0,0].item(),
                "FP": cm[0,1].item(), 
                "FN": cm[1,0].item(),
                "TP": cm[1,1].item(),
                "matthews_corrcoef": mcc.item(),
                "f1_score": f1.item(),
                "precision": prec.item(),
                "recall": rec.item(),
                "specificity": spec.item(),
                "informedness": informedness.item()
            }
        else:
            metrics = {
                f"c{constraint}_TN": cm[0,0].item(),
                f"c{constraint}_FP": cm[0,1].item(), 
                f"c{constraint}_FN": cm[1,0].item(),
                f"c{constraint}_TP": cm[1,1].item(),
                f"c{constraint}_matthews_corrcoef": mcc.item(),
                f"c{constraint}_f1_score": f1.item(),
                f"c{constraint}_precision": prec.item(),
                f"c{constraint}_recall": rec.item(),
                f"c{constraint}_specificity": spec.item(),
                f"c{constraint}_informedness": informedness.item()
            }
        return metrics