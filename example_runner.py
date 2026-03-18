import os
import random
import warnings

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import AffineInputTransform

from acquisitions.cas_eci import ExpectedCoverageImprovement
from acquisitions.pofvar import PofVarAcquisition
from acquisitions.scalarization import ScalarizationBasedAcquisition
from acquisitions.boundary_methods import ProbabilityBoundaryEntropy, EntropyFeasibleAcquisition
from problems import ProblemType
from util import get_and_fit_gp, identify_samples_which_satisfy_constraints, plot_data
from metrics.igd import IGD
from metrics.model_metrics import ModelMetrics

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    tkwargs = {
            "device": torch.device("cpu"),
            "dtype": torch.double,
        }
    torch.set_default_dtype(torch.double)
    
    print(f"Using device: {tkwargs['device']}")

    # Define experiment configuration
    config={
            "seed": 1234,
            # problem parameters
            "problem_name": "branin",
            # test parameters
            "test_set_path": "test_sets/test_set_branin_212_5000.pt", 
            "reference_set_path": "test_sets/reference_clusters_branin_826_600_20000.pt",
            # general BO parameters
            "init_points": 5,
            "num_acq_points": 5,
            "batch_size": 1,
            # acquisition function parameters
            "acquisition_function": "PoFV",  # "ECI" or "PoFV" or "EF" or "PBE" "ATCH"
        }

    # PROBLEM SETUP
    problem_func = ProblemType.get_problem(config["problem_name"])

    config["dimensions"] = ProblemType[config["problem_name"]].num_dimensions()
    config["bounds"] = ProblemType[config["problem_name"]].get_bounds()
    config["constraints"] = ProblemType[config["problem_name"]].get_constraints()
    config["num_constraints"] = ProblemType[config["problem_name"]].num_constraints()
    
    # Set up bounds tensor
    bounds = torch.tensor(config["bounds"], **tkwargs)
    lb, ub = bounds
    input_transform = AffineInputTransform(d=config["dimensions"], coefficient=ub-lb, offset=lb)
    
    # LOAD TEST SET AND REFERENCE SET FOR IGD
    test_set_data = torch.load(config["test_set_path"])
    X_test = test_set_data["X_test"].to(**tkwargs)
    Y_test = test_set_data["Y_test"].to(**tkwargs)
    Y_test_feas = test_set_data["Y_test_feas"].to(**tkwargs) # NOTE shape (num_points, num_constraints)
    Y_test_all_feas = test_set_data["Y_test_all_feas"].to(**tkwargs) # NOTE shape (num_points, )

    reference_clusters = torch.load(config["reference_set_path"])
    igd = IGD(reference_set=reference_clusters["X_ref_feas"], reference_clusters=reference_clusters["X_ref_clusters"], input_transform=input_transform)

    # MODEL METRICS SETUP
    model_metrics = ModelMetrics(X_test, Y_test, Y_test_feas, Y_test_all_feas, config["constraints"])

    # Set random seeds for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    log_metrics = {}
    os.makedirs(f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}", exist_ok=True)

    # INITIAL DATASET
    X = lb + (ub - lb) * SobolEngine(config["dimensions"], scramble=True, seed=config["seed"]).draw(config["init_points"]).to(**tkwargs)
    Y = problem_func(X)
    Y_feas = identify_samples_which_satisfy_constraints(Y, config["constraints"])
    Y_feas_all = torch.all(Y_feas, dim=-1).int().to(**tkwargs)
    torch.save(
        {
            "X": X,
            "Y": Y,
            "Y_feas": Y_feas,
            "Y_feas_all": Y_feas_all,
        },
        f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/dataset_0.pt",
    )

    # METRICS ON INITIAL DATASET
    # Calculate total number of successful designs
    # successful is a boolean tensor, we need all constraints to be satisfied for each sample
    all_constraints_satisfied = torch.all(Y_feas, dim=-1)  # Check if all outputs satisfy constraints
    num_successful = torch.sum(all_constraints_satisfied).item()

    log_metrics.update({
        "num_points": len(X),
        "num_successful_designs": num_successful,
        "success_rate": num_successful / len(X)  
    })

    eval_set = X[Y_feas_all.bool() == 1] if num_successful > 0 else X[Y_feas_all.bool() == 0]
    igd_values = igd.compute(eval_set=eval_set, only_infeasible=num_successful==0)
    log_metrics.update(igd_values)

    gp_models = [get_and_fit_gp(X, Y[:, i : i + 1], inp_transform=input_transform) for i in range(Y.shape[-1])]
    model_list_gp = ModelListGP(*gp_models)
    torch.save(model_list_gp.state_dict(), f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/model_0.pt")

    model_metrics_values = model_metrics.compute(model_list_gp, tkwargs)
    log_metrics.update(model_metrics_values)

    torch.save(log_metrics, f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/metrics_0.pt")
    
    # WEIGHTS FOR SCALARIZATION
    if config["acquisition_function"] == "ATCH":
        weights_1 = torch.rand(config["num_acq_points"], **tkwargs)
        weights_2 = 1 - weights_1
        weights = torch.stack([weights_1, weights_2], dim=-1)  # shape: (num_acq_points, 2)
    else:
        weights = None

    for iter in range(config["num_acq_points"]):
        
        log_metrics = {}

        if config["acquisition_function"] == "ECI":
            acq = ExpectedCoverageImprovement(
                model=model_list_gp, 
                constraints=config["constraints"], 
                bounds=bounds,
                input_transform=input_transform,
                num_samples=128,
                punchout_radius=0.1
            )
        elif config["acquisition_function"] == "PoFV":
            acq = PofVarAcquisition(
                model=model_list_gp, 
                constraints=config["constraints"], 
                bounds=bounds
            )
        elif config["acquisition_function"] == "EF":
            acq = EntropyFeasibleAcquisition(
                model=model_list_gp, 
                constraints=config["constraints"], 
                bounds=bounds
            )
        elif config["acquisition_function"] == "PBE":
            acq = ProbabilityBoundaryEntropy(
                model=model_list_gp, 
                constraints=config["constraints"], 
                bounds=bounds
            )
        elif config["acquisition_function"] == "ATCH":
            assert weights is not None, "Weights must be defined for ATCH acquisition function"
            acq = ScalarizationBasedAcquisition(
                model=model_list_gp,
                constraints=config["constraints"],
                bounds=bounds,
                weights=weights[iter]
            )
        else:
            raise ValueError(f"Unknown acquisition function: {config['acquisition_function']}")
        
        x_next, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=config["batch_size"],
            num_restarts=10,
            raw_samples=512,
        )

        log_metrics.update({
            "acq_value": acq_value.item(),
        })

        # EVALUATE NEW POINT AND UPDATE DATASET
        y_next = problem_func(x_next)
        X = torch.cat((X, x_next))
        Y = torch.cat((Y, y_next))
        Y_feas = identify_samples_which_satisfy_constraints(Y, config['constraints'])
        Y_feas_all = torch.all(Y_feas, dim=-1).int().to(**tkwargs)
        torch.save(
            {
                "X": X,
                "Y": Y,
                "Y_feas": Y_feas,
                "Y_feas_all": Y_feas_all,
            },
            f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/dataset_{iter+1}.pt",
        )

        # METRICS ON UPDATED DATASET
        # Calculate total number of successful designs
        # successful is a boolean tensor, we need all constraints to be satisfied for each sample
        all_constraints_satisfied = torch.all(Y_feas, dim=-1)  # Check if all outputs satisfy constraints
        num_successful = torch.sum(all_constraints_satisfied).item()

        log_metrics.update({
            "num_points": len(X),
            "num_successful_designs": num_successful,
            "success_rate": num_successful / len(X)
        })
        eval_set = X[Y_feas_all.bool() == 1] if num_successful > 0 else X[Y_feas_all.bool() == 0]
        igd_values = igd.compute(eval_set=eval_set, only_infeasible=num_successful==0)
        log_metrics.update(igd_values)

        # MODEL AND MODEL METRICS ON UPDATED DATASET
        gp_models = [get_and_fit_gp(X, Y[:, i : i + 1], inp_transform=input_transform) for i in range(Y.shape[-1])]
        model_list_gp = ModelListGP(*gp_models)
        torch.save(model_list_gp.state_dict(), f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/model_{iter+1}.pt")

        model_metrics_values = model_metrics.compute(model_list_gp, tkwargs)
        log_metrics.update(model_metrics_values)

        torch.save(log_metrics, f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/metrics_{iter+1}.pt")

    plot_data(X, Y_feas_all, f"experiments/{config['problem_name']}_{config['seed']}_{config['acquisition_function']}/plot_{iter+1}.png", config["problem_name"], config["bounds"], config["init_points"])