"""
Inference utilities for ARC models.

Provides functions for running model predictions on grids and tasks.
"""

from typing import List, Dict, Any, Callable, Optional, Union
import torch
import torch.nn as nn
from .data_utils import ARCTaskData


# Type aliases for clarity
Grid = List[List[int]]
PredictFn = Callable[[nn.Module, Grid], Grid]


def predict_on_grid(
    model: nn.Module,
    input_grid: Grid,
    predict_fn: Optional[PredictFn] = None,
    device: str = "cpu",
) -> Grid:
    """
    Run a single prediction on an input grid.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction
    input_grid : list of list of int
        2D grid of integers (ARC input)
    predict_fn : callable, optional
        Custom prediction function with signature (model, grid) -> grid.
        If None, assumes model.forward() returns the output.
        Recommended to provide this for model-specific preprocessing/postprocessing.
    device : str
        Device to run on ("cpu" or "cuda")
        
    Returns
    -------
    output_grid : list of list of int
        Predicted output grid
        
    Raises
    ------
    RuntimeError
        If prediction fails
    """
    
    model.to(device)
    model.eval()
    
    try:
        with torch.no_grad():
            if predict_fn is not None:
                # Use custom prediction function
                output_grid = predict_fn(model, input_grid)
            else:
                # Direct model inference (assumes model expects and returns grids)
                # This is a fallback; custom predict_fn is recommended
                input_tensor = _grid_to_tensor(input_grid).to(device)
                output_tensor = model(input_tensor)
                output_grid = _tensor_to_grid(output_tensor)
        
        return output_grid
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


def predict_on_task(
    model: nn.Module,
    task_data: ARCTaskData,
    predict_fn: Optional[PredictFn] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run predictions on all test examples in a task.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use
    task_data : ARCTaskData
        Task data with test inputs
    predict_fn : callable, optional
        Custom prediction function
    device : str
        Device to run on
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary with:
        - task_id: str
        - predictions: list of output grids
        - errors: list of error messages (empty strings if successful)
        - num_successful: int
        - num_failed: int
    """
    
    results = {
        "task_id": task_data.task_id,
        "predictions": [],
        "errors": [],
        "num_successful": 0,
        "num_failed": 0,
    }
    
    model.eval()
    
    for idx, pair in enumerate(task_data.test_pairs):
        input_grid = pair.get("input")
        
        if input_grid is None:
            results["predictions"].append(None)
            results["errors"].append("No input grid")
            results["num_failed"] += 1
            continue
        
        try:
            prediction = predict_on_grid(
                model,
                input_grid,
                predict_fn=predict_fn,
                device=device,
            )
            
            results["predictions"].append(prediction)
            results["errors"].append("")
            results["num_successful"] += 1
            
            if verbose:
                print(f"  {task_data.task_id} test {idx}: ✓")
        
        except Exception as e:
            results["predictions"].append(None)
            results["errors"].append(str(e))
            results["num_failed"] += 1
            
            if verbose:
                print(f"  {task_data.task_id} test {idx}: ✗ ({e})")
    
    return results


def batch_predict(
    model: nn.Module,
    input_grids: List[Grid],
    predict_fn: Optional[PredictFn] = None,
    device: str = "cpu",
    batch_size: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run predictions on multiple grids (batch mode).
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use
    input_grids : list of grids
        List of input grids
    predict_fn : callable, optional
        Custom prediction function
    device : str
        Device to run on
    batch_size : int
        How many grids to process per batch
        (Note: actual batching logic depends on predict_fn implementation)
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary with:
        - predictions: list of output grids
        - num_successful: int
        - num_failed: int
        - errors: list of error messages
    """
    
    results = {
        "predictions": [],
        "num_successful": 0,
        "num_failed": 0,
        "errors": [],
    }
    
    model.eval()
    
    # Process grids
    for idx, input_grid in enumerate(input_grids):
        try:
            prediction = predict_on_grid(
                model,
                input_grid,
                predict_fn=predict_fn,
                device=device,
            )
            
            results["predictions"].append(prediction)
            results["num_successful"] += 1
            results["errors"].append("")
            
            if verbose and (idx + 1) % batch_size == 0:
                print(f"  Processed {idx + 1}/{len(input_grids)} grids")
        
        except Exception as e:
            results["predictions"].append(None)
            results["num_failed"] += 1
            results["errors"].append(str(e))
            
            if verbose:
                print(f"  Grid {idx}: Error - {e}")
    
    return results


def predict_with_ensemble(
    models: List[nn.Module],
    input_grid: Grid,
    predict_fn: Optional[PredictFn] = None,
    ensemble_strategy: str = "majority_vote",
    device: str = "cpu",
) -> Grid:
    """
    Run prediction using an ensemble of models.
    
    Parameters
    ----------
    models : list of torch.nn.Module
        Multiple models for ensemble
    input_grid : list of list of int
        Input grid
    predict_fn : callable, optional
        Prediction function (same function used for all models)
    ensemble_strategy : str
        How to combine predictions:
        - "majority_vote": use most common prediction (requires exact grid matching)
        - "average": average predictions at each cell (for continuous outputs)
    device : str
        Device to run on
        
    Returns
    -------
    output_grid : list of list of int
        Ensemble prediction
        
    Raises
    ------
    ValueError
        If strategy is unknown or ensemble voting fails
    """
    
    predictions = []
    
    for model in models:
        pred = predict_on_grid(
            model,
            input_grid,
            predict_fn=predict_fn,
            device=device,
        )
        predictions.append(pred)
    
    if ensemble_strategy == "majority_vote":
        return _ensemble_majority_vote(predictions)
    elif ensemble_strategy == "average":
        return _ensemble_average(predictions)
    else:
        raise ValueError(f"Unknown ensemble strategy: {ensemble_strategy}")


def _ensemble_majority_vote(predictions: List[Grid]) -> Grid:
    """
    Combine predictions via majority voting on grid cells.
    """
    if not predictions:
        raise ValueError("No predictions to ensemble")
    
    first_grid = predictions[0]
    height = len(first_grid)
    width = len(first_grid[0]) if first_grid else 0
    
    # Check all predictions have same shape
    for pred in predictions:
        if len(pred) != height or (pred and len(pred[0]) != width):
            raise ValueError("All predictions must have the same shape for majority voting")
    
    result = []
    for i in range(height):
        row = []
        for j in range(width):
            # Collect votes for this cell
            votes = [pred[i][j] for pred in predictions]
            # Majority vote: most common value
            most_common = max(set(votes), key=votes.count)
            row.append(most_common)
        result.append(row)
    
    return result


def _ensemble_average(predictions: List[Grid]) -> Grid:
    """
    Combine predictions via averaging on grid cells.
    """
    if not predictions:
        raise ValueError("No predictions to ensemble")
    
    first_grid = predictions[0]
    height = len(first_grid)
    width = len(first_grid[0]) if first_grid else 0
    
    # Check all predictions have same shape
    for pred in predictions:
        if len(pred) != height or (pred and len(pred[0]) != width):
            raise ValueError("All predictions must have the same shape for averaging")
    
    result = []
    for i in range(height):
        row = []
        for j in range(width):
            # Average the values at this cell
            values = [pred[i][j] for pred in predictions]
            avg_value = sum(values) / len(values)
            # Round to nearest integer for discrete colors
            row.append(int(round(avg_value)))
        result.append(row)
    
    return result


def _grid_to_tensor(grid: Grid, device: str = "cpu") -> torch.Tensor:
    """
    Convert a 2D grid to a PyTorch tensor.
    
    Parameters
    ----------
    grid : list of list of int
        2D grid
    device : str
        Device to place tensor on
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (1, 1, H, W) with dtype float32
    """
    tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def _tensor_to_grid(tensor: torch.Tensor) -> Grid:
    """
    Convert a PyTorch tensor back to a 2D grid.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor (assumes shape (1, 1, H, W) or (H, W))
        
    Returns
    -------
    grid : list of list of int
        2D grid
    """
    # Squeeze out batch and channel dims if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    # Convert to CPU and numpy, then to list of lists
    grid = tensor.cpu().detach().numpy()
    # Round to nearest integer (colors are 0-9)
    grid = [[int(round(val)) for val in row] for row in grid]
    
    return grid


def create_predict_fn_wrapper(
    preprocess_fn: Optional[Callable[[Grid], torch.Tensor]] = None,
    postprocess_fn: Optional[Callable[[torch.Tensor], Grid]] = None,
) -> PredictFn:
    """
    Create a prediction function with custom preprocessing/postprocessing.
    
    Parameters
    ----------
    preprocess_fn : callable, optional
        Function to convert grid -> tensor: (grid) -> tensor
    postprocess_fn : callable, optional
        Function to convert tensor -> grid: (tensor) -> grid
        
    Returns
    -------
    predict_fn : callable
        Prediction function with signature (model, grid) -> grid
        
    Example
    -------
    >>> def my_preprocess(grid):
    ...     # Custom preprocessing logic
    ...     return torch.tensor(grid, dtype=torch.float32)
    >>> 
    >>> predict_fn = create_predict_fn_wrapper(
    ...     preprocess_fn=my_preprocess,
    ...     postprocess_fn=_tensor_to_grid,
    ... )
    """
    
    def predict_fn(model: nn.Module, grid: Grid) -> Grid:
        with torch.no_grad():
            # Preprocess
            if preprocess_fn is not None:
                tensor = preprocess_fn(grid)
            else:
                tensor = _grid_to_tensor(grid)
            
            # Forward pass
            output = model(tensor)
            
            # Postprocess
            if postprocess_fn is not None:
                result = postprocess_fn(output)
            else:
                result = _tensor_to_grid(output)
        
        return result
    
    return predict_fn
