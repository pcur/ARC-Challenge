"""
Evaluation utilities for ARC models.

Provides functions for evaluating model performance on tasks and datasets.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import torch
from .data_utils import ARCTaskData


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    
    task_id: str
    num_train_pairs: int
    num_test_pairs: int
    
    # Per-task metrics
    correct_predictions: int = 0
    total_predictions: int = 0
    
    # Loss metrics (if applicable)
    total_loss: float = 0.0
    num_batches: int = 0
    
    # Detailed results
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def average_loss(self) -> float:
        """Average loss across batches."""
        if self.num_batches == 0:
            return float('nan')
        return self.total_loss / self.num_batches
    
    def __str__(self) -> str:
        lines = [
            f"Task: {self.task_id}",
            f"  Train pairs: {self.num_train_pairs}",
            f"  Test pairs: {self.num_test_pairs}",
            f"  Accuracy: {self.accuracy:.2%}",
        ]
        if self.num_batches > 0:
            lines.append(f"  Avg Loss: {self.average_loss:.6f}")
        return "\n".join(lines)


def grids_equal(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """
    Check if two grids are equal.
    
    Parameters
    ----------
    grid1, grid2 : list of list of int
        Grids to compare
        
    Returns
    -------
    bool
        True if grids are identical
    """
    if len(grid1) != len(grid2):
        return False
    
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if list(row1) != list(row2):
            return False
    
    return True


def evaluate_model_on_task(
    model: torch.nn.Module,
    task_data: ARCTaskData,
    predict_fn: Callable[[torch.nn.Module, List[List[int]]], List[List[int]]],
    device: str = "cpu",
    verbose: bool = False,
) -> EvaluationMetrics:
    """
    Evaluate a model on a single ARC task.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    task_data : ARCTaskData
        The task data containing train and test pairs
    predict_fn : callable
        Function with signature (model, input_grid) -> output_grid
        Should return a list of lists representing the output grid
    device : str
        Device to run model on ("cpu" or "cuda")
    verbose : bool
        Whether to print detailed results
        
    Returns
    -------
    EvaluationMetrics
        Evaluation results and metrics
    """
    model.eval()
    metrics = EvaluationMetrics(
        task_id=task_data.task_id,
        num_train_pairs=len(task_data.train_pairs),
        num_test_pairs=len(task_data.test_pairs),
    )
    
    # Evaluate on test pairs if available
    for idx, pair in enumerate(task_data.test_pairs):
        input_grid = pair.get("input")
        expected_output = pair.get("output")
        
        if input_grid is None or expected_output is None:
            continue
        
        try:
            # Get prediction from model
            with torch.no_grad():
                predicted_output = predict_fn(model, input_grid)
            
            # Check if prediction matches expected output
            is_correct = grids_equal(predicted_output, expected_output)
            
            metrics.correct_predictions += int(is_correct)
            metrics.total_predictions += 1
            
            result = {
                "pair_idx": idx,
                "correct": is_correct,
                "input_shape": (len(input_grid), len(input_grid[0]) if input_grid else 0),
                "output_shape": (len(expected_output), len(expected_output[0]) if expected_output else 0),
            }
            metrics.results.append(result)
            
            if verbose:
                print(f"  Pair {idx}: {'✓' if is_correct else '✗'}")
        
        except Exception as e:
            if verbose:
                print(f"  Pair {idx}: Error - {e}")
            metrics.results.append({
                "pair_idx": idx,
                "correct": False,
                "error": str(e),
            })
    
    return metrics


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    tasks: Dict[str, ARCTaskData],
    predict_fn: Callable[[torch.nn.Module, List[List[int]]], List[List[int]]],
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate a model on a set of ARC tasks.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    tasks : dict
        Dictionary of task_id -> ARCTaskData
    predict_fn : callable
        Function with signature (model, input_grid) -> output_grid
    device : str
        Device to run model on
    verbose : bool
        Whether to print detailed results
        
    Returns
    -------
    dict
        Dictionary of task_id -> EvaluationMetrics
    """
    results = {}
    
    for task_id, task_data in tasks.items():
        if verbose:
            print(f"Evaluating {task_id}...")
        
        metrics = evaluate_model_on_task(
            model,
            task_data,
            predict_fn,
            device=device,
            verbose=verbose,
        )
        results[task_id] = metrics
        
        if verbose:
            print(f"  {metrics}")
    
    return results


def compute_dataset_statistics(
    results: Dict[str, EvaluationMetrics],
) -> Dict[str, float]:
    """
    Compute aggregate statistics across multiple task evaluations.
    
    Parameters
    ----------
    results : dict
        Dictionary of task_id -> EvaluationMetrics
        
    Returns
    -------
    dict
        Dictionary with aggregate statistics:
        - mean_accuracy: average accuracy across all tasks
        - total_correct: total correct predictions
        - total_predictions: total predictions made
        - num_tasks_evaluated: number of tasks
    """
    total_correct = 0
    total_predictions = 0
    num_tasks = 0
    
    for metrics in results.values():
        total_correct += metrics.correct_predictions
        total_predictions += metrics.total_predictions
        if metrics.total_predictions > 0:
            num_tasks += 1
    
    mean_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "mean_accuracy": mean_accuracy,
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "num_tasks_evaluated": num_tasks,
    }


def print_evaluation_summary(
    results: Dict[str, EvaluationMetrics],
    sort_by: str = "accuracy",
) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Parameters
    ----------
    results : dict
        Dictionary of task_id -> EvaluationMetrics
    sort_by : str
        Sort results by "accuracy", "task_id", or "test_pairs"
    """
    # Sort results
    if sort_by == "accuracy":
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True,
        )
    elif sort_by == "test_pairs":
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].num_test_pairs,
            reverse=True,
        )
    else:  # sort_by == "task_id"
        sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    total_correct = 0
    total_predictions = 0
    
    for task_id, metrics in sorted_results:
        total_correct += metrics.correct_predictions
        total_predictions += metrics.total_predictions
        
        acc_str = f"{metrics.accuracy:.1%}"
        test_str = f"{metrics.total_predictions} test"
        print(f"  {task_id:15s} | {acc_str:>6s} | {test_str:>8s}")
    
    print("-"*70)
    overall_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
    print(f"  {'OVERALL':15s} | {overall_acc:.1%} | {total_predictions} total")
    print("="*70 + "\n")


def compute_task_difficulty(
    predict_fn: Callable[[torch.nn.Module, List[List[int]]], List[List[int]]],
    task_data: ARCTaskData,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """
    Analyze task difficulty based on grid statistics.
    
    This is a heuristic measure that doesn't require model evaluation.
    
    Parameters
    ----------
    predict_fn : callable
        Prediction function (not used, for API consistency)
    task_data : ARCTaskData
        The task to analyze
    model : torch.nn.Module, optional
        Model (not used, for API consistency)
        
    Returns
    -------
    dict
        Difficulty analysis with:
        - avg_input_size: average input grid size
        - avg_output_size: average output grid size
        - avg_size_change: average ratio of output to input size
        - num_colors_range: (min, max) colors across all grids
    """
    from .data_utils import get_grid_dimensions, grid_stats
    
    input_sizes = []
    output_sizes = []
    all_color_counts = set()
    
    for pair in task_data.train_pairs + task_data.test_pairs:
        if "input" in pair:
            h, w = get_grid_dimensions(pair["input"])
            input_sizes.append(h * w)
            all_color_counts.update(grid_stats(pair["input"])["color_counts"].keys())
        
        if "output" in pair:
            h, w = get_grid_dimensions(pair["output"])
            output_sizes.append(h * w)
    
    avg_input = sum(input_sizes) / len(input_sizes) if input_sizes else 0
    avg_output = sum(output_sizes) / len(output_sizes) if output_sizes else 0
    avg_size_change = avg_output / avg_input if avg_input > 0 else 1.0
    
    return {
        "avg_input_size": avg_input,
        "avg_output_size": avg_output,
        "avg_size_change": avg_size_change,
        "num_unique_colors": len(all_color_counts),
    }
