"""
Plotting and visualization utilities for ARC models.

Provides functions for:
- Saving training/evaluation results to CSV
- Loading results from CSV
- Generating publication-quality plots
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import plotting libraries - graceful fallback if not installed
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


# ─────────────────────────────────────────────────────────────────────────────
# CSV UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_finetuning_results_to_csv(
    results_dict: Dict[str, Any],
    output_path: str,
    create_dirs: bool = True,
) -> str:
    """
    Save finetuning results to a CSV file.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with structure:
        {
            "task_id": str,
            "train_losses": [float, ...],
            "val_losses": [float, ...],
            "initial_loss": float,
            "final_loss": float,
            "best_loss": float,
            "num_epochs_trained": int,
            "stopped_early": bool,
        }
    output_path : str
        Path to save CSV file
    create_dirs : bool
        Whether to create parent directories if they don't exist
        
    Returns
    -------
    str
        Path to saved CSV file
    """
    
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'task_id',
            'epoch',
            'train_loss',
            'val_loss',
            'initial_loss',
            'final_loss',
            'best_loss',
            'num_epochs_trained',
            'stopped_early',
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Get list lengths for iteration
        num_epochs = len(results_dict.get('train_losses', []))
        val_losses = results_dict.get('val_losses', [])
        
        for epoch in range(num_epochs):
            row = {
                'task_id': results_dict.get('task_id', ''),
                'epoch': epoch + 1,
                'train_loss': results_dict['train_losses'][epoch],
                'val_loss': val_losses[epoch] if epoch < len(val_losses) else '',
                'initial_loss': results_dict.get('initial_loss', ''),
                'final_loss': results_dict.get('final_loss', ''),
                'best_loss': results_dict.get('best_loss', ''),
                'num_epochs_trained': results_dict.get('num_epochs_trained', ''),
                'stopped_early': results_dict.get('stopped_early', ''),
            }
            writer.writerow(row)
    
    return str(output_path)


def save_evaluation_results_to_csv(
    task_id: str,
    accuracy: float,
    correct: int,
    total: int,
    output_path: str,
    create_dirs: bool = True,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save evaluation results to a CSV file.
    
    Parameters
    ----------
    task_id : str
        Task identifier
    accuracy : float
        Accuracy (0-1)
    correct : int
        Number of correct predictions
    total : int
        Total predictions
    output_path : str
        Path to CSV file
    create_dirs : bool
        Whether to create parent directories
    additional_metrics : dict, optional
        Additional metrics to save as columns
        
    Returns
    -------
    str
        Path to saved CSV
    """
    
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = output_path.exists()
    
    fieldnames = ['task_id', 'accuracy', 'correct', 'total']
    if additional_metrics:
        fieldnames.extend(additional_metrics.keys())
    
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {
            'task_id': task_id,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        
        if additional_metrics:
            row.update(additional_metrics)
        
        writer.writerow(row)
    
    return str(output_path)


def load_finetuning_results_from_csv(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load finetuning results from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file saved by save_finetuning_results_to_csv
        
    Returns
    -------
    dict
        Dictionary with task_id -> list of epoch records
    """
    
    results = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            task_id = row['task_id']
            
            if task_id not in results:
                results[task_id] = []
            
            # Parse numeric fields
            epoch_data = {
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
            }
            
            if row.get('val_loss'):
                epoch_data['val_loss'] = float(row['val_loss'])
            
            if row.get('initial_loss'):
                epoch_data['initial_loss'] = float(row['initial_loss'])
            if row.get('final_loss'):
                epoch_data['final_loss'] = float(row['final_loss'])
            if row.get('best_loss'):
                epoch_data['best_loss'] = float(row['best_loss'])
            
            results[task_id].append(epoch_data)
    
    return results


def load_evaluation_results_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation results from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with evaluation results
        
    Returns
    -------
    dict
        Dictionary with task_id -> metrics dict
    """
    
    results = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            task_id = row['task_id']
            
            metrics = {
                'accuracy': float(row['accuracy']),
                'correct': int(row['correct']),
                'total': int(row['total']),
            }
            
            # Include any additional columns
            for key, value in row.items():
                if key not in ['task_id', 'accuracy', 'correct', 'total']:
                    try:
                        metrics[key] = float(value)
                    except (ValueError, TypeError):
                        metrics[key] = value
            
            results[task_id] = metrics
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def check_plotting_available() -> bool:
    """Check if plotting libraries are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )
    return True


def set_plot_style(style: str = "paper") -> None:
    """
    Set plotting style for better-looking figures.
    
    Parameters
    ----------
    style : str
        Style to use: "paper", "presentation", or "minimal"
    """
    
    if not HAS_MATPLOTLIB:
        return
    
    if style == "paper":
        plt.style.use('seaborn-v0_8-darkgrid' if HAS_SEABORN else 'default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'figure.dpi': 100,
        })
    
    elif style == "presentation":
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 3,
            'lines.markersize': 8,
        })
    
    elif style == "minimal":
        plt.rcParams.update({
            'figure.figsize': (8, 5),
            'font.size': 10,
            'axes.grid': False,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })


def plot_finetuning_loss(
    results: Dict[str, List[Dict]],
    output_path: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
    title: str = "Finetuning Loss Curves",
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Figure]:
    """
    Plot finetuning loss curves for one or more tasks.
    
    Parameters
    ----------
    results : dict
        Dictionary from load_finetuning_results_from_csv
    output_path : str, optional
        Path to save figure. If None, returns figure object.
    task_ids : list, optional
        Specific tasks to plot. If None, plots all.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    Figure or None
        Matplotlib figure object if output_path is None
    """
    
    check_plotting_available()
    
    if task_ids is None:
        task_ids = list(results.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_ids)))
    
    for task_id, color in zip(task_ids, colors):
        if task_id not in results:
            continue
        
        epochs = [r['epoch'] for r in results[task_id]]
        losses = [r['train_loss'] for r in results[task_id]]
        
        ax.plot(epochs, losses, marker='o', label=task_id, color=color, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_accuracy_distribution(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Accuracy Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Figure]:
    """
    Plot distribution of accuracies across tasks.
    
    Parameters
    ----------
    results : dict
        Dictionary from load_evaluation_results_from_csv
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure object if output_path is None
    """
    
    check_plotting_available()
    
    accuracies = [metrics['accuracy'] for metrics in results.values()]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(accuracies, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.2%}')
    ax.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.2%}')
    
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_task_performance(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[Figure]:
    """
    Plot accuracy for each task as a horizontal bar chart.
    
    Parameters
    ----------
    results : dict
        Dictionary from load_evaluation_results_from_csv
    output_path : str, optional
        Path to save figure
    top_n : int, optional
        Only plot top N tasks by accuracy
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure object if output_path is None
    """
    
    check_plotting_available()
    
    # Sort by accuracy
    sorted_tasks = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    if top_n:
        sorted_tasks = sorted_tasks[:top_n]
    
    task_ids = [task for task, _ in sorted_tasks]
    accuracies = [metrics['accuracy'] for _, metrics in sorted_tasks]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.array(accuracies))
    bars = ax.barh(range(len(task_ids)), accuracies, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Task Accuracy{"(Top " + str(top_n) + ")" if top_n else ""}')
    ax.set_xlim([0, 1])
    
    # Add accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.01, i, f'{acc:.1%}', va='center', fontsize=8)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_training_and_validation(
    results: Dict[str, List[Dict]],
    task_id: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Figure]:
    """
    Plot training and validation loss for a single task.
    
    Parameters
    ----------
    results : dict
        Dictionary from load_finetuning_results_from_csv
    task_id : str
        Task to plot
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure object if output_path is None
    """
    
    check_plotting_available()
    
    if task_id not in results:
        raise ValueError(f"Task {task_id} not found in results")
    
    data = results[task_id]
    epochs = [r['epoch'] for r in data]
    train_losses = [r['train_loss'] for r in data]
    val_losses = [r.get('val_loss') for r in data]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=6)
    
    if any(loss is not None for loss in val_losses):
        val_losses = [loss for loss in val_losses if loss is not None]
        val_epochs = epochs[:len(val_losses)]
        ax.plot(val_epochs, val_losses, marker='s', label='Validation Loss', linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Curves: {task_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def create_comparison_plot(
    results_list: List[Dict[str, Dict[str, Any]]],
    labels: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[Figure]:
    """
    Compare accuracies across multiple model runs or configurations.
    
    Parameters
    ----------
    results_list : list of dict
        List of evaluation result dictionaries
    labels : list of str
        Labels for each result (e.g., model names, configs)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure object if output_path is None
    """
    
    check_plotting_available()
    
    if len(results_list) != len(labels):
        raise ValueError("results_list and labels must have same length")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all unique task IDs
    all_tasks = set()
    for results in results_list:
        all_tasks.update(results.keys())
    
    all_tasks = sorted(list(all_tasks))
    
    x = np.arange(len(all_tasks))
    width = 0.8 / len(results_list)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))
    
    for idx, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        accuracies = [results.get(task, {}).get('accuracy', 0) for task in all_tasks]
        ax.bar(x + idx * width, accuracies, width, label=label, color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(results_list) - 1) / 2)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def generate_paper_figures(
    results_dir: str,
    output_dir: str,
    csv_files: Dict[str, str] = None,
) -> Dict[str, str]:
    """
    Generate a set of publication-ready figures from results CSVs.
    
    Parameters
    ----------
    results_dir : str
        Directory containing CSV files
    output_dir : str
        Directory to save generated figures
    csv_files : dict, optional
        Map of plot_name -> csv_filename
        If None, looks for standard filenames
        
    Returns
    -------
    dict
        Dictionary mapping figure_name -> saved_path
    """
    
    check_plotting_available()
    set_plot_style("paper")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Load results
    eval_csv = Path(results_dir) / "evaluation_results.csv"
    finetune_csv = Path(results_dir) / "finetuning_results.csv"
    
    if eval_csv.exists():
        eval_results = load_evaluation_results_from_csv(str(eval_csv))
        
        # Generate evaluation plots
        acc_dist_path = output_dir / "accuracy_distribution.png"
        plot_accuracy_distribution(eval_results, str(acc_dist_path))
        figures['accuracy_distribution'] = str(acc_dist_path)
        
        perf_path = output_dir / "task_performance.png"
        plot_task_performance(eval_results, str(perf_path), top_n=20)
        figures['task_performance'] = str(perf_path)
    
    if finetune_csv.exists():
        finetune_results = load_finetuning_results_from_csv(str(finetune_csv))
        
        # Generate finetuning plots
        loss_path = output_dir / "finetuning_loss.png"
        plot_finetuning_loss(finetune_results, str(loss_path))
        figures['finetuning_loss'] = str(loss_path)
    
    return figures
