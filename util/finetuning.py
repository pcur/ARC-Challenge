"""
Test-time finetuning utilities for ARC models.

Provides functions for finetuning models on task-specific data.
"""

from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
import torch
import torch.optim as optim
from .data_utils import ARCTaskData


@dataclass
class TestTimeFinetuneConfig:
    """Configuration for test-time finetuning."""
    
    # Finetuning hyperparameters
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    
    # Stopping criteria
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: float = 1e-5
    
    # Regularization
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None
    
    # Device
    device: str = "cpu"
    
    # Logging
    verbose: bool = True
    log_interval: int = 5
    
    # Optimization
    optimizer_type: str = "adam"  # "adam", "sgd", "adamw"


@dataclass
class FinetuneResult:
    """Results from finetuning."""
    
    task_id: str
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    num_epochs_trained: int = 0
    stopped_early: bool = False
    
    # History
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = [
            f"Finetuning Results for {self.task_id}:",
            f"  Epochs trained: {self.num_epochs_trained}",
            f"  Early stopping: {self.stopped_early}",
        ]
        if self.initial_loss is not None:
            lines.append(f"  Initial loss: {self.initial_loss:.6f}")
        if self.final_loss is not None:
            lines.append(f"  Final loss: {self.final_loss:.6f}")
        if self.best_loss is not None:
            lines.append(f"  Best loss: {self.best_loss:.6f}")
        if self.initial_loss and self.final_loss:
            improvement = (self.initial_loss - self.final_loss) / self.initial_loss
            lines.append(f"  Loss improvement: {improvement:.2%}")
        return "\n".join(lines)


def finetune_model_on_task(
    model: torch.nn.Module,
    task_data: ARCTaskData,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batching_fn: Callable[[List[Dict], int, str], Any],
    config: TestTimeFinetuneConfig = TestTimeFinetuneConfig(),
) -> FinetuneResult:
    """
    Finetune a model on a specific ARC task using training examples.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to finetune
    task_data : ARCTaskData
        Task data containing training pairs
    loss_fn : callable
        Function to compute loss: (model, batch) -> loss (scalar tensor)
    batching_fn : callable
        Function to create batches: (samples, batch_size, device) -> batch_obj
    config : TestTimeFinetuneConfig
        Finetuning configuration
        
    Returns
    -------
    FinetuneResult
        Results of the finetuning process
        
    Notes
    -----
    This function modifies the model in-place. Consider cloning the model
    before calling if you want to preserve the original weights.
    """
    
    model.to(config.device)
    model.train()
    
    result = FinetuneResult(task_id=task_data.task_id)
    
    # Prepare training data
    train_samples = task_data.train_pairs
    
    if not train_samples:
        if config.verbose:
            print(f"No training samples for {task_data.task_id}, skipping finetuning")
        return result
    
    # Setup optimizer
    optimizer = _build_optimizer(model, config)
    
    # Finetuning loop
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches
        batches = _create_batches(train_samples, config.batch_size)
        
        for batch in batches:
            optimizer.zero_grad()
            
            # Compute loss
            loss = loss_fn(model, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Optimization step
            optimizer.step()
            
            epoch_loss += float(loss.item())
            num_batches += 1
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        result.train_losses.append(avg_epoch_loss)
        
        # Track best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Store initial and final loss
        if epoch == 0:
            result.initial_loss = avg_epoch_loss
        result.final_loss = avg_epoch_loss
        result.best_loss = best_loss
        result.num_epochs_trained += 1
        
        # Logging
        if config.verbose and (epoch + 1) % config.log_interval == 0:
            print(f"  Epoch {epoch + 1}/{config.num_epochs} | "
                  f"Loss: {avg_epoch_loss:.6f} | "
                  f"Best: {best_loss:.6f}")
        
        # Early stopping
        if config.early_stopping_patience is not None:
            if epochs_without_improvement >= config.early_stopping_patience:
                if config.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                result.stopped_early = True
                break
            
            # Check if improvement is too small
            if best_loss > 0:
                relative_improvement = (best_loss - avg_epoch_loss) / best_loss
                if relative_improvement < config.early_stopping_threshold:
                    epochs_without_improvement += 1
    
    return result


def test_time_finetune(
    model: torch.nn.Module,
    task_data: ARCTaskData,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batching_fn: Callable[[List[Dict], int, str], Any],
    num_finetune_examples: Optional[int] = None,
    config: Optional[TestTimeFinetuneConfig] = None,
) -> FinetuneResult:
    """
    Run test-time finetuning on a task using a subset of training examples.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to finetune
    task_data : ARCTaskData
        Task data
    loss_fn : callable
        Loss function: (model, batch) -> loss
    batching_fn : callable
        Batching function: (samples, batch_size, device) -> batch
    num_finetune_examples : int, optional
        Number of training examples to use for finetuning.
        If None, uses all available training examples.
    config : TestTimeFinetuneConfig, optional
        Finetuning config. If None, uses default config.
        
    Returns
    -------
    FinetuneResult
        Results of finetuning
    """
    
    if config is None:
        config = TestTimeFinetuneConfig()
    
    # Select finetuning examples
    finetune_samples = task_data.train_pairs
    if num_finetune_examples is not None:
        finetune_samples = task_data.train_pairs[:num_finetune_examples]
    
    # Create a temporary task data object with selected examples
    finetune_task = ARCTaskData(
        task_id=task_data.task_id,
        train_pairs=finetune_samples,
        test_pairs=[],
        raw_json=task_data.raw_json,
    )
    
    if config.verbose:
        print(f"Running test-time finetuning on {task_data.task_id} "
              f"with {len(finetune_samples)} examples")
    
    result = finetune_model_on_task(
        model,
        finetune_task,
        loss_fn,
        batching_fn,
        config,
    )
    
    return result


def _build_optimizer(
    model: torch.nn.Module,
    config: TestTimeFinetuneConfig,
) -> torch.optim.Optimizer:
    """Build optimizer based on config."""
    
    if config.optimizer_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def _create_batches(samples: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Split samples into batches.
    
    Parameters
    ----------
    samples : list
        Samples to batch
    batch_size : int
        Size of each batch
        
    Returns
    -------
    list
        List of batches (each batch is a list of samples)
    """
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    return batches


def get_finetuned_model_copy(
    model: torch.nn.Module,
    task_data: ARCTaskData,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batching_fn: Callable[[List[Dict], int, str], Any],
    config: TestTimeFinetuneConfig = TestTimeFinetuneConfig(),
) -> tuple[torch.nn.Module, FinetuneResult]:
    """
    Get a finetuned copy of the model without modifying the original.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to finetune (will not be modified)
    task_data : ARCTaskData
        Task data for finetuning
    loss_fn : callable
        Loss function
    batching_fn : callable
        Batching function
    config : TestTimeFinetuneConfig
        Finetuning config
        
    Returns
    -------
    (finetuned_model, result)
        Tuple of (finetuned model copy, finetuning result)
    """
    import copy
    
    # Deep copy the model
    model_copy = copy.deepcopy(model)
    
    # Finetune the copy
    result = finetune_model_on_task(
        model_copy,
        task_data,
        loss_fn,
        batching_fn,
        config,
    )
    
    return model_copy, result
