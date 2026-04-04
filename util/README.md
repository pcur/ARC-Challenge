"""
ARC Challenge Utility Package
=============================

A comprehensive set of utility functions for:
- Loading and parsing ARC task data
- Test-time finetuning with PyTorch
- Evaluating model performance
- Running inference/predictions
- Saving/loading results to CSV
- Generating publication-quality plots

Directory Structure
-------------------
util/
├── __init__.py              # Package exports
├── data_utils.py            # Data loading and parsing
├── evaluation.py            # Evaluation metrics and functions
├── finetuning.py            # Test-time finetuning
├── inference.py             # Prediction/inference utilities
├── plotting.py              # CSV I/O and plotting
├── example_usage.py         # Examples demonstrating the API
└── README.md                # This file

Quick Start
-----------

1. LOADING DATA
   from util import load_arc_tasks_batch
   
   tasks = load_arc_tasks_batch("data/evaluation")
   # Returns: Dict[task_id -> ARCTaskData]
   
   # ARCTaskData contains:
   # - task_id: str
   # - train_pairs: [{"input": grid, "output": grid}, ...]
   # - test_pairs: [{"input": grid, "output": grid}, ...]

2. FINETUNING ON A TASK
   from util import test_time_finetune, TestTimeFinetuneConfig
   
   config = TestTimeFinetuneConfig(
       learning_rate=1e-4,
       num_epochs=10,
       device="cuda",
   )
   
   result = test_time_finetune(
       model=my_model,
       task_data=tasks["task_id"],
       loss_fn=my_loss_function,        # (model, batch) -> loss
       batching_fn=my_batching_function, # (samples, batch_size, device) -> batch
       config=config,
   )
   # Returns: FinetuneResult with train_losses, val_losses, etc.

3. EVALUATION
   from util import evaluate_model_on_task
   
   metrics = evaluate_model_on_task(
       model=my_model,
       task_data=tasks["task_id"],
       predict_fn=my_predict_function,  # (model, input_grid) -> output_grid
       device="cuda",
   )
   # Returns: EvaluationMetrics with accuracy, correct predictions, etc.

4. SAVE & PLOT RESULTS
   from util import (
       save_finetuning_results_to_csv,
       save_evaluation_results_to_csv,
       generate_paper_figures,
   )
   
   # Save finetuning results to CSV
   save_finetuning_results_to_csv(
       {
           "task_id": result.task_id,
           "train_losses": result.train_losses,
           "val_losses": result.val_losses,
           "initial_loss": result.initial_loss,
           "final_loss": result.final_loss,
           "best_loss": result.best_loss,
           "num_epochs_trained": result.num_epochs_trained,
           "stopped_early": result.stopped_early,
       },
       "results/finetuning_results.csv",
   )
   
   # Save evaluation results to CSV
   save_evaluation_results_to_csv(
       task_id="task_id",
       accuracy=0.95,
       correct=19,
       total=20,
       output_path="results/evaluation_results.csv",
   )
   
   # Generate all plots from CSVs
   figures = generate_paper_figures(
       results_dir="results",
       output_dir="figures",
   )


Detailed Module Documentation
------------------------------

data_utils.py
~~~~~~~~~~~~~
Data loading, parsing, and utilities.

Functions:
- load_arc_task(task_path)               Load single task JSON
- load_arc_tasks_batch(directory, task_ids, limit)  Load multiple tasks
- parse_arc_json(data)                   Parse raw JSON
- get_train_test_split(task_data)        Split training examples
- get_grid_dimensions(grid)              Get height/width
- grid_stats(grid)                       Compute color/size statistics
- filter_tasks_by_size(tasks, min/max height/width)  Filter tasks

Classes:
- ARCTaskData                            Dataclass holding task data


evaluation.py
~~~~~~~~~~~~~
Model evaluation metrics and functions.

Functions:
- evaluate_model_on_task(model, task_data, predict_fn, device)
  Evaluate model on a single task's test set
  
- evaluate_model_on_dataset(model, tasks, predict_fn, device)
  Evaluate on multiple tasks
  
- compute_dataset_statistics(results)
  Aggregate metrics across tasks
  
- print_evaluation_summary(results, sort_by="accuracy")
  Print formatted results table
  
- compute_task_difficulty(task_data)
  Analyze task via statistics (not model-dependent)
  
- grids_equal(grid1, grid2)
  Check if two grids are identical

Classes:
- EvaluationMetrics                      Results dataclass with accuracy, loss


finetuning.py
~~~~~~~~~~~~~~
Test-time finetuning on specific tasks.

Key Functions:
- finetune_model_on_task(model, task_data, loss_fn, batching_fn, config)
  Finetune on single task (modifies model in-place)
  
- test_time_finetune(model, task_data, loss_fn, batching_fn, 
                     num_finetune_examples, config)
  Wrapper with example selection
  
- get_finetuned_model_copy(model, task_data, loss_fn, batching_fn, config)
  Returns finetuned copy without modifying original

Classes:
- TestTimeFinetuneConfig                Configuration with hyperparameters
- FinetuneResult                        Results with loss history

Config Options:
- learning_rate        (default: 1e-4)
- num_epochs           (default: 10)
- batch_size           (default: 4)
- early_stopping_patience  (default: 5)
- optimizer_type       (default: "adam") - can be "adam", "adamw", "sgd"
- weight_decay         (default: 0.0)
- gradient_clip        (default: None)


inference.py
~~~~~~~~~~~~
Prediction and inference utilities.

Functions:
- predict_on_grid(model, input_grid, predict_fn, device)
  Single grid prediction
  
- predict_on_task(model, task_data, predict_fn, device, verbose)
  Predict on all test inputs in task
  
- batch_predict(model, input_grids, predict_fn, device, batch_size)
  Batch prediction on multiple grids
  
- predict_with_ensemble(models, input_grid, predict_fn, 
                       ensemble_strategy, device)
  Ensemble prediction (majority_vote or average)
  
- create_predict_fn_wrapper(preprocess_fn, postprocess_fn)
  Create custom predict_fn with preprocessing/postprocessing

Internal:
- _grid_to_tensor(grid)                 Convert grid to torch tensor
- _tensor_to_grid(tensor)               Convert tensor back to grid


plotting.py
~~~~~~~~~~~
CSV utilities and plotting for papers.

CSV Functions:
- save_finetuning_results_to_csv(results_dict, output_path)
  Save finetuning results from FinetuneResult
  
- save_evaluation_results_to_csv(task_id, accuracy, correct, total, 
                                   output_path, additional_metrics)
  Append evaluation results

- load_finetuning_results_from_csv(csv_path)
  Load finetuning results back as dict
  
- load_evaluation_results_from_csv(csv_path)
  Load evaluation results back as dict)

Plotting Functions:
- set_plot_style(style)                 Set matplotlib style ("paper", "presentation", "minimal")
- plot_finetuning_loss(results, output_path, task_ids, title, figsize)
  Line plot of training loss over epochs
  
- plot_accuracy_distribution(results, output_path, title, figsize)
  Histogram of accuracies across tasks
  
- plot_task_performance(results, output_path, top_n, figsize)
  Horizontal bar chart of per-task accuracies
  
- plot_training_and_validation(results, task_id, output_path, figsize)
  Training vs validation curves for single task
  
- create_comparison_plot(results_list, labels, output_path, figsize)
  Compare accuracies across model runs
  
- generate_paper_figures(results_dir, output_dir, csv_files)
  Generate all standard plots at once from CSVs

Returns: Dict[figure_name -> saved_path]


CSV Format
----------

finetuning_results.csv:
- task_id              Task identifier
- epoch                Epoch number (1-indexed)
- train_loss           Training loss for epoch
- val_loss             Validation loss (optional)
- initial_loss         Loss at epoch 1
- final_loss           Final loss value
- best_loss            Best loss achieved
- num_epochs_trained   Total epochs trained
- stopped_early        Boolean for early stopping


evaluation_results.csv:
- task_id              Task identifier
- accuracy             Accuracy (0-1)
- correct              Number correct
- total                Total predictions
- [additional columns] User-defined metrics


Usage Patterns
--------------

PATTERN 1: Quick evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from util import load_arc_task, evaluate_model_on_task

task = load_arc_task("data/evaluation/task_123.json")
metrics = evaluate_model_on_task(model, task, predict_fn, device="cuda")
print(f"Accuracy: {metrics.accuracy:.2%}")


PATTERN 2: Finetuning + Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from util import test_time_finetune, evaluate_model_on_task

config = TestTimeFinetuneConfig(num_epochs=20, learning_rate=1e-4)
finetune_result = test_time_finetune(model, task, loss_fn, batch_fn, config)
eval_metrics = evaluate_model_on_task(model, task, predict_fn)
print(f"Loss improved from {finetune_result.initial_loss:.4f} "
      f"to {finetune_result.final_loss:.4f}")


PATTERN 3: Batch processing + CSV + Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tasks = load_arc_tasks_batch("data/evaluation")

for task_id, task in tasks.items():
    result = test_time_finetune(model, task, loss_fn, batch_fn)
    save_finetuning_results_to_csv(vars(result), "results/finetune.csv")

figures = generate_paper_figures("results", "figures")
print(f"Generated {len(figures)} figures in figures/")


PATTERN 4: Ensemble predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
models = [model1, model2, model3]
prediction = predict_with_ensemble(
    models, 
    input_grid, 
    predict_fn=predict_fn,
    ensemble_strategy="majority_vote"
)


Important Notes
---------------

1. PREDICT FUNCTION
   You must provide a predict_fn with signature:
   
   def predict_fn(model: torch.nn.Module, input_grid: List[List[int]]) -> List[List[int]]:
       # Your preprocessing, forward pass, postprocessing here
       return output_grid
   
   Use create_predict_fn_wrapper() to build this from components.

2. LOSS FUNCTION
   loss_fn must have signature:
   
   def loss_fn(model: torch.nn.Module, batch: Any) -> torch.Tensor:
       # Compute loss given batch
       return scalar_loss

3. BATCHING FUNCTION
   batching_fn must have signature:
   
   def batching_fn(samples: List[Dict], batch_size: int, device: str) -> Any:
       # Convert list of sample dicts to model-ready batch
       return batch_obj

4. DEVICE MANAGEMENT
   Functions handle device management, but ensure your losses/batches
   are on the correct device after calling batching_fn.

5. MATPLOTLIB
   Plotting functions require matplotlib. Install with:
   pip install matplotlib
   
   Optional: install seaborn for better styling:
   pip install seaborn

6. CSV I/O
   Results CSVs are simple and can be edited manually or processed
   with pandas:
   
   import pandas as pd
   df = pd.read_csv("results/evaluation_results.csv")
   print(df.groupby("task_id")["accuracy"].mean())


Examples
--------

See util/example_usage.py for complete, runnable examples.
"""
