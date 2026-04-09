"""
Example usage of ARC utility functions.

This script demonstrates a complete workflow:
1. Load ARC tasks
2. Run test-time finetuning
3. Evaluate models
4. Save results to CSV
5. Generate plots for paper
"""

import torch
from pathlib import Path

from util import (
    # Data loading
    load_arc_tasks_batch,
    ARCTaskData,
    # Finetuning
    TestTimeFinetuneConfig,
    test_time_finetune,
    # Evaluation
    evaluate_model_on_task,
    # Plotting and CSV
    save_finetuning_results_to_csv,
    save_evaluation_results_to_csv,
    set_plot_style,
    plot_finetuning_loss,
    plot_accuracy_distribution,
    plot_task_performance,
    generate_paper_figures,
)


def example_workflow(
    model,
    arc_data_dir: str = "data/evaluation",
    results_dir: str = "results",
    figures_dir: str = "figures",
):
    """
    Example of a complete workflow: finetuning → evaluation → plotting.
    
    Parameters
    ----------
    model : torch.nn.Module
        Your trained model
    arc_data_dir : str
        Directory with ARC task JSON files
    results_dir : str
        Directory to save CSV results
    figures_dir : str
        Directory to save generated plots
    """
    
    # === SETUP ===
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # === LOAD DATA ===
    print(f"\nLoading tasks from {arc_data_dir}...")
    tasks = load_arc_tasks_batch(arc_data_dir, limit=10)  # Load 10 tasks for example
    print(f"Loaded {len(tasks)} tasks")
    
    # === FINETUNING ===
    print("\nRunning test-time finetuning on each task...")
    
    finetune_config = TestTimeFinetuneConfig(
        learning_rate=1e-4,
        num_epochs=20,
        batch_size=4,
        device=device,
        verbose=True,
    )
    
    # Note: You'll need to provide your own loss_fn and batching_fn
    # These depend on your model architecture
    # Example:
    # loss_fn = model.compute_loss  # or YourCustomLoss()
    # batching_fn = prepare_batch_for_model
    
    finetuning_results = {}
    
    for task_id, task_data in list(tasks.items())[:3]:  # Example: finetune on first 3 tasks
        print(f"\nFinetuning on {task_id}...")
        
        # Uncomment when you have loss_fn and batching_fn:
        # result = test_time_finetune(
        #     model,
        #     task_data,
        #     loss_fn=loss_fn,
        #     batching_fn=batching_fn,
        #     config=finetune_config,
        # )
        # finetuning_results[task_id] = result
        
    # === SAVE FINETUNING RESULTS TO CSV ===
    # Example of saving finetuning results:
    # for task_id, result in finetuning_results.items():
    #     csv_path = Path(results_dir) / "finetuning_results.csv"
    #     save_finetuning_results_to_csv(
    #         {
    #             "task_id": result.task_id,
    #             "train_losses": result.train_losses,
    #             "val_losses": result.val_losses,
    #             "initial_loss": result.initial_loss,
    #             "final_loss": result.final_loss,
    #             "best_loss": result.best_loss,
    #             "num_epochs_trained": result.num_epochs_trained,
    #             "stopped_early": result.stopped_early,
    #         },
    #         str(csv_path),
    #     )
    
    # === EVALUATION ===
    print("\nEvaluating on test sets...")
    
    # Note: You'll need to provide predict_fn
    # def predict_fn(model, input_grid):
    #     # Your prediction logic here
    #     return output_grid
    
    evaluation_results = {}
    
    for task_id, task_data in list(tasks.items())[:3]:
        print(f"Evaluating {task_id}...")
        
        # Uncomment when you have predict_fn:
        # metrics = evaluate_model_on_task(
        #     model,
        #     task_data,
        #     predict_fn=predict_fn,
        #     device=device,
        # )
        # evaluation_results[task_id] = metrics
        # 
        # # Save to CSV
        # csv_path = Path(results_dir) / "evaluation_results.csv"
        # save_evaluation_results_to_csv(
        #     task_id,
        #     metrics.accuracy,
        #     metrics.correct_predictions,
        #     metrics.total_predictions,
        #     str(csv_path),
        # )
    
    # === PLOTTING ===
    print("\nGenerating plots...")
    
    # Set publication-quality style
    set_plot_style("paper")
    
    # Example: generate all standard plots from CSV files
    # figures = generate_paper_figures(
    #     results_dir=results_dir,
    #     output_dir=figures_dir,
    # )
    # print(f"Generated figures: {figures}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")


def example_load_and_plot():
    """
    Example of loading pre-existing CSV results and generating plots.
    
    Use this if you've already run finetuning/evaluation and saved results.
    """
    
    from util import (
        load_finetuning_results_from_csv,
        load_evaluation_results_from_csv,
        plot_finetuning_loss,
        plot_accuracy_distribution,
        plot_task_performance,
        plot_training_and_validation,
    )
    
    results_dir = Path("results")
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load finetuning results
    finetune_csv = results_dir / "finetuning_results.csv"
    if finetune_csv.exists():
        finetune_results = load_finetuning_results_from_csv(str(finetune_csv))
        
        # Plot loss curves
        plot_finetuning_loss(
            finetune_results,
            output_path=str(figures_dir / "loss_curves.png"),
        )
        
        # Plot specific task
        if finetune_results:
            task_id = list(finetune_results.keys())[0]
            plot_training_and_validation(
                finetune_results,
                task_id,
                output_path=str(figures_dir / f"training_{task_id}.png"),
            )
    
    # Load evaluation results
    eval_csv = results_dir / "evaluation_results.csv"
    if eval_csv.exists():
        eval_results = load_evaluation_results_from_csv(str(eval_csv))
        
        # Plot accuracy distribution
        plot_accuracy_distribution(
            eval_results,
            output_path=str(figures_dir / "accuracy_distribution.png"),
        )
        
        # Plot performance across tasks
        plot_task_performance(
            eval_results,
            output_path=str(figures_dir / "task_performance.png"),
            top_n=20,
        )
    
    print(f"Plots generated in: {figures_dir}")


if __name__ == "__main__":
    # Example 1: Complete workflow (requires model, loss_fn, predict_fn)
    # example_workflow(model=my_model)
    
    # Example 2: Load existing results and generate plots
    print("To use this example script:")
    print("1. Provide your model, loss_fn, and predict_fn in example_workflow()")
    print("2. Or use example_load_and_plot() to plot existing CSV results")
    print("\nCSV columns and structure:")
    print("- finetuning_results.csv: epoch, train_loss, val_loss, initial_loss, etc.")
    print("- evaluation_results.csv: task_id, accuracy, correct, total")
