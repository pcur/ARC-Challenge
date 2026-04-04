"""
Utility module for ARC Challenge tasks.

Provides functionality for:
- Loading and parsing ARC challenge data
- Test-time finetuning
- Running evaluations on test sets
- Model inference and prediction
"""

from .data_utils import (
    load_arc_task,
    load_arc_tasks_batch,
    parse_arc_json,
    ARCTaskData,
)

from .evaluation import (
    evaluate_model_on_task,
    evaluate_model_on_dataset,
    EvaluationMetrics,
)

from .finetuning import (
    finetune_model_on_task,
    test_time_finetune,
    TestTimeFinetuneConfig,
)

from .inference import (
    predict_on_grid,
    predict_on_task,
    batch_predict,
)

from .plotting import (
    # CSV utilities
    save_finetuning_results_to_csv,
    save_evaluation_results_to_csv,
    load_finetuning_results_from_csv,
    load_evaluation_results_from_csv,
    # Plotting
    set_plot_style,
    plot_finetuning_loss,
    plot_accuracy_distribution,
    plot_task_performance,
    plot_training_and_validation,
    create_comparison_plot,
    generate_paper_figures,
)

__all__ = [
    # Data utilities
    "load_arc_task",
    "load_arc_tasks_batch",
    "parse_arc_json",
    "ARCTaskData",
    # Evaluation
    "evaluate_model_on_task",
    "evaluate_model_on_dataset",
    "EvaluationMetrics",
    # Finetuning
    "finetune_model_on_task",
    "test_time_finetune",
    "TestTimeFinetuneConfig",
    # Inference
    "predict_on_grid",
    "predict_on_task",
    "batch_predict",
    # Plotting
    "save_finetuning_results_to_csv",
    "save_evaluation_results_to_csv",
    "load_finetuning_results_from_csv",
    "load_evaluation_results_from_csv",
    "set_plot_style",
    "plot_finetuning_loss",
    "plot_accuracy_distribution",
    "plot_task_performance",
    "plot_training_and_validation",
    "create_comparison_plot",
    "generate_paper_figures",
]
