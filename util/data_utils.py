"""
Data utilities for loading and parsing ARC challenge datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ARCTaskData:
    """Container for an ARC task's train/test data."""
    
    task_id: str
    train_pairs: List[Dict[str, Any]]  # List of {"input": [...], "output": [...]}
    test_pairs: List[Dict[str, Any]]   # List of {"input": [...], "output": [...]} (may be empty for actual test)
    raw_json: Dict[str, Any]           # Original raw JSON
    
    @property
    def num_train_pairs(self) -> int:
        return len(self.train_pairs)
    
    @property
    def num_test_pairs(self) -> int:
        return len(self.test_pairs)


def parse_arc_json(data: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse raw ARC JSON data into train and test pairs.
    
    Parameters
    ----------
    data : dict
        Raw ARC JSON with "train" and "test" keys
        
    Returns
    -------
    train_pairs : list of dict
        List of {"input": grid, "output": grid} for training examples
    test_pairs : list of dict
        List of {"input": grid, "output": grid} for test examples
    """
    train_pairs = data.get("train", [])
    test_pairs = data.get("test", [])
    
    return train_pairs, test_pairs


def load_arc_task(task_path: str) -> ARCTaskData:
    """
    Load a single ARC task from a JSON file.
    
    Parameters
    ----------
    task_path : str
        Path to the ARC task JSON file
        
    Returns
    -------
    ARCTaskData
        Parsed task data with train/test pairs
        
    Raises
    ------
    FileNotFoundError
        If the task file does not exist
    ValueError
        If the JSON format is invalid
    """
    task_path = Path(task_path)
    
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    
    with open(task_path, 'r') as f:
        raw_json = json.load(f)
    
    train_pairs, test_pairs = parse_arc_json(raw_json)
    task_id = task_path.stem  # filename without extension
    
    return ARCTaskData(
        task_id=task_id,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        raw_json=raw_json,
    )


def load_arc_tasks_batch(
    directory: str,
    task_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, ARCTaskData]:
    """
    Load multiple ARC tasks from a directory.
    
    Parameters
    ----------
    directory : str
        Path to directory containing ARC JSON files
    task_ids : list of str, optional
        Specific task IDs to load. If None, loads all files in directory.
    limit : int, optional
        Maximum number of tasks to load
        
    Returns
    -------
    dict
        Dictionary mapping task_id -> ARCTaskData
        
    Raises
    ------
    FileNotFoundError
        If directory does not exist
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Get JSON files to load
    json_files = sorted(dir_path.glob("*.json"))
    
    if task_ids is not None:
        # Filter to specific task_ids
        task_id_set = set(task_ids)
        json_files = [f for f in json_files if f.stem in task_id_set]
    
    if limit is not None:
        json_files = json_files[:limit]
    
    tasks = {}
    for json_file in json_files:
        try:
            task_data = load_arc_task(str(json_file))
            tasks[task_data.task_id] = task_data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue
    
    return tasks


def get_train_test_split(
    task_data: ARCTaskData,
    train_size: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split task data into training and validation examples.
    
    Parameters
    ----------
    task_data : ARCTaskData
        The task data to split
    train_size : int, optional
        Number of examples to use for training. If None, uses all but 1.
        
    Returns
    -------
    train_examples : list of dict
        Examples to use for training/finetuning
    val_examples : list of dict
        Examples to use for validation
    """
    if train_size is None:
        train_size = max(1, len(task_data.train_pairs) - 1)
    
    train_size = min(train_size, len(task_data.train_pairs))
    
    train_examples = task_data.train_pairs[:train_size]
    val_examples = task_data.train_pairs[train_size:]
    
    return train_examples, val_examples


def get_grid_dimensions(grid: List[List[int]]) -> Tuple[int, int]:
    """
    Get height and width of a grid.
    
    Parameters
    ----------
    grid : list of list of int
        2D grid of integers
        
    Returns
    -------
    height : int
        Number of rows
    width : int
        Number of columns (based on first row)
    """
    height = len(grid)
    width = len(grid[0]) if grid else 0
    return height, width


def grid_stats(grid: List[List[int]]) -> Dict[str, Any]:
    """
    Compute basic statistics about a grid.
    
    Parameters
    ----------
    grid : list of list of int
        2D grid of integers
        
    Returns
    -------
    dict
        Dictionary with keys:
        - height: number of rows
        - width: number of columns
        - num_colors: number of unique colors (0-9)
        - color_counts: dict of color -> count
    """
    height, width = get_grid_dimensions(grid)
    
    colors = []
    for row in grid:
        colors.extend(row)
    
    unique_colors = set(colors)
    color_counts = {c: colors.count(c) for c in unique_colors}
    
    return {
        "height": height,
        "width": width,
        "num_colors": len(unique_colors),
        "color_counts": color_counts,
        "total_cells": height * width,
    }


def filter_tasks_by_size(
    tasks: Dict[str, ARCTaskData],
    min_height: int = 1,
    min_width: int = 1,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Dict[str, ARCTaskData]:
    """
    Filter tasks based on grid dimensions.
    
    Parameters
    ----------
    tasks : dict
        Dictionary of task_id -> ARCTaskData
    min_height, min_width : int
        Minimum grid dimensions
    max_height, max_width : int, optional
        Maximum grid dimensions
        
    Returns
    -------
    dict
        Filtered dictionary of tasks
    """
    filtered = {}
    
    for task_id, task_data in tasks.items():
        # Check train pairs
        valid = True
        for pair in task_data.train_pairs:
            for grid in [pair.get("input"), pair.get("output")]:
                if grid:
                    h, w = get_grid_dimensions(grid)
                    if h < min_height or w < min_width:
                        valid = False
                        break
                    if max_height and h > max_height:
                        valid = False
                        break
                    if max_width and w > max_width:
                        valid = False
                        break
            if not valid:
                break
        
        # Check test pairs
        if valid:
            for pair in task_data.test_pairs:
                for grid in [pair.get("input"), pair.get("output")]:
                    if grid:
                        h, w = get_grid_dimensions(grid)
                        if h < min_height or w < min_width:
                            valid = False
                            break
                        if max_height and h > max_height:
                            valid = False
                            break
                        if max_width and w > max_width:
                            valid = False
                            break
                if not valid:
                    break
        
        if valid:
            filtered[task_id] = task_data
    
    return filtered
