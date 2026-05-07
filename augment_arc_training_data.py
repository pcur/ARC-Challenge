"""
augment_arc_training_data.py

Creates augmented ARC-AGI-style JSON files from the official data/training folder.

Expected folder layout:
    your_project/
    ├── arc_coord_color_trm.py
    ├── augment_arc_training_data.py
    └── data/
        ├── training/*.json
        └── evaluation/*.json

What this does:
    - Reads every original JSON task in data/training/
    - Applies consistent augmentations to every input/output grid in the task
    - Writes new JSON files back into data/training/ by default
    - Keeps the same ARC task format:
        {
            "train": [{"input": ..., "output": ...}, ...],
            "test":  [{"input": ..., "output": ...}, ...]
        }

Important:
    - The same spatial transform is applied to inputs and outputs.
    - The same color permutation is applied to inputs and outputs.
    - Color index convention is preserved: ARC colors are integers 0-9.
    - Existing files are not overwritten unless --overwrite is used.

Examples:
    python augment_arc_training_data.py

    python augment_arc_training_data.py --num_color_perms 3

    python augment_arc_training_data.py --output_dir data/training_augmented

    python augment_arc_training_data.py --include_rotations --include_flips --num_color_perms 5
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Grid = List[List[int]]
Task = Dict[str, List[Dict[str, Grid]]]


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> Task:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Task, path: Path, overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}\n"
            f"Use --overwrite if you really want to replace it."
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def validate_grid(grid: Grid, where: str):
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError(f"{where}: grid must be a non-empty list")

    width = len(grid[0])

    if width == 0:
        raise ValueError(f"{where}: grid width cannot be zero")

    if len(grid) > 30 or width > 30:
        raise ValueError(f"{where}: grid shape {len(grid)}x{width} exceeds ARC max 30x30")

    for r, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(f"{where}: ragged grid at row {r}")

        for c, value in enumerate(row):
            if not isinstance(value, int) or not (0 <= value <= 9):
                raise ValueError(f"{where}: invalid color {value} at ({r},{c}); expected integer 0-9")


def validate_task(task: Task, task_name: str):
    if "train" not in task:
        raise ValueError(f"{task_name}: missing 'train' key")

    for split_name in ["train", "test"]:
        if split_name not in task:
            continue

        for i, pair in enumerate(task[split_name]):
            if "input" in pair:
                validate_grid(pair["input"], f"{task_name}.{split_name}[{i}].input")
            if "output" in pair:
                validate_grid(pair["output"], f"{task_name}.{split_name}[{i}].output")


# ============================================================
# Grid transforms
# ============================================================

def rotate_grid(grid: Grid, k: int) -> Grid:
    """
    Rotates a grid by k * 90 degrees counterclockwise.
    k can be 0, 1, 2, or 3.
    """
    k = k % 4

    if k == 0:
        return [row[:] for row in grid]

    out = [row[:] for row in grid]

    for _ in range(k):
        # Counterclockwise rotation:
        # transpose then reverse row order.
        out = [list(row) for row in zip(*out)][::-1]

    return out


def flip_horizontal(grid: Grid) -> Grid:
    """
    Mirrors left-right.
    """
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    """
    Mirrors top-bottom.
    """
    return [row[:] for row in reversed(grid)]


def transpose_grid(grid: Grid) -> Grid:
    """
    Swaps rows and columns.
    This is useful for ARC but optional because it changes orientation differently than rotation/flip.
    """
    return [list(row) for row in zip(*grid)]


def apply_spatial_transform(grid: Grid, transform_name: str) -> Grid:
    if transform_name == "identity":
        return [row[:] for row in grid]
    if transform_name == "rot90":
        return rotate_grid(grid, 1)
    if transform_name == "rot180":
        return rotate_grid(grid, 2)
    if transform_name == "rot270":
        return rotate_grid(grid, 3)
    if transform_name == "flip_h":
        return flip_horizontal(grid)
    if transform_name == "flip_v":
        return flip_vertical(grid)
    if transform_name == "transpose":
        return transpose_grid(grid)

    raise ValueError(f"Unknown spatial transform: {transform_name}")


def apply_color_permutation(grid: Grid, perm: Sequence[int]) -> Grid:
    """
    Applies a color remapping.

    perm[old_color] = new_color

    Example:
        perm = [0,2,1,3,4,5,6,7,8,9]
        color 1 becomes 2
        color 2 becomes 1
    """
    if len(perm) != 10:
        raise ValueError("Color permutation must have length 10")

    if sorted(perm) != list(range(10)):
        raise ValueError(f"Invalid color permutation: {perm}")

    return [[perm[value] for value in row] for row in grid]


# ============================================================
# Task-level transforms
# ============================================================

def transform_pair(pair: Dict[str, Grid], spatial: str, color_perm: Sequence[int]) -> Dict[str, Grid]:
    new_pair = {}

    if "input" in pair:
        x = apply_spatial_transform(pair["input"], spatial)
        x = apply_color_permutation(x, color_perm)
        new_pair["input"] = x

    if "output" in pair:
        y = apply_spatial_transform(pair["output"], spatial)
        y = apply_color_permutation(y, color_perm)
        new_pair["output"] = y

    # Preserve any extra keys, though official ARC usually only has input/output.
    for k, v in pair.items():
        if k not in new_pair:
            new_pair[k] = v

    return new_pair


def transform_task(task: Task, spatial: str, color_perm: Sequence[int]) -> Task:
    new_task = {}

    for split_name, pairs in task.items():
        if not isinstance(pairs, list):
            new_task[split_name] = pairs
            continue

        new_task[split_name] = [
            transform_pair(pair, spatial=spatial, color_perm=color_perm)
            for pair in pairs
        ]

    return new_task


# ============================================================
# Color permutation creation
# ============================================================

def identity_perm() -> List[int]:
    return list(range(10))


def random_color_perm(rng: random.Random, keep_zero_fixed: bool = False) -> List[int]:
    """
    Creates a random permutation over ARC colors 0-9.

    keep_zero_fixed:
        If True, color 0 remains 0.
        This is sometimes useful because color 0 is often background in ARC.
        If False, all colors can move freely.
    """
    if keep_zero_fixed:
        rest = list(range(1, 10))
        rng.shuffle(rest)
        return [0] + rest

    perm = list(range(10))
    rng.shuffle(perm)
    return perm


def perm_to_suffix(perm: Sequence[int]) -> str:
    return "p" + "".join(str(x) for x in perm)


# ============================================================
# Main augmentation routine
# ============================================================

def build_transform_list(args) -> List[str]:
    transforms = ["identity"]

    if args.include_rotations:
        transforms.extend(["rot90", "rot180", "rot270"])

    if args.include_flips:
        transforms.extend(["flip_h", "flip_v"])

    if args.include_transpose:
        transforms.append("transpose")

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for t in transforms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def should_skip_file(path: Path) -> bool:
    """
    Prevents repeatedly augmenting already-augmented files.

    Generated files contain '__aug_' in the stem.
    """
    return "__aug_" in path.stem


def augment_training_data(args):
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(input_dir.glob("*.json"))

    if args.skip_existing_augmented:
        json_paths = [p for p in json_paths if not should_skip_file(p)]

    if len(json_paths) == 0:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    transforms = build_transform_list(args)

    color_perms = []

    if args.include_identity_color:
        color_perms.append(identity_perm())

    for _ in range(args.num_color_perms):
        color_perms.append(random_color_perm(rng, keep_zero_fixed=args.keep_zero_fixed))

    # If no color perms requested, still use identity so spatial transforms work.
    if len(color_perms) == 0:
        color_perms.append(identity_perm())

    print("=" * 80)
    print("ARC training data augmentation")
    print("=" * 80)
    print(f"Input dir:  {input_dir.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Original files found: {len(json_paths)}")
    print(f"Spatial transforms: {transforms}")
    print(f"Color permutations: {len(color_perms)}")
    print(f"Keep color 0 fixed: {args.keep_zero_fixed}")
    print(f"Overwrite: {args.overwrite}")
    print()

    written = 0
    skipped = 0

    for task_path in json_paths:
        task = load_json(task_path)
        validate_task(task, task_path.name)

        # Optionally copy original file into output dir if output differs from input.
        if args.copy_originals and input_dir.resolve() != output_dir.resolve():
            dest = output_dir / task_path.name
            if not dest.exists() or args.overwrite:
                shutil.copy2(task_path, dest)

        for spatial in transforms:
            for perm_idx, perm in enumerate(color_perms):
                # Avoid creating a duplicate exact copy of the original task unless requested.
                if (
                    spatial == "identity"
                    and perm == identity_perm()
                    and not args.write_identity_aug
                ):
                    continue

                suffix = f"__aug_{spatial}_{perm_idx:03d}_{perm_to_suffix(perm)}"
                out_name = f"{task_path.stem}{suffix}.json"
                out_path = output_dir / out_name

                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                aug_task = transform_task(task, spatial=spatial, color_perm=perm)
                validate_task(aug_task, out_name)
                save_json(aug_task, out_path, overwrite=args.overwrite)
                written += 1

        if written % 100 == 0 and written > 0:
            print(f"Written {written} augmented files so far...")

    print()
    print("Done.")
    print(f"Augmented files written: {written}")
    print(f"Skipped existing files: {skipped}")
    print()
    print("Reminder:")
    print("  Your training script automatically loads all *.json files in data/training/.")
    print("  If you wrote augmentations into data/training/, the next training run will include them.")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/training",
        help="Folder containing original ARC training JSON files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/training",
        help="Folder to write augmented JSON files. Default writes back into data/training/.",
    )

    parser.add_argument(
        "--include_rotations",
        action="store_true",
        help="Add rot90, rot180, and rot270 augmentations.",
    )

    parser.add_argument(
        "--include_flips",
        action="store_true",
        help="Add horizontal and vertical flip augmentations.",
    )

    parser.add_argument(
        "--include_transpose",
        action="store_true",
        help="Add transpose augmentation.",
    )

    parser.add_argument(
        "--num_color_perms",
        type=int,
        default=3,
        help="Number of random color permutations to create per spatial transform.",
    )

    parser.add_argument(
        "--include_identity_color",
        action="store_true",
        help="Also include identity color mapping. Useful when using spatial transforms only.",
    )

    parser.add_argument(
        "--write_identity_aug",
        action="store_true",
        help="Write identity spatial + identity color as an augmented duplicate. Usually not needed.",
    )

    parser.add_argument(
        "--keep_zero_fixed",
        action="store_true",
        help="Keep ARC color 0 mapped to 0 during random color permutations.",
    )

    parser.add_argument(
        "--copy_originals",
        action="store_true",
        help="If output_dir differs from input_dir, copy original files too.",
    )

    parser.add_argument(
        "--skip_existing_augmented",
        action="store_true",
        default=True,
        help="Skip files whose names already contain '__aug_'. Default: enabled.",
    )

    parser.add_argument(
        "--process_existing_augmented",
        action="store_false",
        dest="skip_existing_augmented",
        help="Allow augmenting already-augmented files. Usually a bad idea.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for color permutations.",
    )

    return parser


def main():
    args = build_argparser().parse_args()

    if args.num_color_perms < 0:
        raise ValueError("--num_color_perms must be >= 0")

    augment_training_data(args)


if __name__ == "__main__":
    main()
