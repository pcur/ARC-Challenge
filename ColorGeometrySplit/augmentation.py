"""
augmentation.py
===============
On-the-fly data augmentation for ARC grids.

All augmentations operate on raw ARC grids (list-of-lists of ints) before
graph construction. This guarantees that both the object graph and pixel
graph derived from the same grid see the same transformation — consistency
is critical for the dual encoder.

Augmentations
-------------
Geometric (8 orientations — 4 rotations × 2 flips):
  rot0, rot90, rot180, rot270
  rot0_flip, rot90_flip, rot180_flip, rot270_flip

Color permutation:
  Randomly reassign non-background color indices. Background (0) is always
  kept as 0. The permutation maps each color that appears in the grid to a
  different randomly chosen non-background color, with no repeats.

  Example: if a grid uses colors {0, 3, 5}, one permutation might map
  3→7 and 5→2, producing a grid with colors {0, 7, 2}. The pattern of
  which cells share colors is preserved exactly.

Usage
-----
    from augmentation import augment_grid

    # single augmentation with both geometric and color
    aug_grid = augment_grid(grid, geometric=True, color_perm=True)

    # reproducible augmentation with a specific seed
    aug_grid = augment_grid(grid, geometric=True, color_perm=True, seed=42)
"""

import random
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC AUGMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(grid):
    return np.array(grid, dtype=np.int32)

def _to_list(arr):
    return arr.tolist()

def rot90_grid(grid):
    """Rotate 90 degrees clockwise."""
    return _to_list(np.rot90(_to_np(grid), k=-1))

def rot180_grid(grid):
    """Rotate 180 degrees."""
    return _to_list(np.rot90(_to_np(grid), k=2))

def rot270_grid(grid):
    """Rotate 270 degrees clockwise."""
    return _to_list(np.rot90(_to_np(grid), k=1))

def fliph_grid(grid):
    """Flip horizontally (left-right mirror)."""
    return _to_list(np.fliplr(_to_np(grid)))

def flipv_grid(grid):
    """Flip vertically (top-bottom mirror)."""
    return _to_list(np.flipud(_to_np(grid)))


def fliph_grid(grid):
    """Flip horizontally (left-right mirror)."""
    return [list(reversed(row)) for row in grid]


def flipv_grid(grid):
    """Flip vertically (top-bottom mirror)."""
    return list(reversed(grid))


# All 8 distinct orientations (D4 symmetry group)
_GEO_TRANSFORMS = [
    lambda g: g,                          # identity
    rot90_grid,                           # 90 CW
    rot180_grid,                          # 180
    rot270_grid,                          # 270 CW
    fliph_grid,                           # horizontal flip
    lambda g: rot90_grid(fliph_grid(g)),  # 90 + flip
    lambda g: rot180_grid(fliph_grid(g)), # 180 + flip
    lambda g: rot270_grid(fliph_grid(g)), # 270 + flip
]


def apply_geometric(grid, transform_idx: int):
    """Apply one of the 8 geometric transforms by index."""
    return _GEO_TRANSFORMS[transform_idx](grid)


# ─────────────────────────────────────────────────────────────────────────────
# COLOR PERMUTATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_color_permutation(grid, perm: dict):
    """
    Apply a color permutation mapping to a grid.

    perm : dict mapping old_color → new_color
           Background (0) must map to 0 (or be absent from perm).
    """
    return [[perm.get(cell, cell) for cell in row] for row in grid]


def random_color_permutation(grid, rng=None):
    """
    Generate and apply a random color permutation.

    Only non-background colors that appear in the grid are permuted.
    Background (0) always stays 0.
    The permutation is a bijection over the colors that appear, mapped
    to randomly chosen colors from the full set {1..9}.

    Returns
    -------
    aug_grid : augmented grid
    perm     : the permutation dict applied (for reproducibility)
    """
    if rng is None:
        rng = random.Random()

    # find which non-background colors appear
    present = sorted({cell for row in grid for cell in row if cell != 0})
    if not present:
        return grid, {}

    # pick a random target set of the same size from {1..9}
    all_colors = list(range(1, 10))
    targets    = rng.sample(all_colors, len(present))
    perm       = {src: tgt for src, tgt in zip(present, targets)}

    return apply_color_permutation(grid, perm), perm


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_grid(
    grid,
    geometric: bool  = True,
    color_perm: bool = True,
    seed: int        = None,
):
    """
    Apply random augmentation to an ARC grid.

    Parameters
    ----------
    grid       : list-of-lists of int (ARC grid)
    geometric  : if True, apply a random geometric transform
    color_perm : if True, apply a random color permutation
    seed       : optional int seed for reproducibility

    Returns
    -------
    aug_grid : augmented grid (list-of-lists)
    info     : dict with keys 'geo_idx' and 'color_perm' for debugging
    """
    rng = random.Random(seed)

    # convert to list-of-lists in case it's tuples
    grid = [list(row) for row in grid]

    info = {}

    if geometric:
        geo_idx = rng.randint(0, 7)
        grid    = apply_geometric(grid, geo_idx)
        info["geo_idx"] = geo_idx
    else:
        info["geo_idx"] = 0

    if color_perm:
        grid, perm = random_color_permutation(grid, rng)
        info["color_perm"] = perm
    else:
        info["color_perm"] = {}

    return grid, info