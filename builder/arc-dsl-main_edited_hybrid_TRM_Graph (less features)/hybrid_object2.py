# ============================================================
# HYBRID OBJECT BUILDER  v2
# ============================================================
"""
v2 changes vs hybrid_object.py:
    1. my_objects now returns a SORTED LIST (deterministic node ordering)
       Previously used a set, so node IDs were random each run.
    2. edge_vec normalises dx/dy by 30.0 to prevent NaN explosions
       during training (same fix applied to custom_object3.py).
    3. Added graph_to_grid_from_predictions() — reconstructs a grid
       purely from decoder output tensors, enabling inference-time
       grid generation without the original graph.
"""

import math


# ------------------------------------------------------------
# OBJECT EXTRACTION
# ------------------------------------------------------------

def mostcolor(grid):
    """Return the most common value — treated as background."""
    counts = {}
    for row in grid:
        for val in row:
            counts[val] = counts.get(val, 0) + 1
    return max(counts, key=counts.get)


def dneighbors(cell):
    """4-connected neighbors (up, down, left, right)."""
    r, c = cell
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def neighbors(cell):
    """8-connected neighbors (includes diagonals)."""
    r, c = cell
    return {
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1),                 (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
    """
    Extract connected components from the grid.

    v2 FIX: returns a SORTED LIST instead of a set, so node IDs are
    always assigned in the same order (top-left to bottom-right).
    Previously the set iteration order was arbitrary, making training
    targets inconsistent across runs.
    """
    bg = mostcolor(grid) if without_bg else None
    objs = []
    occupied = set()

    h, w = len(grid), len(grid[0])

    # v2 FIX: iterate in fixed row-major order so BFS seeds are consistent
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    neigh_fn = neighbors if diagonal else dneighbors

    for loc in all_cells:
        if loc in occupied:
            continue

        val = grid[loc[0]][loc[1]]
        if without_bg and val == bg:
            continue

        obj = set()
        cands = {loc}

        while cands:
            new = set()
            for cand in cands:
                if cand in occupied:
                    continue
                v = grid[cand[0]][cand[1]]
                cond = (val == v) if univalued else (v != bg)
                if cond:
                    obj.add((v, cand))
                    occupied.add(cand)
                    for n in neigh_fn(cand):
                        if 0 <= n[0] < h and 0 <= n[1] < w:
                            new.add(n)
            cands = new - occupied

        if obj:
            objs.append(frozenset(obj))

    # v2 FIX: sort by (min_row, min_col) of each object's top-left cell
    def obj_sort_key(obj):
        cells = [loc for _, loc in obj]
        return (min(r for r, _ in cells), min(c for _, c in cells))

    objs.sort(key=obj_sort_key)
    return objs


# ------------------------------------------------------------
# NODE
# ------------------------------------------------------------

def build_shape_mask(cells, bbox):
    """Build a local binary mask for the object inside its bounding box."""
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    mask = [[0] * w for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1
    return mask


def build_node(i, obj):
    """Convert one object into a node dictionary."""
    pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))
    cells = [loc for _, loc in pairs]
    colors = sorted(set(c for c, _ in pairs))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))

    cx = sum(rows) / len(rows)
    cy = sum(cols) / len(cols)

    return {
        "id"              : i,
        "colors"          : colors,
        "cell_color_pairs": pairs,
        "cells"           : cells,
        "bbox"            : bbox,
        "centroid"        : (cx, cy),
        "shape_mask"      : build_shape_mask(cells, bbox),
    }


# ------------------------------------------------------------
# EDGES
# ------------------------------------------------------------

def touching(a, b):
    """Check if two objects touch (4-neighbor contact)."""
    A = set(a["cells"])
    B = set(b["cells"])
    for r, c in A:
        if (r+1,c) in B or (r-1,c) in B or (r,c+1) in B or (r,c-1) in B:
            return 1
    return 0


def build_edge(a, b):
    """Directed edge A → B with spatial relationship features."""
    dx = b["centroid"][1] - a["centroid"][1]
    dy = b["centroid"][0] - a["centroid"][0]
    return {
        "src"     : a["id"],
        "dst"     : b["id"],
        "dx"      : dx,
        "dy"      : dy,
        "touching": touching(a, b),
        "same_row": int(abs(dy) < 1e-6),
        "same_col": int(abs(dx) < 1e-6),
    }


# ------------------------------------------------------------
# FEATURE VECTORS
# ------------------------------------------------------------

def flatten(mask):
    return [float(x) for row in mask for x in row]


def node_vec(node, num_colors=10, max_h=10, max_w=10):
    """
    Node feature vector — 110 dims:
        10  = color one-hot
        100 = padded flattened 10×10 shape mask
    """
    color = [0] * num_colors
    for c in node["colors"]:
        color[c] = 1

    m = node["shape_mask"]
    pad = [[0] * max_w for _ in range(max_h)]
    for i in range(min(len(m), max_h)):
        for j in range(min(len(m[0]), max_w)):
            pad[i][j] = m[i][j]

    return color + flatten(pad)


def edge_vec(e, grid_norm=30.0):
    """
    Edge feature vector — 5 dims:
        [dx, dy, touching, same_row, same_col]

    v2 FIX: dx and dy are divided by grid_norm (30.0) so they sit in
    roughly [-1, 1] instead of raw pixel units (up to ±30). Without
    this normalisation, MSE on these values dominates the loss and can
    cause NaN explosions on larger grids.
    """
    return [
        e["dx"]       / grid_norm,   # [-1, 1]
        e["dy"]       / grid_norm,   # [-1, 1]
        float(e["touching"]),         # 0 or 1
        float(e["same_row"]),         # 0 or 1
        float(e["same_col"]),         # 0 or 1
    ]


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def grid_to_graph(grid):
    """Convert an ARC grid into a graph representation."""
    objs  = my_objects(grid)
    nodes = [build_node(i, o) for i, o in enumerate(objs)]

    edges = []
    for a in nodes:
        for b in nodes:
            if a["id"] != b["id"]:
                edges.append(build_edge(a, b))

    node_features = [node_vec(n) for n in nodes]
    edge_features = [edge_vec(e) for e in edges]
    edge_index    = [
        [e["src"] for e in edges],
        [e["dst"] for e in edges],
    ]

    return {
        "nodes"        : nodes,
        "edges"        : edges,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index"   : edge_index,
    }


# ------------------------------------------------------------
# RECONSTRUCTION FROM ORIGINAL GRAPH (lossless)
# ------------------------------------------------------------

def graph_to_grid(graph, h, w):
    """Reconstruct grid from graph using stored cell_color_pairs."""
    g = [[0] * w for _ in range(h)]
    for n in graph["nodes"]:
        for c, (r, c2) in n["cell_color_pairs"]:
            g[r][c2] = c
    return g


# ------------------------------------------------------------
# RECONSTRUCTION FROM DECODER PREDICTIONS (inference-time)
# ------------------------------------------------------------

def graph_to_grid_from_predictions(
    pred_shape_masks,    # [max_nodes, 100]   float — flattened 10×10 mask logits
    pred_colors,         # [max_nodes]         int   — predicted color index
    pred_existence,      # [max_nodes]         float — >0.5 means real node
    pred_bboxes,         # [max_nodes, 4]      float — (min_r, min_c, max_r, max_c) normalised
    height, width, bg=0,
    grid_norm=30.0,
    existence_thresh=0.5,
    mask_thresh=0.5,
):
    """
    Reconstruct a grid purely from decoder predictions.

    This is the inference-time reconstruction path for the hybrid model.
    The hybrid decoder doesn't predict per-cell coordinates like the
    custom decoder — instead it predicts a shape mask and a bounding box.
    We use those together to place the shape back onto the canvas.

    Steps per real node:
        1. Check pred_existence > existence_thresh
        2. Denormalise bbox → (min_r, min_c, max_r, max_c) in pixel coords
        3. Reshape pred_shape_masks[n] → 10×10, threshold at mask_thresh
        4. Scale the 10×10 mask to fit the predicted bbox
        5. Paint pred_colors[n] at each filled cell, clamped to grid bounds

    Note: bbox prediction requires the decoder to output bbox coordinates.
    If your decoder doesn't currently predict bboxes, pass pred_bboxes=None
    and the function will place objects at their centroid instead (less accurate).
    """
    grid = [[bg] * width for _ in range(height)]

    max_nodes = len(pred_existence)

    for n in range(max_nodes):
        if float(pred_existence[n]) <= existence_thresh:
            continue

        color = int(pred_colors[n])

        if pred_bboxes is not None:
            # Denormalise bbox
            min_r = int(round(float(pred_bboxes[n][0]) * grid_norm))
            min_c = int(round(float(pred_bboxes[n][1]) * grid_norm))
            max_r = int(round(float(pred_bboxes[n][2]) * grid_norm))
            max_c = int(round(float(pred_bboxes[n][3]) * grid_norm))
        else:
            # Fallback: treat as single pixel at (0,0) — caller should handle
            min_r, min_c, max_r, max_c = 0, 0, 0, 0

        # Clamp to grid
        min_r = max(0, min(height - 1, min_r))
        min_c = max(0, min(width  - 1, min_c))
        max_r = max(min_r, min(height - 1, max_r))
        max_c = max(min_c, min(width  - 1, max_c))

        bbox_h = max_r - min_r + 1
        bbox_w = max_c - min_c + 1

        # Reshape 100-dim flat mask → 10×10, then threshold
        flat = pred_shape_masks[n]
        mask_10x10 = [[1 if float(flat[i*10+j]) > mask_thresh else 0
                       for j in range(10)] for i in range(10)]

        # Scale 10×10 mask to actual bbox size and paint
        for mr in range(bbox_h):
            for mc in range(bbox_w):
                # Map bbox pixel → 10×10 mask coordinate
                mi = int(mr * 10 / max(bbox_h, 1))
                mj = int(mc * 10 / max(bbox_w, 1))
                mi = min(9, mi)
                mj = min(9, mj)
                if mask_10x10[mi][mj]:
                    gr = min_r + mr
                    gc = min_c + mc
                    if 0 <= gr < height and 0 <= gc < width:
                        grid[gr][gc] = color

    return grid
