# ============================================================
# HYBRID OBJECT BUILDER
# ============================================================
"""
This builder converts an ARC grid into a graph for a GNN.

"Hybrid" means:
- Node = visual structure (color + shape mask)
- Edge = explicit spatial relationships (dx, dy, touching, alignment)

This is a middle ground between:
- Minimal (too little info)
- Full custom (lots of engineered features)
"""

import math


# ------------------------------------------------------------
# 1. OBJECT EXTRACTION (NO DSL USED HERE)
# ------------------------------------------------------------
"""
We implement our own version of object extraction instead of using DSL.

Why?
- Gives full control over connectivity rules
- Makes behavior explicit and easier to debug
- Ensures consistency with hybrid representation
"""


def mostcolor(grid):
    """Return the most common value → treated as background."""
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
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
    """
    Extract connected components (objects) from grid.

    Key idea:
    - Each object = group of connected pixels
    - Connectivity = 4 or 8 neighbors
    - Can ignore background color

    Output:
    - List of objects
    - Each object = set of (color, (row, col))
    """

    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()

    h, w = len(grid), len(grid[0])
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    neigh_fn = neighbors if diagonal else dneighbors

    for loc in all_cells:
        if loc in occupied:
            continue

        val = grid[loc[0]][loc[1]]

        # Skip background pixels
        if without_bg and val == bg:
            continue

        obj = set()
        cands = {loc}

        # BFS-style expansion
        while cands:
            new = set()
            for cand in cands:
                if cand in occupied:
                    continue

                v = grid[cand[0]][cand[1]]

                # Grow object based on color rule
                cond = (val == v) if univalued else (v != bg)

                if cond:
                    obj.add((v, cand))
                    occupied.add(cand)

                    for n in neigh_fn(cand):
                        if 0 <= n[0] < h and 0 <= n[1] < w:
                            new.add(n)

            cands = new - occupied

        if obj:
            objs.add(frozenset(obj))

    return list(objs)


# ------------------------------------------------------------
# 2. NODE CONSTRUCTION (OBJECT → NODE)
# ------------------------------------------------------------
"""
Each object becomes ONE node.

Node contains:
- color(s)
- pixel locations
- bounding box
- shape mask (VERY important for hybrid)
"""


def build_shape_mask(cells, bbox):
    """
    Create a local binary image of the object.

    Why?
    → Gives the model actual SHAPE information
    → Allows learning of patterns (lines, holes, etc.)
    """
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = [[0] * w for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


def build_node(i, obj):
    """
    Convert object → node dictionary.

    Important:
    - centroid used ONLY for edges (not node features)
    """

    pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))
    cells = [loc for _, loc in pairs]

    colors = sorted(set(c for c, _ in pairs))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))

    # Used later for spatial relationships
    cx = sum(rows) / len(rows)
    cy = sum(cols) / len(cols)

    return {
        "id": i,
        "colors": colors,
        "cell_color_pairs": pairs,
        "cells": cells,
        "bbox": bbox,
        "centroid": (cx, cy),
        "shape_mask": build_shape_mask(cells, bbox)
    }


# ------------------------------------------------------------
# 3. EDGE CONSTRUCTION (RELATIONSHIPS)
# ------------------------------------------------------------
"""
Edges encode HOW objects relate in space.

This is what makes it "hybrid":
- shape = what objects are
- edges = how objects interact
"""


def touching(a, b):
    """Check if two objects touch (4-neighbor contact)."""
    A = set(a["cells"])
    B = set(b["cells"])

    for r, c in A:
        if (r + 1, c) in B or (r - 1, c) in B or (r, c + 1) in B or (r, c - 1) in B:
            return 1
    return 0


def build_edge(a, b):
    """
    Directed edge A → B.

    Features:
    - dx, dy → relative position
    - touching → adjacency
    - alignment → row/column structure
    """

    dx = b["centroid"][1] - a["centroid"][1]
    dy = b["centroid"][0] - a["centroid"][0]

    same_row = int(abs(dy) < 1e-6)
    same_col = int(abs(dx) < 1e-6)

    return {
        "src": a["id"],
        "dst": b["id"],
        "dx": dx,
        "dy": dy,
        "touching": touching(a, b),
        "same_row": same_row,
        "same_col": same_col
    }


# ------------------------------------------------------------
# 4. FEATURE VECTORS (FOR MODEL INPUT)
# ------------------------------------------------------------
"""
Convert human-readable nodes/edges → numeric tensors
"""


def flatten(mask):
    """Flatten 2D mask → 1D vector."""
    return [float(x) for row in mask for x in row]


def node_vec(node, num_colors=10, max_h=10, max_w=10):
    """
    Node feature = 110 dims:
    - 10 = color one-hot
    - 100 = shape mask
    """

    # Color encoding
    color = [0] * num_colors
    for c in node["colors"]:
        color[c] = 1

    # Shape (padded to fixed size)
    m = node["shape_mask"]
    pad = [[0] * max_w for _ in range(max_h)]

    for i in range(min(len(m), max_h)):
        for j in range(min(len(m[0]), max_w)):
            pad[i][j] = m[i][j]

    return color + flatten(pad)


def edge_vec(e):
    """Edge feature = 5 dims."""
    return [
        e["dx"], e["dy"],
        e["touching"],
        e["same_row"], e["same_col"]
    ]


# ------------------------------------------------------------
# 5. MAIN PIPELINE
# ------------------------------------------------------------
"""
This is what your training code calls.

Flow:
    grid → objects → nodes → edges → features
"""


def grid_to_graph(grid):
    # Step 1: extract objects
    objs = my_objects(grid)

    # Step 2: build nodes
    nodes = [build_node(i, o) for i, o in enumerate(objs)]

    # Step 3: fully connected graph
    edges = []
    for a in nodes:
        for b in nodes:
            if a["id"] != b["id"]:
                edges.append(build_edge(a, b))

    # Step 4: numeric conversion
    node_features = [node_vec(n) for n in nodes]
    edge_features = [edge_vec(e) for e in edges]
    edge_index = [
        [e["src"] for e in edges],
        [e["dst"] for e in edges]
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index
    }


def graph_to_grid(graph, h, w):
    """
    Reconstruct grid from graph.

    Used to verify:
    → Did we lose information?
    """
    g = [[0] * w for _ in range(h)]
    for n in graph["nodes"]:
        for c, (r, c2) in n["cell_color_pairs"]:
            g[r][c2] = c
    return g