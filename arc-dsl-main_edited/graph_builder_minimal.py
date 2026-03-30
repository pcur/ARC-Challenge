# ============================================================
# graph_builder_minimal_custom.py
# ============================================================
#
# Minimal custom ARC graph builder
#
# Uses ONLY:
#   1. Color - Obj
#   2. Color - Pixel
#   3. Shape - Obj
#
# Does NOT use:
#   - dsl.objects
#   - centroid
#   - area
#   - bbox as model feature
#   - width / height
#   - density
#   - aspect ratio
#   - relational edges
#
# The object extraction is fully custom.
# ============================================================


# ------------------------------------------------------------
# BASIC HELPERS FOR CUSTOM OBJECT EXTRACTION
# ------------------------------------------------------------

def mostcolor(grid):
    """
    Return the most common color in the grid.
    Used as the background color when without_bg=True.
    """
    counts = {}
    for row in grid:
        for val in row:
            counts[val] = counts.get(val, 0) + 1
    return max(counts, key=counts.get)


def dneighbors(cell):
    """
    4-connected neighbors:
    up, down, left, right
    """
    r, c = cell
    return {
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def neighbors(cell):
    """
    8-connected neighbors:
    includes diagonals
    """
    r, c = cell
    return {
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1),                 (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
    """
    Custom object extractor.

    Parameters
    ----------
    grid : ARC grid
    univalued : if True, object must stay same color as seed cell
    diagonal : if True, use 8-connectivity; else 4-connectivity
    without_bg : if True, ignore most common color

    Returns
    -------
    frozenset of objects, where each object is:
        frozenset of (color, (row, col))
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

        if without_bg and val == bg:
            continue

        obj = set()
        cands = {loc}

        while cands:
            neighborhood = set()

            for cand in cands:
                if cand in occupied:
                    continue

                v = grid[cand[0]][cand[1]]

                # Same-color connected component if univalued=True
                # Otherwise any non-background connected region
                condition = (val == v) if univalued else (v != bg)

                if condition:
                    obj.add((v, cand))
                    occupied.add(cand)

                    for nr, nc in neigh_fn(cand):
                        if 0 <= nr < h and 0 <= nc < w:
                            neighborhood.add((nr, nc))

            cands = neighborhood - occupied

        if obj:
            objs.add(frozenset(obj))

    return frozenset(objs)


# ------------------------------------------------------------
# SHAPE HELPER
# ------------------------------------------------------------

def build_shape_mask(cells, bbox):
    """
    Build a local binary mask for the object inside its bbox.

    Example:
        If object cells occupy a 2x3 local box, the mask marks
        filled cells as 1 and empty cells as 0.
    """
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


# ------------------------------------------------------------
# MINIMAL NODE CONSTRUCTION
# ------------------------------------------------------------

def build_node(node_id, obj):
    """
    Build a minimal node using ONLY:

    1. Color - Obj   -> colors
    2. Color - Pixel -> cell_color_pairs
    3. Shape - Obj   -> shape_mask
    """
    # Stable ordering for reproducibility/debugging
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))

    # Extract coordinates
    cells = [loc for _, loc in cell_color_pairs]

    # Object-level colors
    colors = sorted(list(set(color for color, _ in cell_color_pairs)))

    # Bounding box is needed only to define local shape coordinates
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    bbox = (min_r, min_c, max_r, max_c)

    # Object-level shape representation
    shape_mask = build_shape_mask(cells, bbox)

    return {
        "id": node_id,

        # Requested minimal parameters
        "colors": colors,                      # Color - Obj
        "cell_color_pairs": cell_color_pairs,  # Color - Pixel
        "shape_mask": shape_mask,              # Shape - Obj

        # Kept only as support/debug info
        "cells": cells,
        "bbox": bbox,
    }


# ------------------------------------------------------------
# FEATURE CONVERSION
# ------------------------------------------------------------

def flatten_shape_mask(mask):
    """
    Flatten a 2D shape mask into a 1D list.
    """
    return [float(v) for row in mask for v in row]


def node_to_feature_vector(node, num_colors=10, max_shape_h=10, max_shape_w=10):
    """
    Convert minimal node to a fixed-length numeric vector.

    Output:
        [ object-color multi-hot | padded flattened shape mask ]

    Notes:
    - Color - Obj is encoded directly
    - Shape - Obj is encoded directly
    - Color - Pixel is preserved in the node dictionary, but not
      flattened into the numeric vector because object pixel count
      varies from object to object
    """
    # --------------------------------------------------------
    # Color - Obj
    # --------------------------------------------------------
    color_multi_hot = [0.0] * num_colors
    for color in node["colors"]:
        if 0 <= color < num_colors:
            color_multi_hot[color] = 1.0

    # --------------------------------------------------------
    # Shape - Obj
    # --------------------------------------------------------
    mask = node["shape_mask"]
    h = len(mask)
    w = len(mask[0]) if h > 0 else 0

    # Pad/crop shape mask to a fixed size
    padded = [[0.0 for _ in range(max_shape_w)] for _ in range(max_shape_h)]
    for r in range(min(h, max_shape_h)):
        for c in range(min(w, max_shape_w)):
            padded[r][c] = float(mask[r][c])

    shape_vec = flatten_shape_mask(padded)

    return color_multi_hot + shape_vec


def graph_to_numeric(graph, num_colors=10, max_shape_h=10, max_shape_w=10):
    """
    Convert graph to numeric model-ready form.

    Since this is the minimal builder, we intentionally do not
    create relational edge features.
    """
    node_features = [
        node_to_feature_vector(
            node,
            num_colors=num_colors,
            max_shape_h=max_shape_h,
            max_shape_w=max_shape_w,
        )
        for node in graph["nodes"]
    ]

    edge_features = []
    edge_index = [[], []]

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
    }


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def grid_to_graph(
    grid,
    univalued=True,
    diagonal=False,
    without_bg=True,
    num_colors=10,
    max_shape_h=10,
    max_shape_w=10,
):
    """
    Main minimal custom builder pipeline:

        ARC grid
          -> custom object extraction
          -> minimal nodes
          -> numeric node features
    """
    # Step 1: custom object extraction
    raw_objs = my_objects(
        grid,
        univalued=univalued,
        diagonal=diagonal,
        without_bg=without_bg,
    )

    # Step 2: minimal node construction
    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

    # Step 3: no relational edges in this minimal version
    graph = {
        "nodes": nodes,
        "edges": [],
    }

    # Step 4: numeric form
    numeric = graph_to_numeric(
        graph,
        num_colors=num_colors,
        max_shape_h=max_shape_h,
        max_shape_w=max_shape_w,
    )
    graph["node_features"] = numeric["node_features"]
    graph["edge_features"] = numeric["edge_features"]
    graph["edge_index"] = numeric["edge_index"]

    return graph


# ------------------------------------------------------------
# RECONSTRUCTION
# ------------------------------------------------------------

def graph_to_grid(graph, height, width, bg=0):
    """
    Reconstruct exact grid from Color - Pixel information.
    """
    grid = [[bg for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color

    return grid