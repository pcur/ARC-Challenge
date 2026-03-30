import math
from dsl import objects


# ------------------------------------------------------------
# safe_div(a, b)
# ------------------------------------------------------------
# Utility helper to avoid division-by-zero errors.
# If b is 0, return 0.0 instead of crashing.
# This is useful for quantities like:
#   density = area / bbox_area
#   aspect_ratio = width / height
# ------------------------------------------------------------
def safe_div(a, b):
    return a / b if b != 0 else 0.0


# ------------------------------------------------------------
# build_shape_mask(cells, bbox)
# ------------------------------------------------------------
# Purpose:
#   Build a small binary mask showing the object's shape INSIDE
#   its own bounding box.
#
# Inputs:
#   cells = list of (row, col) positions belonging to the object
#   bbox  = (min_row, min_col, max_row, max_col)
#
# Example:
#   If an object occupies these grid cells:
#       (4,5), (4,6), (5,5)
#   and its bbox is:
#       (4,5,5,6)
#   then the local mask becomes:
#       [1, 1]
#       [1, 0]
#
# Why this is useful:
#   This preserves the object's local shape relative to its bbox.
#   Right now this is mainly helpful for debugging/inspection.
# ------------------------------------------------------------
def build_shape_mask(cells, bbox):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Initialize an empty local mask (all zeros)
    mask = [[0 for _ in range(w)] for _ in range(h)]

    # Mark each object cell as 1 in local bbox coordinates
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


# ------------------------------------------------------------
# touching(cells_a, cells_b)
# ------------------------------------------------------------
# Purpose:
#   Check whether two objects are directly adjacent using
#   4-neighborhood connectivity (up, down, left, right).
#
# Inputs:
#   cells_a = list of cells for object A
#   cells_b = list of cells for object B
#
# Returns:
#   True if any cell in A touches any cell in B
#   False otherwise
#
# Why this matters:
#   "Touching" is a useful relationship between objects and
#   becomes one of the edge features in the graph.
# ------------------------------------------------------------
def touching(cells_a, cells_b):
    # Turn B into a set for fast membership checking
    bset = set(cells_b)

    # For every cell in A, check its 4-neighbors
    for r, c in cells_a:
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) in bset:
                return True

    return False


# ------------------------------------------------------------
# bbox_overlap(bbox_a, bbox_b)
# ------------------------------------------------------------
# Purpose:
#   Check whether the bounding boxes of two objects overlap.
#
# Inputs:
#   bbox_a = (min_row, min_col, max_row, max_col)
#   bbox_b = (min_row, min_col, max_row, max_col)
#
# Returns:
#   True if the rectangles overlap in both row and column ranges
#   False otherwise
#
# Why this matters:
#   Even if two objects do not directly touch, their bounding
#   boxes might overlap or intersect, which is another useful
#   edge feature.
# ------------------------------------------------------------
def bbox_overlap(bbox_a, bbox_b):
    min_ra, min_ca, max_ra, max_ca = bbox_a
    min_rb, min_cb, max_rb, max_cb = bbox_b

    # Check overlap in row direction
    row_overlap = not (max_ra < min_rb or max_rb < min_ra)

    # Check overlap in column direction
    col_overlap = not (max_ca < min_cb or max_cb < min_ca)

    return row_overlap and col_overlap


# ------------------------------------------------------------
# build_node(node_id, obj)
# ------------------------------------------------------------
# Purpose:
#   Convert one raw DSL object into a structured graph node.
#
# Input:
#   obj = a frozenset of (color, (row, col)) pairs returned
#         by dsl.objects(...)
#
# What this function computes:
#   - exact cells belonging to the object
#   - object colors
#   - area (# of pixels)
#   - centroid (average row, average col)
#   - bounding box
#   - width / height
#   - density
#   - aspect ratio
#   - whether it's a single-pixel object
#   - local shape mask
#
# Important:
#   cell_color_pairs is stored so we can reconstruct the exact
#   original grid later.
# ------------------------------------------------------------
def build_node(node_id, obj):
    # obj is a frozenset of (color, (row, col))
    # Sort it so the representation is stable and easier to inspect
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))

    # Extract just the coordinates
    cells = [loc for _, loc in cell_color_pairs]

    # Collect unique colors present in this object
    colors = sorted(list(set(color for color, _ in cell_color_pairs)))

    # Separate rows and cols for geometry calculations
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    # Area = number of pixels in the object
    area = len(cells)

    # Bounding box extents
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Width/height of the bbox
    width = max_c - min_c + 1
    height = max_r - min_r + 1

    # Centroid = average row and average col
    centroid = (sum(rows) / area, sum(cols) / area)

    # Bounding box stored as tuple
    bbox = (min_r, min_c, max_r, max_c)

    # Density = fraction of bbox actually occupied by the object
    bbox_area = width * height
    density = safe_div(area, bbox_area)

    # Aspect ratio = width / height
    aspect_ratio = safe_div(width, height)

    # Local binary shape inside the bbox
    shape_mask = build_shape_mask(cells, bbox)

    # Return a node dictionary
    return {
        "id": node_id,
        "colors": colors,
        "cell_color_pairs": cell_color_pairs,  # exact reverse mapping support
        "cells": cells,
        "area": area,
        "centroid": centroid,
        "bbox": bbox,
        "width": width,
        "height": height,
        "density": density,
        "aspect_ratio": aspect_ratio,
        "is_single_pixel": int(area == 1),
        "shape_mask": shape_mask,
    }


# ------------------------------------------------------------
# build_edge(a, b)
# ------------------------------------------------------------
# Purpose:
#   Build one directed edge from node A to node B.
#
# Inputs:
#   a = source node dictionary
#   b = target node dictionary
#
# What this computes:
#   - relative position (dx, dy)
#   - distances
#   - same color?
#   - touching?
#   - bbox overlap?
#   - same row / same col?
#   - same area?
#   - relative size ratios
#
# Important:
#   This is a directed edge, so A->B and B->A are different.
#   In particular, dx/dy and area ratios depend on direction.
# ------------------------------------------------------------
def build_edge(a, b):
    # Source node centroid
    ar, ac = a["centroid"]

    # Target node centroid
    br, bc = b["centroid"]

    # Relative offset from A to B
    dx = bc - ac
    dy = br - ar

    # Object sizes
    area_a = a["area"]
    area_b = b["area"]

    return {
        "src": a["id"],
        "dst": b["id"],

        # Relative position
        "dx": dx,
        "dy": dy,

        # Distance measures
        "manhattan": abs(dx) + abs(dy),
        "dist": math.sqrt(dx * dx + dy * dy),

        # Appearance relationship
        "same_color": int(bool(set(a["colors"]) & set(b["colors"]))),

        # Contact / overlap relationships
        "touching": int(touching(a["cells"], b["cells"])),
        "bbox_overlap": int(bbox_overlap(a["bbox"], b["bbox"])),

        # Alignment relationships
        "same_row": int(round(ar) == round(br)),
        "same_col": int(round(ac) == round(bc)),

        # Size relationships
        "same_area": int(area_a == area_b),
        "area_ratio_ab": safe_div(area_a, area_b),
        "area_ratio_ba": safe_div(area_b, area_a),
    }


# ------------------------------------------------------------
# node_to_feature_vector(node, num_colors=10)
# ------------------------------------------------------------
# Purpose:
#   Convert one node dictionary into a fixed-length numeric
#   feature vector for a model.
#
# Output length:
#   22 dimensions
#
# Breakdown:
#   10 dims = one-hot color
#   12 dims = geometry / size / shape summary
#
# Important:
#   This is what gets passed to the model, not the full node dict.
# ------------------------------------------------------------
def node_to_feature_vector(node, num_colors=10):
    # One-hot encode color(s)
    color_one_hot = [0.0] * num_colors
    for color in node["colors"]:
        if 0 <= color < num_colors:
            color_one_hot[color] = 1.0

    min_r, min_c, max_r, max_c = node["bbox"]

    return color_one_hot + [
        float(node["area"]),
        float(node["centroid"][0]),
        float(node["centroid"][1]),
        float(min_r),
        float(min_c),
        float(max_r),
        float(max_c),
        float(node["width"]),
        float(node["height"]),
        float(node["density"]),
        float(node["aspect_ratio"]),
        float(node["is_single_pixel"]),
    ]


# ------------------------------------------------------------
# edge_to_feature_vector(edge)
# ------------------------------------------------------------
# Purpose:
#   Convert one edge dictionary into a fixed-length numeric
#   feature vector for a model.
#
# Output length:
#   12 dimensions
#
# Important:
#   This is the model-facing version of the edge, not the full
#   edge dictionary used for debugging.
# ------------------------------------------------------------
def edge_to_feature_vector(edge):
    return [
        float(edge["dx"]),
        float(edge["dy"]),
        float(edge["manhattan"]),
        float(edge["dist"]),
        float(edge["same_color"]),
        float(edge["touching"]),
        float(edge["bbox_overlap"]),
        float(edge["same_row"]),
        float(edge["same_col"]),
        float(edge["same_area"]),
        float(edge["area_ratio_ab"]),
        float(edge["area_ratio_ba"]),
    ]


# ------------------------------------------------------------
# graph_to_numeric(graph, num_colors=10)
# ------------------------------------------------------------
# Purpose:
#   Convert the human-readable graph dictionary into the three
#   numeric structures that a GNN/GAT/VAE would actually use:
#
#   1. node_features : [N, F_node]
#   2. edge_features : [E, F_edge]
#   3. edge_index    : [2, E]
#
# edge_index format:
#   [source_nodes,
#    target_nodes]
#
# Important:
#   This is the actual "model input" representation.
# ------------------------------------------------------------
def graph_to_numeric(graph, num_colors=10):
    # Convert each node dict to a numeric feature vector
    node_features = [
        node_to_feature_vector(node, num_colors=num_colors)
        for node in graph["nodes"]
    ]

    # Convert each edge dict to a numeric feature vector
    edge_features = [
        edge_to_feature_vector(edge)
        for edge in graph["edges"]
    ]

    # Standard COO-style connectivity representation
    edge_index = [
        [edge["src"] for edge in graph["edges"]],
        [edge["dst"] for edge in graph["edges"]],
    ]

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
    }


# ------------------------------------------------------------
# grid_to_graph(...)
# ------------------------------------------------------------
# Purpose:
#   Main builder function.
#
# Pipeline:
#   ARC grid
#     → DSL objects(...)
#     → nodes
#     → edges
#     → numeric graph representation
#
# Inputs:
#   grid            = ARC grid
#   univalued       = whether objects must be single-color
#   diagonal        = whether diagonal connectivity counts
#   without_bg      = whether to ignore background color
#   fully_connected = whether every pair of nodes gets edges
#   num_colors      = size of color one-hot encoding
#
# Output:
#   A graph dictionary containing:
#     - nodes / edges (human-readable)
#     - node_features / edge_features / edge_index (model-ready)
# ------------------------------------------------------------
def grid_to_graph(
    grid,
    univalued=True,
    diagonal=False,
    without_bg=True,
    fully_connected=True,
    num_colors=10,
):
    # --------------------------------------------------------
    # Step 1: Use the DSL to extract raw objects from the grid.
    #
    # Each object is returned as a frozenset of:
    #   (color, (row, col))
    # --------------------------------------------------------
    raw_objs = objects(
        grid,
        univalued=univalued,
        diagonal=diagonal,
        without_bg=without_bg,
    )

    # --------------------------------------------------------
    # Step 2: Convert each raw object into a structured node.
    # --------------------------------------------------------
    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

    # --------------------------------------------------------
    # Step 3: Build edges between nodes.
    #
    # If fully_connected=True:
    #   connect every node to every other node (directed)
    #
    # If fully_connected=False:
    #   only connect nodes when there is some meaningful
    #   relationship like touching, alignment, or overlap
    # --------------------------------------------------------
    edges = []

    if fully_connected:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edges.append(build_edge(nodes[i], nodes[j]))
    else:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                e1 = build_edge(nodes[i], nodes[j])
                e2 = build_edge(nodes[j], nodes[i])

                if e1["touching"] or e1["same_row"] or e1["same_col"] or e1["bbox_overlap"]:
                    edges.append(e1)
                    edges.append(e2)

    # --------------------------------------------------------
    # Step 4: Assemble the human-readable graph
    # --------------------------------------------------------
    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    # --------------------------------------------------------
    # Step 5: Build numeric model-ready representation
    # --------------------------------------------------------
    numeric = graph_to_numeric(graph, num_colors=num_colors)
    graph["node_features"] = numeric["node_features"]
    graph["edge_features"] = numeric["edge_features"]
    graph["edge_index"] = numeric["edge_index"]

    return graph


# ------------------------------------------------------------
# graph_to_grid(graph, height, width, bg=0)
# ------------------------------------------------------------
# Purpose:
#   Reconstruct the original grid from the graph.
#
# Why this matters:
#   This lets us verify that the graph representation preserved
#   all original information.
#
# How it works:
#   We initialize a blank background grid, then repaint every
#   object's original cells using the stored cell_color_pairs.
#
# Important:
#   This is why we kept exact cell location/color information
#   inside each node.
# ------------------------------------------------------------
def graph_to_grid(graph, height, width, bg=0):
    # Start with a blank grid filled with background color
    grid = [[bg for _ in range(width)] for _ in range(height)]

    # Paint back each stored pixel exactly where it came from
    for node in graph["nodes"]:
        # Exact reverse mapping using stored per-cell color data
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color

    return grid