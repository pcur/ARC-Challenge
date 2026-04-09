import math


# ============================================================
# CUSTOM OBJECT / GRAPH BUILDER FOR ARC GRIDS
# ============================================================
#
# This file does four main things:
#
# 1. Extracts connected objects from an ARC grid
# 2. Turns each object into a graph node
# 3. Builds edges between nodes based on relationships
# 4. Converts the graph into numeric feature arrays that a
#    machine learning model (GNN / GAT / Graph Autoencoder /
#    etc.) could consume
#
# The script also includes a reverse path:
#
#     graph -> grid
#
# so that you can verify whether the representation is
# lossless with respect to the original colored cells.
#
# ------------------------------------------------------------
# IMPORTANT DESIGN IDEA
# ------------------------------------------------------------
# In ARC, the grid is a 2D colored matrix.
# Instead of feeding raw pixels directly into a model,
# we first group related cells into "objects."
#
# Each object becomes a node with:
#   - color information
#   - geometry information
#   - position information
#   - shape information
#
# Relationships between pairs of objects become edges with:
#   - relative position
#   - distance
#   - touching / overlap
#   - alignment
#   - size comparisons
#
# This creates a structured representation that is often
# easier for reasoning systems than raw grid pixels alone.
# ============================================================


# ------------------------------------------------------------
# BASIC HELPERS FOR CUSTOM OBJECT EXTRACTION
# ------------------------------------------------------------

def mostcolor(grid):
    """
    Return the most common color in the grid.

    Why this matters:
        In many ARC tasks, the most frequent color acts as the
        background color. If we want to ignore the background
        during object extraction, we need a way to estimate what
        the background is.

    Inputs:
        grid : list-of-lists or tuple-of-tuples of integers

    Returns:
        int : the color value that appears most often

    Example:
        grid =
        [
            [0, 0, 0],
            [0, 2, 2],
            [0, 0, 0]
        ]

        mostcolor(grid) -> 0
    """
    counts = {}

    # Count how many times each color appears in the grid
    for row in grid:
        for val in row:
            counts[val] = counts.get(val, 0) + 1

    # Return the color with the highest count
    return max(counts, key=counts.get)


def dneighbors(cell):
    """
    Return the 4-connected neighbors of a cell.

    4-connected means:
        up, down, left, right

    This is the standard "non-diagonal" neighborhood.

    Input:
        cell : (row, col)

    Returns:
        set of neighboring coordinates

    Example:
        dneighbors((3, 5)) ->
        {
            (2, 5),   # up
            (4, 5),   # down
            (3, 4),   # left
            (3, 6),   # right
        }
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
    Return the 8-connected neighbors of a cell.

    8-connected means:
        up, down, left, right, plus the four diagonals

    Input:
        cell : (row, col)

    Returns:
        set of neighboring coordinates
    """
    r, c = cell
    return {
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1),                 (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
    """
    Custom replacement for the ARC DSL objects(...) function.

    PURPOSE
    -------
    This function scans the grid and groups cells into connected
    objects.

    CONNECTION RULES
    ----------------
    The connectivity depends on the arguments:
      - diagonal=False  -> use 4-connectivity
      - diagonal=True   -> use 8-connectivity

    COLOR RULES
    -----------
    The color consistency depends on:
      - univalued=True
            Each object must contain only the same color as the
            starting seed cell. This is the usual "one object =
            one color blob" behavior.
      - univalued=False
            Any connected non-background cells can be grouped
            together even if their colors differ.

    BACKGROUND RULES
    ----------------
      - without_bg=True
            Ignore the most common color entirely.
      - without_bg=False
            Treat every color as eligible for object extraction.

    INPUTS
    ------
    grid        : ARC grid (list-of-lists or tuple-of-tuples)
    univalued   : bool
    diagonal    : bool
    without_bg  : bool

    RETURNS
    -------
    frozenset of objects

    Each object is represented as:
        frozenset of (color, (row, col))

    WHY THIS FORMAT?
    ----------------
    We keep both the cell location and the original color.
    That makes reconstruction possible later.

    HIGH-LEVEL ALGORITHM
    --------------------
    1. Determine background color if needed
    2. Visit every cell in the grid
    3. If the cell is not already assigned to an object,
       start a region-growth / flood-fill process
    4. Expand through valid neighbors
    5. Save the completed object
    """
    # Determine which color should be treated as background.
    # If without_bg=False, then background is ignored and we
    # simply set bg=None.
    bg = mostcolor(grid) if without_bg else None

    # "objs" will hold all discovered objects
    objs = set()

    # "occupied" tracks cells that are already assigned to
    # some object so that we do not process them twice
    occupied = set()

    # Grid dimensions
    h, w = len(grid), len(grid[0])

    # Build a set of every coordinate in the grid
    all_cells = {(r, c) for r in range(h) for c in range(w)}

    # Choose the neighborhood function:
    # 4-neighbor or 8-neighbor
    neigh_fn = neighbors if diagonal else dneighbors

    # Iterate through every cell in the grid
    for loc in all_cells:
        # If this cell has already been assigned to an object,
        # skip it
        if loc in occupied:
            continue

        # Value/color at this starting cell
        val = grid[loc[0]][loc[1]]

        # If we are ignoring background and this cell is
        # background, skip it
        if without_bg and val == bg:
            continue

        # Start building a new object
        obj = set()

        # "cands" = current frontier of cells to consider
        # This begins with the seed cell
        cands = {loc}

        # Region-growth loop
        while cands:
            # Collect the next frontier of neighboring cells
            neighborhood = set()

            for cand in cands:
                # If this candidate already belongs to an object,
                # ignore it
                if cand in occupied:
                    continue

                # Candidate cell color
                v = grid[cand[0]][cand[1]]

                # Decide whether this cell is allowed to join
                # the current object.
                #
                # If univalued=True:
                #   only cells of the same color as the seed cell
                #   are allowed to join
                #
                # If univalued=False:
                #   any non-background connected cell is allowed
                condition = (val == v) if univalued else (v != bg)

                if condition:
                    # Store the cell in the object as:
                    #   (its color, its coordinate)
                    obj.add((v, cand))

                    # Mark the cell as already assigned
                    occupied.add(cand)

                    # Add all valid neighbors to the neighborhood
                    # set so they can be checked on the next pass
                    for nr, nc in neigh_fn(cand):
                        if 0 <= nr < h and 0 <= nc < w:
                            neighborhood.add((nr, nc))

            # New frontier = neighbor cells that are not yet used
            cands = neighborhood - occupied

        # If we found at least one cell, save this object
        if obj:
            objs.add(frozenset(obj))

    return frozenset(objs)


# ------------------------------------------------------------
# GENERAL MATH / GEOMETRY HELPERS
# ------------------------------------------------------------

def safe_div(a, b):
    """
    Safely divide a by b.

    Why this exists:
        Some geometric ratios can accidentally divide by zero,
        especially if a malformed input or edge case appears.

    Behavior:
        If b != 0, return a / b
        If b == 0, return 0.0

    Example:
        safe_div(6, 3) -> 2.0
        safe_div(6, 0) -> 0.0
    """
    return a / b if b != 0 else 0.0


def build_shape_mask(cells, bbox):
    """
    Build a local binary mask of an object inside its bounding box.

    PURPOSE
    -------
    Suppose an object occupies only part of a rectangle.
    This function creates a small local grid showing which cells
    are filled (1) and which are empty (0) within that box.

    INPUTS
    ------
    cells : list of (row, col) coordinates belonging to object
    bbox  : (min_row, min_col, max_row, max_col)

    RETURNS
    -------
    2D list of 0/1 values

    EXAMPLE
    -------
    If an object occupies:
        (2,3), (2,4), (3,3)

    and bbox is:
        (2,3,3,4)

    then the local mask is:
        [
            [1, 1],
            [1, 0]
        ]

    WHY THIS IS USEFUL
    ------------------
    This can later be used for:
      - debugging
      - shape comparison
      - future shape descriptors
      - reconstruction checks
    """
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Initialize an empty local mask
    mask = [[0 for _ in range(w)] for _ in range(h)]

    # Mark object cells as 1 inside the local bounding box frame
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


def touching(cells_a, cells_b):
    """
    Check whether two objects touch using 4-connectivity.

    DEFINITION OF TOUCHING HERE
    ---------------------------
    Two objects are considered touching if at least one cell in
    object A is directly adjacent (up/down/left/right) to a cell
    in object B.

    NOTE
    ----
    Diagonal contact does NOT count here.

    INPUTS
    ------
    cells_a : list of coordinates for object A
    cells_b : list of coordinates for object B

    RETURNS
    -------
    True  -> objects touch
    False -> objects do not touch
    """
    # Convert B into a set for fast membership checks
    bset = set(cells_b)

    # For each cell in A, check its 4-neighbors
    for r, c in cells_a:
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) in bset:
                return True

    return False


def bbox_overlap(bbox_a, bbox_b):
    """
    Check whether two bounding boxes overlap.

    BBOX FORMAT
    -----------
        (min_row, min_col, max_row, max_col)

    OVERLAP MEANING
    ---------------
    Two bounding boxes overlap if they share at least one row
    interval and at least one column interval.

    INPUTS
    ------
    bbox_a, bbox_b : bounding box tuples

    RETURNS
    -------
    True  -> boxes overlap
    False -> boxes do not overlap

    NOTE
    ----
    This is box overlap, not necessarily object-cell overlap.
    Two sparse shapes could have overlapping bounding boxes
    without actually sharing cells.
    """
    min_ra, min_ca, max_ra, max_ca = bbox_a
    min_rb, min_cb, max_rb, max_cb = bbox_b

    # Check interval overlap on rows
    row_overlap = not (max_ra < min_rb or max_rb < min_ra)

    # Check interval overlap on columns
    col_overlap = not (max_ca < min_cb or max_cb < min_ca)

    return row_overlap and col_overlap


# ------------------------------------------------------------
# NODE CONSTRUCTION
# ------------------------------------------------------------

def build_node(node_id, obj):
    """
    Convert one extracted object into a structured node dictionary.

    INPUT
    -----
    obj : frozenset of (color, (row, col))

    OUTPUT
    ------
    dictionary describing one graph node

    The node includes:
      - id
      - color list
      - original cells
      - area
      - centroid
      - bounding box
      - width / height
      - density
      - aspect ratio
      - single-pixel flag
      - local shape mask

    WHY THIS FUNCTION EXISTS
    ------------------------
    Object extraction gives us raw cell sets.
    This function computes richer properties that are useful
    for graph reasoning and machine learning.
    """
    # Sort cells for stable ordering.
    # This helps debugging and makes output deterministic.
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))

    # Extract only the coordinates
    cells = [loc for _, loc in cell_color_pairs]

    # Collect the unique colors present in the object
    colors = sorted(list(set(color for color, _ in cell_color_pairs)))

    # Separate row list and column list
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    # Area = number of occupied cells / pixels
    area = len(cells)

    # Bounding box extremes
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Width / height of the bounding box
    width = max_c - min_c + 1
    height = max_r - min_r + 1

    # Centroid = average row and average column position
    centroid = (sum(rows) / area, sum(cols) / area)

    # Save bbox in a standard tuple format
    bbox = (min_r, min_c, max_r, max_c)

    # Bbox area = rectangle area enclosing the object
    bbox_area = width * height

    # Density tells us how "filled" the bbox is:
    #   1.0 = completely filled rectangle
    #   lower values = sparser shape
    density = safe_div(area, bbox_area)

    # Aspect ratio = width / height
    aspect_ratio = safe_div(width, height)

    # Local binary shape representation
    shape_mask = build_shape_mask(cells, bbox)

    return {
        "id": node_id,
        "colors": colors,
        "cell_color_pairs": cell_color_pairs,   # exact colored cells for reconstruction
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
# EDGE CONSTRUCTION
# ------------------------------------------------------------

def build_edge(a, b):
    """
    Build one directed edge from node A to node B.

    INPUTS
    ------
    a, b : node dictionaries

    OUTPUT
    ------
    edge dictionary describing the relationship from A -> B

    IMPORTANT
    ---------
    This is a DIRECTED edge.
    That means A -> B and B -> A are treated separately.

    WHY DIRECTED?
    -------------
    Because relative position depends on direction.
    For example:
        dx from A to B = +4
        dx from B to A = -4

    INCLUDED RELATIONS
    ------------------
    - relative position
    - distance
    - same color overlap
    - touching
    - bbox overlap
    - row / column alignment
    - same area
    - area ratios
    """
    # Centroids of the two nodes
    ar, ac = a["centroid"]
    br, bc = b["centroid"]

    # Relative position of B with respect to A
    # dy = row difference
    # dx = column difference
    dx = bc - ac
    dy = br - ar

    area_a = a["area"]
    area_b = b["area"]

    return {
        "src": a["id"],
        "dst": b["id"],

        # Relative position from A to B
        "dx": dx,
        "dy": dy,

        # Manhattan distance = |dx| + |dy|
        "manhattan": abs(dx) + abs(dy),

        # Euclidean distance
        "dist": math.sqrt(dx * dx + dy * dy),

        # same_color = 1 if the two objects share at least one color
        "same_color": int(bool(set(a["colors"]) & set(b["colors"]))),

        # touching = 1 if the objects are 4-neighbor adjacent anywhere
        "touching": int(touching(a["cells"], b["cells"])),

        # bbox_overlap = 1 if their bounding boxes overlap
        "bbox_overlap": int(bbox_overlap(a["bbox"], b["bbox"])),

        # same_row / same_col check rough centroid alignment.
        # round(...) is used because centroids may be fractional.
        "same_row": int(round(ar) == round(br)),
        "same_col": int(round(ac) == round(bc)),

        # same_area = identical number of cells
        "same_area": int(area_a == area_b),

        # Relative size ratios
        "area_ratio_ab": safe_div(area_a, area_b),
        "area_ratio_ba": safe_div(area_b, area_a),
    }


# ------------------------------------------------------------
# NODE FEATURE VECTOR
# ------------------------------------------------------------

def node_to_feature_vector(node, num_colors=10):
    """
    Convert a node dictionary into a numeric feature vector.

    PURPOSE
    -------
    Human-readable node dictionaries are good for debugging,
    but machine learning models need numeric arrays.

    OUTPUT DIMENSION
    ----------------
    22

    FEATURE BREAKDOWN
    -----------------
    10 values : one-hot / multi-hot color encoding
    12 values : geometric / positional features

    COLOR ENCODING
    --------------
    We allocate one slot per color index from 0..num_colors-1.
    If the object contains a color, that slot becomes 1.0.

    NOTE
    ----
    This is actually multi-hot, not strictly one-hot,
    because a multi-colored object can activate multiple slots.
    """
    # Initialize all color positions to 0
    color_one_hot = [0.0] * num_colors

    # Activate the positions corresponding to the node's colors
    for color in node["colors"]:
        if 0 <= color < num_colors:
            color_one_hot[color] = 1.0

    min_r, min_c, max_r, max_c = node["bbox"]

    # Concatenate color encoding + scalar geometry features
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
# EDGE FEATURE VECTOR
# ------------------------------------------------------------

def edge_to_feature_vector(edge):
    """
    Convert an edge dictionary into a numeric feature vector.

    PURPOSE
    -------
    Like nodes, edges must also be converted to plain numbers
    before being fed into most graph models.

    OUTPUT DIMENSION
    ----------------
    12

    FEATURE ORDER
    -------------
    [dx, dy, manhattan, dist, same_color, touching,
     bbox_overlap, same_row, same_col, same_area,
     area_ratio_ab, area_ratio_ba]
    """
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
# GRAPH -> NUMERIC FORMAT
# ------------------------------------------------------------

def graph_to_numeric(graph, num_colors=10):
    """
    Convert the human-readable graph into model-ready numeric form.

    INPUT
    -----
    graph : dictionary containing:
        - graph["nodes"]
        - graph["edges"]

    RETURNS
    -------
    dictionary with:
        node_features : [N, 22]
        edge_features : [E, 12]
        edge_index    : [2, E]

    WHAT edge_index MEANS
    ---------------------
    edge_index is the standard graph connectivity format:

        [
            [src_0, src_1, src_2, ...],
            [dst_0, dst_1, dst_2, ...]
        ]

    So if edge_index = [[0, 0, 1], [1, 2, 2]],
    that means:
        0 -> 1
        0 -> 2
        1 -> 2

    WHY THIS FORMAT?
    ----------------
    This is common in graph ML libraries like PyTorch Geometric.
    """
    # Build numeric feature row for every node
    node_features = [
        node_to_feature_vector(node, num_colors=num_colors)
        for node in graph["nodes"]
    ]

    # Build numeric feature row for every edge
    edge_features = [
        edge_to_feature_vector(edge)
        for edge in graph["edges"]
    ]

    # Build edge connectivity array
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
# MAIN BUILDER: GRID -> GRAPH
# ------------------------------------------------------------

def grid_to_graph(
    grid,
    univalued=True,
    diagonal=False,
    without_bg=True,
    fully_connected=True,
    num_colors=10,
):
    """
    Main builder pipeline: convert an ARC grid into a graph.

    PARAMETERS
    ----------
    grid            : ARC grid
    univalued       : object extraction color rule
    diagonal        : 4-neighbor or 8-neighbor connectivity
    without_bg      : ignore most-common color as background
    fully_connected : whether every node connects to every other
    num_colors      : maximum color vocabulary size

    RETURNS
    -------
    graph dictionary containing:
        - nodes
        - edges
        - node_features
        - edge_features
        - edge_index

    PIPELINE STEPS
    --------------
    1. Extract raw objects from the grid
    2. Convert each object into a node
    3. Build edges between nodes
    4. Build numeric features for ML usage

    FULLY CONNECTED vs SPARSE
    -------------------------
    fully_connected=True:
        Every node connects to every other node.
        Good when you want the model to decide which edges matter.

    fully_connected=False:
        Only connect nodes with obvious spatial relations such as:
          - touching
          - same row
          - same column
          - overlapping bounding boxes

        This yields a smaller, sparser graph.
    """
    # --------------------------------------------------------
    # STEP 1: EXTRACT OBJECTS
    # --------------------------------------------------------
    raw_objs = my_objects(
        grid,
        univalued=univalued,
        diagonal=diagonal,
        without_bg=without_bg,
    )

    # --------------------------------------------------------
    # STEP 2: BUILD NODE DICTIONARIES
    # --------------------------------------------------------
    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

    # --------------------------------------------------------
    # STEP 3: BUILD EDGE DICTIONARIES
    # --------------------------------------------------------
    edges = []

    if fully_connected:
        # Fully connected directed graph:
        # add an edge i -> j for every i != j
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edges.append(build_edge(nodes[i], nodes[j]))
    else:
        # Sparse graph:
        # only connect pairs if there is some obvious relation
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                e1 = build_edge(nodes[i], nodes[j])
                e2 = build_edge(nodes[j], nodes[i])

                # Keep the pair if the objects are "meaningfully related"
                if e1["touching"] or e1["same_row"] or e1["same_col"] or e1["bbox_overlap"]:
                    edges.append(e1)
                    edges.append(e2)

    # --------------------------------------------------------
    # STEP 4: ASSEMBLE HUMAN-READABLE GRAPH
    # --------------------------------------------------------
    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    # --------------------------------------------------------
    # STEP 5: BUILD NUMERIC REPRESENTATION
    # --------------------------------------------------------
    numeric = graph_to_numeric(graph, num_colors=num_colors)
    graph["node_features"] = numeric["node_features"]
    graph["edge_features"] = numeric["edge_features"]
    graph["edge_index"] = numeric["edge_index"]

    return graph


# ------------------------------------------------------------
# GRAPH -> GRID RECONSTRUCTION
# ------------------------------------------------------------

def graph_to_grid(graph, height, width, bg=0):
    """
    Reconstruct a grid from the graph.

    PURPOSE
    -------
    This is a very important verification step.

    If reconstruction succeeds exactly, that means the graph
    stored enough information to recover the original colored
    cell layout.

    INPUTS
    ------
    graph  : graph dictionary
    height : grid height
    width  : grid width
    bg     : background fill color for empty cells

    RETURNS
    -------
    2D list representing the reconstructed grid

    HOW IT WORKS
    ------------
    We start with a blank background grid, then repaint every
    original cell from each node's stored cell_color_pairs.

    WHY cell_color_pairs MATTERS
    ----------------------------
    If we stored only abstract object features and discarded the
    exact cells, reconstruction would not be exact.
    """
    # Start with a blank background grid
    grid = [[bg for _ in range(width)] for _ in range(height)]

    # Paint back every original colored cell
    for node in graph["nodes"]:
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color

    return grid