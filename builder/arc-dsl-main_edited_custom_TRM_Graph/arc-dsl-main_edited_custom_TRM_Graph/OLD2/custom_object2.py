import math


# ------------------------------------------------------------
# BASIC HELPERS FOR CUSTOM OBJECT EXTRACTION
# ------------------------------------------------------------

def mostcolor(grid):
    """
    Return the most common color in the grid.
    This is used as the background color when without_bg=True.
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
    Custom replacement for the DSL objects(...) function.

    Purpose:
        Extract connected objects from the grid.

    Inputs:
        grid        : tuple-of-tuples or list-of-lists ARC grid
        univalued   : if True, object must stay same color as seed cell
                      if False, any non-background connected cells can join
        diagonal    : if True, use 8-connected neighborhood
                      if False, use 4-connected neighborhood
        without_bg  : if True, ignore the most common color as background

    Returns:
        LIST of objects sorted by (min_row, min_col) of each object's
        top-left cell — deterministic and stable across runs.
        Each object is a frozenset of: (color, (row, col))

    FIX v2:
        Previously returned a frozenset of frozensets, whose iteration
        order is arbitrary in Python. This caused node IDs to be assigned
        differently on every run, making training targets inconsistent.
        Now returns a sorted list so node 0 is always the top-left-most
        object, node 1 is the next, etc.
    """
    bg = mostcolor(grid) if without_bg else None
    objs = []
    occupied = set()

    h, w = len(grid), len(grid[0])

    # FIX v2: iterate cells in a fixed row-major order so the BFS seeds
    # are always visited in the same sequence. Previously used a set
    # comprehension which has no guaranteed order.
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    neigh_fn = neighbors if diagonal else dneighbors

    for loc in all_cells:
        # Skip if already assigned to an object
        if loc in occupied:
            continue

        val = grid[loc[0]][loc[1]]

        # Skip background if requested
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

                # If univalued=True:
                #   only grow through cells of the same color as the seed
                #
                # If univalued=False:
                #   grow through any connected non-background cells
                condition = (val == v) if univalued else (v != bg)

                if condition:
                    obj.add((v, cand))
                    occupied.add(cand)

                    for nr, nc in neigh_fn(cand):
                        if 0 <= nr < h and 0 <= nc < w:
                            neighborhood.add((nr, nc))

            cands = neighborhood - occupied

        if obj:
            objs.append(frozenset(obj))

    # FIX v2: sort by the top-left cell of each object (min row, then min col)
    # so that node IDs are deterministic and stable across runs.
    def obj_sort_key(obj):
        cells = [loc for _, loc in obj]
        return (min(r for r, _ in cells), min(c for _, c in cells))

    objs.sort(key=obj_sort_key)

    return objs


# ------------------------------------------------------------
# GENERAL MATH / GEOMETRY HELPERS
# ------------------------------------------------------------

def safe_div(a, b):
    """
    Divide a by b safely.
    If b == 0, return 0.0 instead of crashing.
    """
    return a / b if b != 0 else 0.0


def build_shape_mask(cells, bbox):
    """
    Build a local binary mask for the object inside its bounding box.

    Example:
        If cells occupy a 2x3 box, this returns a 2D list marking
        which cells are filled (1) or empty (0) inside that box.
    """
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


def touching(cells_a, cells_b):
    """
    Check whether two objects touch using 4-connectivity.

    Returns:
        True  if any cell in A is directly adjacent to any cell in B
        False otherwise
    """
    bset = set(cells_b)

    for r, c in cells_a:
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) in bset:
                return True

    return False


def bbox_overlap(bbox_a, bbox_b):
    """
    Check whether two bounding boxes overlap.

    bbox format:
        (min_row, min_col, max_row, max_col)
    """
    min_ra, min_ca, max_ra, max_ca = bbox_a
    min_rb, min_cb, max_rb, max_cb = bbox_b

    row_overlap = not (max_ra < min_rb or max_rb < min_ra)
    col_overlap = not (max_ca < min_cb or max_cb < min_ca)

    return row_overlap and col_overlap


# ------------------------------------------------------------
# NODE CONSTRUCTION
# ------------------------------------------------------------

def build_node(node_id, obj):
    """
    Convert one extracted object into a structured node dictionary.

    obj:
        frozenset of (color, (row, col))
    """
    # Sort for stable ordering/debugging
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))

    # Extract coordinates only
    cells = [loc for _, loc in cell_color_pairs]

    # Unique colors in this object
    colors = sorted(list(set(color for color, _ in cell_color_pairs)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    # Area = number of pixels in object
    area = len(cells)

    # Bounding box
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Width/height of bbox
    width = max_c - min_c + 1
    height = max_r - min_r + 1

    # Object centroid in grid coordinates
    centroid = (sum(rows) / area, sum(cols) / area)

    bbox = (min_r, min_c, max_r, max_c)

    # Density = fraction of bbox occupied by object pixels
    bbox_area = width * height
    density = safe_div(area, bbox_area)

    # Aspect ratio = width / height
    aspect_ratio = safe_div(width, height)

    # Local binary shape mask for debugging / future feature work
    shape_mask = build_shape_mask(cells, bbox)

    return {
        "id": node_id,
        "colors": colors,
        "cell_color_pairs": cell_color_pairs,   # needed for exact reconstruction
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

    Edge features describe the relationship between two objects.
    """
    ar, ac = a["centroid"]
    br, bc = b["centroid"]

    # Relative position of B with respect to A
    dx = bc - ac
    dy = br - ar

    area_a = a["area"]
    area_b = b["area"]

    return {
        "src": a["id"],
        "dst": b["id"],

        # Relative position
        "dx": dx,
        "dy": dy,

        # Distances
        "manhattan": abs(dx) + abs(dy),
        "dist": math.sqrt(dx * dx + dy * dy),

        # Appearance / relation checks
        "same_color": int(bool(set(a["colors"]) & set(b["colors"]))),
        "touching": int(touching(a["cells"], b["cells"])),
        "bbox_overlap": int(bbox_overlap(a["bbox"], b["bbox"])),

        # Alignment
        "same_row": int(round(ar) == round(br)),
        "same_col": int(round(ac) == round(bc)),

        # Size relations
        "same_area": int(area_a == area_b),
        "area_ratio_ab": safe_div(area_a, area_b),
        "area_ratio_ba": safe_div(area_b, area_a),
    }


# ------------------------------------------------------------
# NODE FEATURE VECTOR
# ------------------------------------------------------------

def node_to_feature_vector(node, num_colors=10):
    """
    Convert a node dictionary into a numeric feature vector.

    Output dimension:
        22
    """
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
# EDGE FEATURE VECTOR
# ------------------------------------------------------------

def edge_to_feature_vector(edge):
    """
    Convert an edge dictionary into a numeric feature vector.

    Output dimension:
        12
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

    Returns:
        node_features : [N, 22]
        edge_features : [E, 12]
        edge_index    : [2, E]
    """
    node_features = [node_to_feature_vector(node, num_colors=num_colors)
                     for node in graph["nodes"]]

    edge_features = [edge_to_feature_vector(edge)
                     for edge in graph["edges"]]

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
    Main builder pipeline.

    Steps:
        1. Extract raw objects from the grid using our custom object extractor
        2. Convert each object to a node
        3. Build edges between nodes
        4. Convert graph into numeric model-ready tensors
    """
    # Step 1: custom object extraction
    # FIX v2: my_objects now returns a sorted list (deterministic ordering)
    raw_objs = my_objects(
        grid,
        univalued=univalued,
        diagonal=diagonal,
        without_bg=without_bg,
    )

    # Step 2: build nodes
    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

    # Step 3: build edges
    edges = []

    if fully_connected:
        # Every node connects to every other node (directed)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edges.append(build_edge(nodes[i], nodes[j]))
    else:
        # Optional sparse graph: only connect nodes if they have some
        # meaningful spatial relation
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                e1 = build_edge(nodes[i], nodes[j])
                e2 = build_edge(nodes[j], nodes[i])

                if e1["touching"] or e1["same_row"] or e1["same_col"] or e1["bbox_overlap"]:
                    edges.append(e1)
                    edges.append(e2)

    # Step 4: assemble human-readable graph
    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    # Step 5: add numeric representation
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
    Reconstruct the original grid from the graph.

    This verifies that the graph representation is lossless.
    """
    grid = [[bg for _ in range(width)] for _ in range(height)]

    # Paint back every original colored cell
    for node in graph["nodes"]:
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color

    return grid
