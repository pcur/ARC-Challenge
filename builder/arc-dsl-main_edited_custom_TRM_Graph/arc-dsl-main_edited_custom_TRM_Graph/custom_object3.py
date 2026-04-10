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
    """4-connected neighbors: up, down, left, right"""
    r, c = cell
    return {(r-1,c),(r+1,c),(r,c-1),(r,c+1)}


def neighbors(cell):
    """8-connected neighbors: includes diagonals"""
    r, c = cell
    return {
        (r-1,c-1),(r-1,c),(r-1,c+1),
        (r,c-1),          (r,c+1),
        (r+1,c-1),(r+1,c),(r+1,c+1),
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
    """
    Extract connected objects from the grid.

    Returns a SORTED LIST of frozensets (deterministic — v2 fix).
    Each frozenset contains (color, (row, col)) tuples.
    Sorted by (min_row, min_col) of each object's top-left cell.
    """
    bg = mostcolor(grid) if without_bg else None
    objs = []
    occupied = set()

    h, w = len(grid), len(grid[0])
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
            neighborhood = set()
            for cand in cands:
                if cand in occupied:
                    continue
                v = grid[cand[0]][cand[1]]
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

    def obj_sort_key(obj):
        cells = [loc for _, loc in obj]
        return (min(r for r,_ in cells), min(c for _,c in cells))

    objs.sort(key=obj_sort_key)
    return objs


# ------------------------------------------------------------
# GENERAL MATH / GEOMETRY HELPERS
# ------------------------------------------------------------

def safe_div(a, b):
    return a / b if b != 0 else 0.0


def build_shape_mask(cells, bbox):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    mask = [[0]*w for _ in range(h)]
    for r, c in cells:
        mask[r-min_r][c-min_c] = 1
    return mask


def touching(cells_a, cells_b):
    bset = set(cells_b)
    for r, c in cells_a:
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (nr,nc) in bset:
                return True
    return False


def bbox_overlap(bbox_a, bbox_b):
    min_ra,min_ca,max_ra,max_ca = bbox_a
    min_rb,min_cb,max_rb,max_cb = bbox_b
    return (not (max_ra<min_rb or max_rb<min_ra)) and \
           (not (max_ca<min_cb or max_cb<min_ca))


# ------------------------------------------------------------
# NODE CONSTRUCTION
# ------------------------------------------------------------

def build_node(node_id, obj, max_cells_per_node=40):
    """
    Convert one extracted object into a structured node dictionary.

    v3 NEW: adds 'cell_coords' and 'cell_mask' fields.

        cell_coords : list of up to max_cells_per_node [row, col] pairs,
                      padded with [0, 0] if the object has fewer cells.
        cell_mask   : list of 1.0/0.0 floats — 1.0 for real cells,
                      0.0 for padding. Same length as cell_coords.
        cell_colors : list of color ints, one per real cell (padded with 0).
                      Needed so the decoder can reconstruct per-cell color
                      for multi-color objects.

    These three fields give the decoder everything it needs to paint the
    object back onto a blank grid without any external bookkeeping.

    Objects with more than max_cells_per_node cells are TRUNCATED — the
    first max_cells_per_node cells (in row-major order) are kept.
    The truncation flag is stored in 'truncated' for debugging.
    """
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))
    cells  = [loc   for _, loc   in cell_color_pairs]
    colors = sorted(list(set(color for color,_ in cell_color_pairs)))

    rows = [r for r,_ in cells]
    cols = [c for _,c in cells]
    area = len(cells)

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    width        = max_c - min_c + 1
    height       = max_r - min_r + 1
    centroid     = (sum(rows)/area, sum(cols)/area)
    bbox         = (min_r, min_c, max_r, max_c)
    density      = safe_div(area, width*height)
    aspect_ratio = safe_div(width, height)
    shape_mask   = build_shape_mask(cells, bbox)

    # ── v3: cell coordinate list ─────────────────────────────────────────
    truncated = area > max_cells_per_node

    # Keep only up to max_cells_per_node cells (already in row-major order)
    kept_pairs = cell_color_pairs[:max_cells_per_node]
    n_kept     = len(kept_pairs)

    cell_coords = [[r, c]    for _, (r,c) in kept_pairs]
    cell_colors_list = [col  for col, _   in kept_pairs]

    # Pad to max_cells_per_node
    pad = max_cells_per_node - n_kept
    cell_coords      += [[0, 0]] * pad
    cell_colors_list += [0]      * pad

    # Mask: 1.0 for real cells, 0.0 for padding
    cell_mask = [1.0] * n_kept + [0.0] * pad

    return {
        "id"               : node_id,
        "colors"           : colors,
        "cell_color_pairs" : cell_color_pairs,   # full list — for graph_to_grid
        "cells"            : cells,
        "area"             : area,
        "centroid"         : centroid,
        "bbox"             : bbox,
        "width"            : width,
        "height"           : height,
        "density"          : density,
        "aspect_ratio"     : aspect_ratio,
        "is_single_pixel"  : int(area == 1),
        "shape_mask"       : shape_mask,
        # ── v3 additions ──────────────────────────────────────────────────
        "cell_coords"      : cell_coords,        # [max_cells, 2]  row/col
        "cell_colors_list" : cell_colors_list,   # [max_cells]     color index
        "cell_mask"        : cell_mask,          # [max_cells]     1=real,0=pad
        "truncated"        : truncated,
    }


# ------------------------------------------------------------
# EDGE CONSTRUCTION
# ------------------------------------------------------------

def build_edge(a, b):
    ar, ac = a["centroid"]
    br, bc = b["centroid"]
    dx = bc - ac
    dy = br - ar
    area_a, area_b = a["area"], b["area"]
    return {
        "src"          : a["id"],
        "dst"          : b["id"],
        "dx"           : dx,
        "dy"           : dy,
        "manhattan"    : abs(dx)+abs(dy),
        "dist"         : math.sqrt(dx*dx+dy*dy),
        "same_color"   : int(bool(set(a["colors"]) & set(b["colors"]))),
        "touching"     : int(touching(a["cells"], b["cells"])),
        "bbox_overlap" : int(bbox_overlap(a["bbox"], b["bbox"])),
        "same_row"     : int(round(ar)==round(br)),
        "same_col"     : int(round(ac)==round(bc)),
        "same_area"    : int(area_a==area_b),
        "area_ratio_ab": safe_div(area_a, area_b),
        "area_ratio_ba": safe_div(area_b, area_a),
    }


# ------------------------------------------------------------
# NODE FEATURE VECTOR
# ------------------------------------------------------------

def node_to_feature_vector(node, num_colors=10, grid_norm=30.0, area_norm=900.0):
    """
    Numeric feature vector for the GAT encoder. Dimension: 22.

    FIX v3b: All spatial values are now normalised so they sit in roughly
    [0, 1]. Without this, raw pixel coordinates (up to 30) and areas
    (up to 900) caused large activations that contributed to NaN explosions
    during training.

    Normalisation constants:
        grid_norm = 30.0   — max ARC grid dimension (rows or cols)
        area_norm = 900.0  — max ARC grid area (30 * 30)
    """
    color_one_hot = [0.0] * num_colors
    for color in node["colors"]:
        if 0 <= color < num_colors:
            color_one_hot[color] = 1.0

    min_r, min_c, max_r, max_c = node["bbox"]
    return color_one_hot + [
        float(node["area"])          / area_norm,   # [0, 1]
        float(node["centroid"][0])   / grid_norm,   # [0, 1]
        float(node["centroid"][1])   / grid_norm,   # [0, 1]
        float(min_r)                 / grid_norm,   # [0, 1]
        float(min_c)                 / grid_norm,   # [0, 1]
        float(max_r)                 / grid_norm,   # [0, 1]
        float(max_c)                 / grid_norm,   # [0, 1]
        float(node["width"])         / grid_norm,   # [0, 1]
        float(node["height"])        / grid_norm,   # [0, 1]
        float(node["density"]),                     # already [0, 1]
        float(node["aspect_ratio"])  / 30.0,        # cap at ~1 for square objects
        float(node["is_single_pixel"]),             # 0 or 1
    ]


# ------------------------------------------------------------
# EDGE FEATURE VECTOR
# ------------------------------------------------------------

def edge_to_feature_vector(edge, grid_norm=30.0):
    """
    Numeric edge feature vector. Dimension: 12.

    FIX v3b: dx, dy, manhattan and dist are divided by grid_norm (30.0)
    so they sit in roughly [-1, 1] / [0, 1] rather than raw pixel units.
    Without this, MSE on these values during reconstruction was the primary
    cause of loss exploding to NaN — a single large grid could produce
    dist values of 40+ which squared to 1600+ in the MSE.

    area_ratio values are clamped to [0, 10] before normalising to prevent
    extreme ratios (tiny object vs huge object) from causing instability.
    """
    ar_ab = min(float(edge["area_ratio_ab"]), 10.0) / 10.0
    ar_ba = min(float(edge["area_ratio_ba"]), 10.0) / 10.0
    return [
        float(edge["dx"])        / grid_norm,   # [-1, 1]
        float(edge["dy"])        / grid_norm,   # [-1, 1]
        float(edge["manhattan"]) / grid_norm,   # [0, ~2]
        float(edge["dist"])      / grid_norm,   # [0, ~1.4]
        float(edge["same_color"]),              # 0 or 1
        float(edge["touching"]),                # 0 or 1
        float(edge["bbox_overlap"]),            # 0 or 1
        float(edge["same_row"]),                # 0 or 1
        float(edge["same_col"]),                # 0 or 1
        float(edge["same_area"]),               # 0 or 1
        ar_ab,                                  # [0, 1]
        ar_ba,                                  # [0, 1]
    ]


# ------------------------------------------------------------
# GRAPH -> NUMERIC FORMAT
# ------------------------------------------------------------

def graph_to_numeric(graph, num_colors=10):
    node_features = [node_to_feature_vector(n, num_colors) for n in graph["nodes"]]
    edge_features = [edge_to_feature_vector(e) for e in graph["edges"]]
    edge_index    = [
        [e["src"] for e in graph["edges"]],
        [e["dst"] for e in graph["edges"]],
    ]
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index"   : edge_index,
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
    max_cells_per_node=40,    # v3: passed through to build_node
):
    """
    Main builder pipeline.

    v3: build_node now receives max_cells_per_node and stores
    cell_coords / cell_mask / cell_colors_list on every node.
    """
    raw_objs = my_objects(grid, univalued=univalued,
                          diagonal=diagonal, without_bg=without_bg)

    # v3: pass max_cells_per_node to build_node
    nodes = [build_node(i, obj, max_cells_per_node=max_cells_per_node)
             for i, obj in enumerate(raw_objs)]

    edges = []
    if fully_connected:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edges.append(build_edge(nodes[i], nodes[j]))
    else:
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                e1 = build_edge(nodes[i], nodes[j])
                e2 = build_edge(nodes[j], nodes[i])
                if e1["touching"] or e1["same_row"] or e1["same_col"] or e1["bbox_overlap"]:
                    edges.append(e1)
                    edges.append(e2)

    graph = {"nodes": nodes, "edges": edges}

    numeric = graph_to_numeric(graph, num_colors=num_colors)
    graph["node_features"] = numeric["node_features"]
    graph["edge_features"] = numeric["edge_features"]
    graph["edge_index"]    = numeric["edge_index"]

    return graph


# ------------------------------------------------------------
# GRAPH -> GRID RECONSTRUCTION
# ------------------------------------------------------------

def graph_to_grid(graph, height, width, bg=0):
    """
    Reconstruct the original grid from the graph.
    Uses cell_color_pairs (full, untruncated) for lossless reconstruction
    when the graph was built from the original grid.
    """
    grid = [[bg]*width for _ in range(height)]
    for node in graph["nodes"]:
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color
    return grid


def graph_to_grid_from_predictions(
    pred_cell_coords,    # [max_nodes, max_cells, 2]   float — predicted row/col
    pred_cell_colors,    # [max_nodes, max_cells]       long  — predicted color index
    pred_cell_mask,      # [max_nodes, max_cells]       float — >0.5 means real cell
    pred_existence,      # [max_nodes]                  float — >0.5 means real node
    height, width, bg=0,
):
    """
    Reconstruct a grid purely from decoder predictions.

    This is the reconstruction path used at inference time when we only
    have the decoder outputs, not the original graph.

    Steps:
      1. For each node slot where pred_existence > 0.5 (real node):
      2.   For each cell slot where pred_cell_mask > 0.5 (real cell):
      3.     Round pred_cell_coords to nearest integer row/col
      4.     Clamp to grid bounds
      5.     Paint pred_cell_colors at that location
    """
    grid = [[bg]*width for _ in range(height)]

    max_nodes = len(pred_existence)
    for n in range(max_nodes):
        if pred_existence[n] <= 0.5:
            continue
        max_cells = len(pred_cell_mask[n])
        for k in range(max_cells):
            if pred_cell_mask[n][k] <= 0.5:
                continue
            r = int(round(float(pred_cell_coords[n][k][0])))
            c = int(round(float(pred_cell_coords[n][k][1])))
            r = max(0, min(height-1, r))
            c = max(0, min(width-1,  c))
            color = int(pred_cell_colors[n][k])
            grid[r][c] = color

    return grid
