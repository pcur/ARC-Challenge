# ============================================================
# HYBRID OBJECT BUILDER
# Minimal nodes + lightweight relational edges
# ============================================================

import math


# ------------------------------------------------------------
# SAME OBJECT EXTRACTION (KEEP THIS EXACT)
# ------------------------------------------------------------
def mostcolor(grid):
    counts = {}
    for row in grid:
        for val in row:
            counts[val] = counts.get(val, 0) + 1
    return max(counts, key=counts.get)


def dneighbors(cell):
    r, c = cell
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def neighbors(cell):
    r, c = cell
    return {
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1),                 (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
    }


def my_objects(grid, univalued=True, diagonal=False, without_bg=True):
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
            objs.add(frozenset(obj))

    return list(objs)


# ------------------------------------------------------------
# NODE
# ------------------------------------------------------------
def build_shape_mask(cells, bbox):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = [[0] * w for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1

    return mask


def build_node(i, obj):
    pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))
    cells = [loc for _, loc in pairs]

    colors = sorted(set(c for c, _ in pairs))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))

    # centroid ONLY for edge calc (not feature)
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
# EDGES (MINIMAL BUT POWERFUL)
# ------------------------------------------------------------
def touching(a, b):
    A = set(a["cells"])
    B = set(b["cells"])
    for r, c in A:
        if (r + 1, c) in B or (r - 1, c) in B or (r, c + 1) in B or (r, c - 1) in B:
            return 1
    return 0


def build_edge(a, b):
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
# FEATURE VECTORS
# ------------------------------------------------------------
def flatten(mask):
    return [float(x) for row in mask for x in row]


def node_vec(node, num_colors=10, max_h=10, max_w=10):
    # color
    color = [0] * num_colors
    for c in node["colors"]:
        color[c] = 1

    # shape
    m = node["shape_mask"]
    pad = [[0] * max_w for _ in range(max_h)]
    for i in range(min(len(m), max_h)):
        for j in range(min(len(m[0]), max_w)):
            pad[i][j] = m[i][j]

    return color + flatten(pad)


def edge_vec(e):
    return [
        e["dx"], e["dy"],
        e["touching"],
        e["same_row"], e["same_col"]
    ]


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def grid_to_graph(grid):
    objs = my_objects(grid)

    nodes = [build_node(i, o) for i, o in enumerate(objs)]

    edges = []
    for a in nodes:
        for b in nodes:
            if a["id"] != b["id"]:
                edges.append(build_edge(a, b))

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
    g = [[0] * w for _ in range(h)]
    for n in graph["nodes"]:
        for c, (r, c2) in n["cell_color_pairs"]:
            g[r][c2] = c
    return g