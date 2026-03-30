# graph_builder.py
#used with run_builder.py Rev1, 2, 3, and 4

import math
from typing import List, Tuple, Dict, Any
from dsl import objects

Grid = List[List[int]]
Cell = Tuple[int, int]


# -------------------------
# Node + Edge (simple dict version)
# -------------------------

def build_node(node_id, obj):
    cells = [loc for _, loc in obj]
    colors = list(set([c for c, _ in obj]))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    area = len(cells)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    centroid = (sum(rows) / area, sum(cols) / area)

    return {
        "id": node_id,
        "colors": colors,
        "cells": cells,   # critical for reverse mapping
        "area": area,
        "centroid": centroid,
        "bbox": (min_r, min_c, max_r, max_c),
        "width": max_c - min_c + 1,
        "height": max_r - min_r + 1,
    }


def touching(cells_a, cells_b):
    bset = set(cells_b)
    for r, c in cells_a:
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if (nr, nc) in bset:
                return True
    return False


def build_edge(a, b):
    ar, ac = a["centroid"]
    br, bc = b["centroid"]

    dx = bc - ac
    dy = br - ar

    return {
        "src": a["id"],
        "dst": b["id"],
        "dx": dx,
        "dy": dy,
        "dist": math.sqrt(dx*dx + dy*dy),
        "same_color": int(bool(set(a["colors"]) & set(b["colors"]))),
        "touching": int(touching(a["cells"], b["cells"])),
    }


# -------------------------
# MAIN BUILDER
# -------------------------

def grid_to_graph(grid: Grid) -> Dict[str, Any]:

    raw_objs = objects(
        grid,
        univalued=True,
        diagonal=False,
        without_bg=True
    )

    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

    edges = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                edges.append(build_edge(nodes[i], nodes[j]))

    return {
        "nodes": nodes,
        "edges": edges
    }


# -------------------------
# REVERSE BUILD
# -------------------------

def graph_to_grid(graph, height, width, bg=0):
    grid = [[bg for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        color = node["colors"][0]
        for r, c in node["cells"]:
            grid[r][c] = color

    return grid