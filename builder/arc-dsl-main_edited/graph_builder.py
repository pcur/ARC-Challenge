import math
from dsl import objects


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def build_shape_mask(cells, bbox):
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    mask = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in cells:
        mask[r - min_r][c - min_c] = 1
    return mask


def touching(cells_a, cells_b):
    bset = set(cells_b)
    for r, c in cells_a:
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) in bset:
                return True
    return False


def bbox_overlap(bbox_a, bbox_b):
    min_ra, min_ca, max_ra, max_ca = bbox_a
    min_rb, min_cb, max_rb, max_cb = bbox_b

    row_overlap = not (max_ra < min_rb or max_rb < min_ra)
    col_overlap = not (max_ca < min_cb or max_cb < min_ca)
    return row_overlap and col_overlap


def build_node(node_id, obj):
    # obj is a frozenset of (color, (row, col))
    cell_color_pairs = sorted(list(obj), key=lambda x: (x[1][0], x[1][1]))
    cells = [loc for _, loc in cell_color_pairs]
    colors = sorted(list(set(color for color, _ in cell_color_pairs)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    area = len(cells)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    width = max_c - min_c + 1
    height = max_r - min_r + 1

    centroid = (sum(rows) / area, sum(cols) / area)
    bbox = (min_r, min_c, max_r, max_c)

    bbox_area = width * height
    density = safe_div(area, bbox_area)
    aspect_ratio = safe_div(width, height)

    shape_mask = build_shape_mask(cells, bbox)

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


def build_edge(a, b):
    ar, ac = a["centroid"]
    br, bc = b["centroid"]

    dx = bc - ac
    dy = br - ar

    area_a = a["area"]
    area_b = b["area"]

    return {
        "src": a["id"],
        "dst": b["id"],
        "dx": dx,
        "dy": dy,
        "manhattan": abs(dx) + abs(dy),
        "dist": math.sqrt(dx * dx + dy * dy),
        "same_color": int(bool(set(a["colors"]) & set(b["colors"]))),
        "touching": int(touching(a["cells"], b["cells"])),
        "bbox_overlap": int(bbox_overlap(a["bbox"], b["bbox"])),
        "same_row": int(round(ar) == round(br)),
        "same_col": int(round(ac) == round(bc)),
        "same_area": int(area_a == area_b),
        "area_ratio_ab": safe_div(area_a, area_b),
        "area_ratio_ba": safe_div(area_b, area_a),
    }


def node_to_feature_vector(node, num_colors=10):
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


def graph_to_numeric(graph, num_colors=10):
    node_features = [node_to_feature_vector(node, num_colors=num_colors) for node in graph["nodes"]]
    edge_features = [edge_to_feature_vector(edge) for edge in graph["edges"]]

    edge_index = [
        [edge["src"] for edge in graph["edges"]],
        [edge["dst"] for edge in graph["edges"]],
    ]

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
    }


def grid_to_graph(
    grid,
    univalued=True,
    diagonal=False,
    without_bg=True,
    fully_connected=True,
    num_colors=10,
):
    raw_objs = objects(
        grid,
        univalued=univalued,
        diagonal=diagonal,
        without_bg=without_bg,
    )

    nodes = [build_node(i, obj) for i, obj in enumerate(raw_objs)]

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

    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    numeric = graph_to_numeric(graph, num_colors=num_colors)
    graph["node_features"] = numeric["node_features"]
    graph["edge_features"] = numeric["edge_features"]
    graph["edge_index"] = numeric["edge_index"]

    return graph


def graph_to_grid(graph, height, width, bg=0):
    grid = [[bg for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        # Exact reverse mapping using stored per-cell color data
        for color, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = color

    return grid