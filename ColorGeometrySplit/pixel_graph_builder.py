"""
pixel_graph_builder.py
======================
Builds a pixel-level PyG Data object from a raw ARC grid.

Every cell in the grid is a node — including background (color 0).
Nodes are connected by 4-connectivity (up/down/left/right only).

Node features (per pixel):
  [0:10]  color one-hot  (10 dims)
  [10]    row normalised to [0, 1]
  [11]    col normalised to [0, 1]
  Total: 12 dims

Edge features (per adjacency pair):
  [0]  same_color  (0/1)
  [1]  dx          normalised relative column offset (-1, 0, +1) / max_dim
  [2]  dy          normalised relative row offset    (-1, 0, +1) / max_dim
  Total: 3 dims

Reversal:
  pixel_graph_to_grid(graph, rows, cols) — reads color argmax from node
  features and reshapes to (rows, cols). Trivially reversible.

Dependencies:
    pip install torch torch_geometric
"""

import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────

def pixel_grid_to_graph(grid) -> Data:
    """
    Convert an ARC grid (list-of-lists or tuple-of-tuples of ints) into a
    pixel-level PyG Data object with 4-connected edges.

    Parameters
    ----------
    grid : 2D sequence of int  (values 0–9)

    Returns
    -------
    torch_geometric.data.Data with fields:
        x          : (rows*cols, 12)   node features
        edge_index : (2, E)            COO edge index  E ≈ 2*(rows*(cols-1) + cols*(rows-1))
        edge_attr  : (E, 3)            edge features
        grid_rows  : int               stored for reversal
        grid_cols  : int               stored for reversal
    """
    rows   = len(grid)
    cols   = len(grid[0])
    N      = rows * cols
    max_dim = max(rows, cols, 1)

    # ── node features ─────────────────────────────────────────────────────────
    color_onehot = torch.zeros(N, 10)
    row_norm     = torch.zeros(N)
    col_norm     = torch.zeros(N)

    for r in range(rows):
        for c in range(cols):
            idx              = r * cols + c
            color            = int(grid[r][c])
            color_onehot[idx, color] = 1.0
            row_norm[idx]    = r / (rows - 1) if rows > 1 else 0.0
            col_norm[idx]    = c / (cols - 1) if cols > 1 else 0.0

    x = torch.cat([color_onehot, row_norm.unsqueeze(1), col_norm.unsqueeze(1)], dim=1)

    # ── edges: 4-connectivity ─────────────────────────────────────────────────
    src_list, dst_list = [], []
    dx_list,  dy_list  = [], []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    src_list.append(idx)
                    dst_list.append(nidx)
                    dx_list.append(dc / max_dim)
                    dy_list.append(dr / max_dim)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # edge features: same_color, dx, dy
    src_colors = color_onehot[src_list].argmax(dim=-1)
    dst_colors = color_onehot[dst_list].argmax(dim=-1)
    same_color = (src_colors == dst_colors).float().unsqueeze(1)
    dx_t       = torch.tensor(dx_list).unsqueeze(1)
    dy_t       = torch.tensor(dy_list).unsqueeze(1)

    edge_attr = torch.cat([same_color, dx_t, dy_t], dim=1)

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        grid_rows  = rows,
        grid_cols  = cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVERSE
# ─────────────────────────────────────────────────────────────────────────────

def pixel_graph_to_grid(x_or_logits, rows: int, cols: int):
    """
    Reconstruct an ARC grid from pixel node features or color logits.

    Parameters
    ----------
    x_or_logits : (N, 10) or (N, >=10) tensor
                  If color is stored as one-hot or logits in dims [:10],
                  argmax gives the predicted color per pixel.
    rows, cols  : grid dimensions

    Returns
    -------
    list[list[int]]  — standard ARC grid format
    """
    colors = x_or_logits[:, :10].argmax(dim=-1).tolist()
    grid   = []
    for r in range(rows):
        row = colors[r * cols : (r + 1) * cols]
        grid.append(row)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_grid = [
        [0, 0, 3, 0],
        [0, 3, 3, 0],
        [0, 0, 3, 0],
    ]

    graph = pixel_grid_to_graph(test_grid)
    print(f"Nodes     : {graph.x.size(0)}  (expected {3*4}=12)")
    print(f"Node feats: {graph.x.size(1)}  (expected 12)")
    print(f"Edges     : {graph.edge_index.size(1)}  (expected {2*(3*3 + 2*4)}=34)")
    print(f"Edge feats: {graph.edge_attr.size(1)}  (expected 3)")
    print(f"Grid dims : {graph.grid_rows}x{graph.grid_cols}")

    rebuilt = pixel_graph_to_grid(graph.x, graph.grid_rows, graph.grid_cols)
    print(f"\nOriginal : {test_grid}")
    print(f"Rebuilt  : {rebuilt}")
    print(f"Match    : {rebuilt == test_grid}")
