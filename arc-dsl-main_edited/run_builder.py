import json
import os
import matplotlib.pyplot as plt
import numpy as np

from graph_builder import grid_to_graph, graph_to_grid

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


def print_grid(grid, title=""):
    print(f"\n{title}")
    for row in grid:
        print(" ".join(str(x) for x in row))


def print_object_map(graph, height, width):
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        nid = node["id"]
        for r, c in node["cells"]:
            grid[r][c] = nid

    print("\nObject Map (node IDs):")
    for row in grid:
        print(" ".join(f"{x:2}" for x in row))


def print_graph_summary(graph):
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")

    if "node_features" in graph and len(graph["node_features"]) > 0:
        print(f"Node feature dim: {len(graph['node_features'][0])}")
    else:
        print("Node feature dim: not available")

    if "edge_features" in graph and len(graph["edge_features"]) > 0:
        print(f"Edge feature dim: {len(graph['edge_features'][0])}")
    else:
        print("Edge feature dim: not available")

    if "edge_index" in graph and len(graph["edge_index"]) == 2:
        print(f"Edge index shape: [2, {len(graph['edge_index'][0])}]")
    else:
        print("Edge index shape: not available")


def build_object_map(graph, height, width):
    obj_map = np.full((height, width), -1, dtype=int)
    for node in graph["nodes"]:
        for r, c in node["cells"]:
            obj_map[r, c] = node["id"]
    return obj_map


def plot_grids(input_grid, rebuilt_grid, graph, title=""):
    input_arr = np.array(input_grid)
    rebuilt_arr = np.array(rebuilt_grid)

    h = len(input_grid)
    w = len(input_grid[0])
    obj_map = build_object_map(graph, h, w)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(input_arr, cmap="tab10", vmin=0, vmax=9)
    axs[0].set_title("Input")

    axs[1].imshow(rebuilt_arr, cmap="tab10", vmin=0, vmax=9)
    axs[1].set_title("Rebuilt")

    axs[2].imshow(obj_map, cmap="tab20")
    axs[2].set_title("Objects")

    for ax in axs:
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="lightgray", linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def test_file(file_path):
    task = load_json(file_path)

    print(f"\nFILE: {os.path.basename(file_path)}")

    for i, pair in enumerate(task["train"]):
        raw_grid = pair["input"]
        grid = to_dsl_grid(raw_grid)

        print_grid(raw_grid, f"Train {i} INPUT")

        graph = grid_to_graph(grid)
        rebuilt = graph_to_grid(graph, len(grid), len(grid[0]))

        print_grid(rebuilt, f"Train {i} REBUILT")

        ok = (rebuilt == raw_grid)

        print(f"\nTrain {i}: nodes={len(graph['nodes'])}, rebuild={ok}")
        print_graph_summary(graph)
        print_object_map(graph, len(grid), len(grid[0]))

        print("\nNodes:")
        for node in graph["nodes"]:
            print(
                f"  Node {node['id']}: "
                f"colors={node['colors']}, "
                f"area={node['area']}, "
                f"centroid={node['centroid']}, "
                f"bbox={node['bbox']}, "
                f"width={node['width']}, "
                f"height={node['height']}, "
                f"density={node['density']:.3f}, "
                f"aspect_ratio={node['aspect_ratio']:.3f}, "
                f"is_single_pixel={node['is_single_pixel']}"
            )

        print("\nSample edges:")
        for edge in graph["edges"][:5]:
            print(
                f"  {edge['src']}->{edge['dst']} | "
                f"dx={edge['dx']:.2f}, dy={edge['dy']:.2f}, "
                f"manhattan={edge['manhattan']:.2f}, dist={edge['dist']:.2f}, "
                f"touching={edge['touching']}, same_color={edge['same_color']}, "
                f"same_row={edge['same_row']}, same_col={edge['same_col']}, "
                f"same_area={edge['same_area']}, bbox_overlap={edge['bbox_overlap']}"
            )

        print("\nNumeric graph:")
        print(f"  node_features shape = [{len(graph['node_features'])}, {len(graph['node_features'][0])}]")
        print(f"  edge_features shape = [{len(graph['edge_features'])}, {len(graph['edge_features'][0])}]")
        print(f"  edge_index shape    = [2, {len(graph['edge_index'][0])}]")

        plot_grids(raw_grid, rebuilt, graph, title=f"{os.path.basename(file_path)} - Train {i}")


if __name__ == "__main__":
    file_path = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file_path)