import json
import os
import matplotlib.pyplot as plt
import numpy as np

from graph_builder_minimal import grid_to_graph, graph_to_grid


# ------------------------------------------------------------
# SAME PATH STYLE AS YOUR CUSTOM RUNNER
# ------------------------------------------------------------
TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\data\training"


# ------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Convert to tuple format (like your original pipeline)
# ------------------------------------------------------------
def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


# ------------------------------------------------------------
# Print grid
# ------------------------------------------------------------
def print_grid(grid, title=""):
    print(f"\n{title}")
    for row in grid:
        print(" ".join(str(x) for x in row))


# ------------------------------------------------------------
# Print object map
# ------------------------------------------------------------
def print_object_map(graph, height, width):
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        nid = node["id"]
        for _, (r, c) in node["cell_color_pairs"]:
            grid[r][c] = nid

    print("\nObject Map (node IDs):")
    for row in grid:
        print(" ".join(f"{x:2}" for x in row))


# ------------------------------------------------------------
# Graph summary
# ------------------------------------------------------------
def print_graph_summary(graph):
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")

    if graph["node_features"]:
        print(f"Node feature dim: {len(graph['node_features'][0])}")
    else:
        print("Node feature dim: not available")

    if graph["edge_features"]:
        print(f"Edge feature dim: {len(graph['edge_features'][0])}")
    else:
        print("Edge feature dim: not available")

    if len(graph["edge_index"]) == 2:
        print(f"Edge index shape: [2, {len(graph['edge_index'][0])}]")
    else:
        print("Edge index shape: not available")


# ------------------------------------------------------------
# Build object map for plotting
# ------------------------------------------------------------
def build_object_map(graph, height, width):
    obj_map = np.full((height, width), -1, dtype=int)
    for node in graph["nodes"]:
        for _, (r, c) in node["cell_color_pairs"]:
            obj_map[r, c] = node["id"]
    return obj_map


# ------------------------------------------------------------
# Plot grids
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# MAIN TEST FUNCTION
# ------------------------------------------------------------
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
            shape_h = len(node["shape_mask"])
            shape_w = len(node["shape_mask"][0]) if shape_h > 0 else 0

            print(
                f"  Node {node['id']}: "
                f"colors={node['colors']}, "
                f"pixels={len(node['cell_color_pairs'])}, "
                f"bbox={node['bbox']}, "
                f"shape={shape_h}x{shape_w}"
            )

        print("\nSample edges:")
        if len(graph["edges"]) == 0:
            print("  (none - minimal builder)")
        else:
            for edge in graph["edges"][:5]:
                print(edge)

        print("\nNumeric graph:")
        if graph["node_features"]:
            print(f"  node_features shape = [{len(graph['node_features'])}, {len(graph['node_features'][0])}]")
        else:
            print("  node_features shape = [0, 0]")

        if graph["edge_features"]:
            print(f"  edge_features shape = [{len(graph['edge_features'])}, {len(graph['edge_features'][0])}]")
        else:
            print(f"  edge_features shape = [{len(graph['edge_features'])}, 0]")

        if len(graph["edge_index"]) == 2:
            print(f"  edge_index shape    = [2, {len(graph['edge_index'][0])}]")
        else:
            print("  edge_index shape    = not available")

        plot_grids(raw_grid, rebuilt, graph, title=f"{os.path.basename(file_path)} - Train {i}")


# ------------------------------------------------------------
# RUN (MATCHES YOUR ORIGINAL STYLE)
# ------------------------------------------------------------
if __name__ == "__main__":
    file_path = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file_path)