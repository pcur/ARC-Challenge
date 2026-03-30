import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Import YOUR custom builder (this replaces DSL objects())
# grid_to_graph: converts ARC grid → graph representation
# graph_to_grid: reconstructs grid from graph
from custom_object import grid_to_graph, graph_to_grid

# Path to ARC training tasks
TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"


# ============================================================
# BASIC UTILITIES
# ============================================================

def load_json(path):
    """
    Load an ARC task JSON file.
    Each file contains:
        - "train": list of input/output pairs
        - "test": list of test inputs
    """
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid):
    """
    Convert Python list grid → immutable tuple format.

    WHY:
    - DSL functions often expect tuples (hashable / immutable)
    - Prevents accidental modification
    """
    return tuple(tuple(row) for row in grid)


def print_grid(grid, title=""):
    """
    Pretty-print a grid to terminal.
    """
    print(f"\n{title}")
    for row in grid:
        print(" ".join(str(x) for x in row))


# ============================================================
# GRAPH DEBUG / VISUALIZATION HELPERS
# ============================================================

def print_object_map(graph, height, width):
    """
    Build and print a grid where each pixel = object ID.

    PURPOSE:
    - Helps visualize how your builder grouped pixels into objects
    - Critical for debugging segmentation errors
    """
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    for node in graph["nodes"]:
        nid = node["id"]
        for r, c in node["cells"]:
            grid[r][c] = nid

    print("\nObject Map (node IDs):")
    for row in grid:
        print(" ".join(f"{x:2}" for x in row))


def print_graph_summary(graph):
    """
    Print high-level graph statistics.

    This helps verify:
    - object count (nodes)
    - relationship count (edges)
    - feature dimensions (important for ML models)
    """
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")

    # Node feature dimension
    if "node_features" in graph and len(graph["node_features"]) > 0:
        print(f"Node feature dim: {len(graph['node_features'][0])}")
    else:
        print("Node feature dim: not available")

    # Edge feature dimension
    if "edge_features" in graph and len(graph["edge_features"]) > 0:
        print(f"Edge feature dim: {len(graph['edge_features'][0])}")
    else:
        print("Edge feature dim: not available")

    # Edge index shape (PyTorch Geometric style)
    if "edge_index" in graph and len(graph["edge_index"]) == 2:
        print(f"Edge index shape: [2, {len(graph['edge_index'][0])}]")
    else:
        print("Edge index shape: not available")


def build_object_map(graph, height, width):
    """
    Convert graph → 2D array of object IDs (for plotting).

    Used by matplotlib visualization.
    """
    obj_map = np.full((height, width), -1, dtype=int)
    for node in graph["nodes"]:
        for r, c in node["cells"]:
            obj_map[r, c] = node["id"]
    return obj_map


def plot_grids(input_grid, rebuilt_grid, graph, title=""):
    """
    Visual comparison of:
        1. Original input
        2. Reconstructed grid
        3. Object segmentation

    This is your BEST debugging tool.
    """
    input_arr = np.array(input_grid)
    rebuilt_arr = np.array(rebuilt_grid)

    h = len(input_grid)
    w = len(input_grid[0])
    obj_map = build_object_map(graph, h, w)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Original input
    axs[0].imshow(input_arr, cmap="tab10", vmin=0, vmax=9)
    axs[0].set_title("Input")

    # Reconstructed grid
    axs[1].imshow(rebuilt_arr, cmap="tab10", vmin=0, vmax=9)
    axs[1].set_title("Rebuilt")

    # Object segmentation
    axs[2].imshow(obj_map, cmap="tab20")
    axs[2].set_title("Objects")

    # Grid lines for clarity
    for ax in axs:
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="lightgray", linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PIPELINE (THIS IS THE IMPORTANT PART)
# ============================================================

def test_file(file_path):
    """
    Full pipeline for ONE ARC task:

    1. Load JSON
    2. Loop through training examples
    3. Convert grid → graph
    4. Convert graph → grid
    5. Verify reconstruction
    6. Print + visualize everything
    """
    task = load_json(file_path)

    print(f"\nFILE: {os.path.basename(file_path)}")

    for i, pair in enumerate(task["train"]):

        # ----------------------------------------------------
        # STEP 1: GET INPUT GRID
        # ----------------------------------------------------
        raw_grid = pair["input"]
        grid = to_dsl_grid(raw_grid)

        print_grid(raw_grid, f"Train {i} INPUT")

        # ----------------------------------------------------
        # STEP 2: BUILD GRAPH (CORE OF YOUR WORK)
        # ----------------------------------------------------
        graph = grid_to_graph(grid)

        # ----------------------------------------------------
        # STEP 3: REBUILD GRID FROM GRAPH
        # ----------------------------------------------------
        rebuilt = graph_to_grid(graph, len(grid), len(grid[0]))

        print_grid(rebuilt, f"Train {i} REBUILT")

        # ----------------------------------------------------
        # STEP 4: VERIFY CORRECTNESS
        # ----------------------------------------------------
        ok = (rebuilt == raw_grid)

        print(f"\nTrain {i}: nodes={len(graph['nodes'])}, rebuild={ok}")

        # ----------------------------------------------------
        # STEP 5: PRINT GRAPH INFO
        # ----------------------------------------------------
        print_graph_summary(graph)
        print_object_map(graph, len(grid), len(grid[0]))

        # ----------------------------------------------------
        # STEP 6: PRINT NODE DETAILS
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # STEP 7: PRINT EDGE DETAILS (RELATIONSHIPS)
        # ----------------------------------------------------
        print("\nSample edges:")
        for edge in graph["edges"][:5]:  # limit output
            print(
                f"  {edge['src']}->{edge['dst']} | "
                f"dx={edge['dx']:.2f}, dy={edge['dy']:.2f}, "
                f"manhattan={edge['manhattan']:.2f}, dist={edge['dist']:.2f}, "
                f"touching={edge['touching']}, same_color={edge['same_color']}, "
                f"same_row={edge['same_row']}, same_col={edge['same_col']}, "
                f"same_area={edge['same_area']}, bbox_overlap={edge['bbox_overlap']}"
            )

        # ----------------------------------------------------
        # STEP 8: PRINT NUMERIC REPRESENTATION (ML READY)
        # ----------------------------------------------------
        print("\nNumeric graph:")
        print(f"  node_features shape = [{len(graph['node_features'])}, {len(graph['node_features'][0])}]")
        print(f"  edge_features shape = [{len(graph['edge_features'])}, {len(graph['edge_features'][0])}]")
        print(f"  edge_index shape    = [2, {len(graph['edge_index'][0])}]")

        # ----------------------------------------------------
        # STEP 9: VISUAL DEBUGGING
        # ----------------------------------------------------
        plot_grids(raw_grid, rebuilt, graph,
                   title=f"{os.path.basename(file_path)} - Train {i}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Run pipeline on a single ARC task file.

    You can swap this filename to test others.
    """
    file_path = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file_path)