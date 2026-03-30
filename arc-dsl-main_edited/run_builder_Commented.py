# ------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------

import json
import os

# ------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------
# matplotlib is used to visualize:
#   1. the original input grid
#   2. the rebuilt grid (from graph -> grid reconstruction)
#   3. the object map (each detected object shown with its node ID)
#
# numpy is used mainly for easier 2D array handling when building
# and plotting the object map.
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Import the two main functions from graph_builder.py
# ------------------------------------------------------------
# grid_to_graph(grid):
#   takes a DSL-format grid and converts it into a graph
#
# graph_to_grid(graph, height, width):
#   reconstructs the original grid from the graph
# ------------------------------------------------------------
from graph_builder import grid_to_graph, graph_to_grid


# ------------------------------------------------------------
# Path to the ARC training folder
# ------------------------------------------------------------
# This is the folder containing the ARC JSON files.
# In this script, we later choose one specific file from here:
#   00d62c1b.json
# ------------------------------------------------------------
TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"


# ------------------------------------------------------------
# load_json(path)
# ------------------------------------------------------------
# Purpose:
#   Open an ARC JSON file and load it into Python as a dictionary.
#
# ARC files are typically structured like:
# {
#   "train": [
#       {"input": [...], "output": [...]},
#       ...
#   ],
#   "test": [
#       {"input": [...], "output": [...]},
#       ...
#   ]
# }
#
# Input:
#   path = full path to the JSON file
#
# Output:
#   Parsed Python dictionary
# ------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# to_dsl_grid(grid)
# ------------------------------------------------------------
# Purpose:
#   Convert the grid loaded from JSON into the format expected
#   by the ARC DSL functions.
#
# Why this is needed:
#   JSON gives us a list of lists, like:
#       [[0,0,3], [0,3,0]]
#
#   The DSL expects an immutable tuple-of-tuples, like:
#       ((0,0,3), (0,3,0))
#
# Input:
#   grid = list-of-lists from JSON
#
# Output:
#   tuple-of-tuples
# ------------------------------------------------------------
def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


# ------------------------------------------------------------
# print_grid(grid, title="")
# ------------------------------------------------------------
# Purpose:
#   Print a grid to the terminal in a readable row-by-row format.
#
# This is useful for:
#   - checking the original input grid
#   - checking the reconstructed grid
#   - confirming reconstruction correctness by visual comparison
#
# Input:
#   grid  = 2D grid (list-of-lists or tuple-of-tuples)
#   title = optional label printed above the grid
# ------------------------------------------------------------
def print_grid(grid, title=""):
    print(f"\n{title}")
    for row in grid:
        print(" ".join(str(x) for x in row))


# ------------------------------------------------------------
# print_object_map(graph, height, width)
# ------------------------------------------------------------
# Purpose:
#   Print a text-based "object map" showing which node ID occupies
#   each cell in the grid.
#
# How it works:
#   - Start with a blank grid filled with -1
#   - For each node, fill in its occupied cells with that node's ID
#
# Meaning of values:
#   -1 = background / no object
#    0 = node 0
#    1 = node 1
#    2 = node 2
#   etc.
#
# This is one of the easiest ways to inspect object segmentation.
#
# Inputs:
#   graph  = graph dictionary returned by grid_to_graph(...)
#   height = grid height
#   width  = grid width
# ------------------------------------------------------------
def print_object_map(graph, height, width):
    # Start with a blank map filled with -1 (background)
    grid = [[-1 for _ in range(width)] for _ in range(height)]

    # For every detected node, mark its cells with that node's ID
    for node in graph["nodes"]:
        nid = node["id"]
        for r, c in node["cells"]:
            grid[r][c] = nid

    print("\nObject Map (node IDs):")
    for row in grid:
        print(" ".join(f"{x:2}" for x in row))


# ------------------------------------------------------------
# print_graph_summary(graph)
# ------------------------------------------------------------
# Purpose:
#   Print a compact summary of the graph produced by the builder.
#
# This includes:
#   - number of nodes
#   - number of edges
#   - node feature dimension
#   - edge feature dimension
#   - edge_index shape
#
# Why this matters:
#   This tells us whether the graph is model-ready and what the
#   tensor sizes will look like for the next stage.
# ------------------------------------------------------------
def print_graph_summary(graph):
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")

    # node_features should be a list of N feature vectors
    if "node_features" in graph and len(graph["node_features"]) > 0:
        print(f"Node feature dim: {len(graph['node_features'][0])}")
    else:
        print("Node feature dim: not available")

    # edge_features should be a list of E feature vectors
    if "edge_features" in graph and len(graph["edge_features"]) > 0:
        print(f"Edge feature dim: {len(graph['edge_features'][0])}")
    else:
        print("Edge feature dim: not available")

    # edge_index should have shape [2, E]
    if "edge_index" in graph and len(graph["edge_index"]) == 2:
        print(f"Edge index shape: [2, {len(graph['edge_index'][0])}]")
    else:
        print("Edge index shape: not available")


# ------------------------------------------------------------
# build_object_map(graph, height, width)
# ------------------------------------------------------------
# Purpose:
#   Build a NumPy array that stores node IDs at each grid location.
#
# This is the array version of the object map used for plotting.
#
# Output:
#   A NumPy array of shape [height, width]
#   containing:
#       -1 for background
#       node_id for object cells
#
# Why separate from print_object_map?
#   print_object_map() is for terminal text display
#   build_object_map() is for plotting with matplotlib
# ------------------------------------------------------------
def build_object_map(graph, height, width):
    # Fill background with -1
    obj_map = np.full((height, width), -1, dtype=int)

    # Write node IDs into the cells occupied by each object
    for node in graph["nodes"]:
        for r, c in node["cells"]:
            obj_map[r, c] = node["id"]

    return obj_map


# ------------------------------------------------------------
# plot_grids(input_grid, rebuilt_grid, graph, title="")
# ------------------------------------------------------------
# Purpose:
#   Visualize three things side-by-side:
#       1. Original input grid
#       2. Rebuilt grid (graph -> grid)
#       3. Object map (objects colored by node ID)
#
# Why this matters:
#   This is the easiest visual sanity check for the builder.
#
# If input == rebuilt:
#   reconstruction is correct
#
# If the object map looks sensible:
#   object extraction is working
#
# Inputs:
#   input_grid   = original grid from JSON
#   rebuilt_grid = reconstructed grid from the graph
#   graph        = graph produced by grid_to_graph(...)
#   title        = optional plot title
# ------------------------------------------------------------
def plot_grids(input_grid, rebuilt_grid, graph, title=""):
    # Convert grids to NumPy arrays for plotting
    input_arr = np.array(input_grid)
    rebuilt_arr = np.array(rebuilt_grid)

    # Get grid shape
    h = len(input_grid)
    w = len(input_grid[0])

    # Build object-ID map for plotting
    obj_map = build_object_map(graph, h, w)

    # Create a 1x3 figure:
    # [Input | Rebuilt | Objects]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # --------------------------------------------------------
    # Plot original input grid
    # tab10 works well for ARC colors 0..9
    # --------------------------------------------------------
    axs[0].imshow(input_arr, cmap="tab10", vmin=0, vmax=9)
    axs[0].set_title("Input")

    # --------------------------------------------------------
    # Plot rebuilt grid
    # Should match the input exactly if reconstruction worked
    # --------------------------------------------------------
    axs[1].imshow(rebuilt_arr, cmap="tab10", vmin=0, vmax=9)
    axs[1].set_title("Rebuilt")

    # --------------------------------------------------------
    # Plot object map
    # Each node ID gets a different display color
    # Note:
    #   These colors are just visualization colors, not ARC colors
    # --------------------------------------------------------
    axs[2].imshow(obj_map, cmap="tab20")
    axs[2].set_title("Objects")

    # --------------------------------------------------------
    # Add light gridlines and hide tick labels for readability
    # --------------------------------------------------------
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
# test_file(file_path)
# ------------------------------------------------------------
# Purpose:
#   Main driver for testing the builder on one ARC JSON file.
#
# What it does for each training example:
#   1. Load the input grid
#   2. Convert it to DSL format
#   3. Print the input grid
#   4. Build the graph
#   5. Reconstruct the grid from the graph
#   6. Print the rebuilt grid
#   7. Print graph summary, nodes, and sample edges
#   8. Plot input / rebuilt / object map
#
# Important:
#   This is NOT training a model.
#   This is just testing and visualizing the builder.
# ------------------------------------------------------------
def test_file(file_path):
    # Load ARC JSON file
    task = load_json(file_path)

    print(f"\nFILE: {os.path.basename(file_path)}")

    # --------------------------------------------------------
    # Loop through every training example in this ARC file
    # --------------------------------------------------------
    for i, pair in enumerate(task["train"]):
        # Raw grid from JSON (list-of-lists)
        raw_grid = pair["input"]

        # Convert to DSL-compatible tuple-of-tuples
        grid = to_dsl_grid(raw_grid)

        # Print original input grid
        print_grid(raw_grid, f"Train {i} INPUT")

        # ----------------------------------------------------
        # Build the graph from the input grid
        # ----------------------------------------------------
        graph = grid_to_graph(grid)

        # ----------------------------------------------------
        # Reconstruct the grid from the graph
        # This checks whether our graph representation is lossless
        # ----------------------------------------------------
        rebuilt = graph_to_grid(graph, len(grid), len(grid[0]))

        # Print rebuilt grid
        print_grid(rebuilt, f"Train {i} REBUILT")

        # Check whether reconstruction exactly matches input
        ok = (rebuilt == raw_grid)

        print(f"\nTrain {i}: nodes={len(graph['nodes'])}, rebuild={ok}")

        # Print compact graph summary
        print_graph_summary(graph)

        # Print text-based object ID map
        print_object_map(graph, len(grid), len(grid[0]))

        # ----------------------------------------------------
        # Print detailed node information
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
        # Print a few sample edges
        # (Not all edges, because fully connected graphs can get large)
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Print model-ready tensor dimensions
        # This is the data that would actually be passed to a GNN/GAT
        # ----------------------------------------------------
        print("\nNumeric graph:")
        print(f"  node_features shape = [{len(graph['node_features'])}, {len(graph['node_features'][0])}]")
        print(f"  edge_features shape = [{len(graph['edge_features'])}, {len(graph['edge_features'][0])}]")
        print(f"  edge_index shape    = [2, {len(graph['edge_index'][0])}]")

        # ----------------------------------------------------
        # Show visual plots for this training example
        # ----------------------------------------------------
        plot_grids(raw_grid, rebuilt, graph, title=f"{os.path.basename(file_path)} - Train {i}")


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------
# This block runs only when the file is executed directly.
#
# It selects one ARC file:
#   00d62c1b.json
#
# Then calls test_file(...) to run the full builder test on it.
# ------------------------------------------------------------
if __name__ == "__main__":
    file_path = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file_path)