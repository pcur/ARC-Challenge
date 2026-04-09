import json
import os
from graph_builder_Rev1 import grid_to_graph, graph_to_grid

import networkx as nx
import matplotlib.pyplot as plt

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\data\training"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


def plot_graph(graph, title="Graph"):
    G = nx.DiGraph()

    # Add nodes
    for node in graph["nodes"]:
        G.add_node(node["id"], label=f"{node['id']}")

    # Add edges
    for edge in graph["edges"]:
        G.add_edge(edge["src"], edge["dst"])

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Draw graph
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=800,
        font_size=10
    )

    plt.title(title)
    plt.show()


def test_file(file_path):
    task = load_json(file_path)

    print(f"\nFILE: {os.path.basename(file_path)}")

    for i, pair in enumerate(task["train"]):
        grid = to_dsl_grid(pair["input"])

        graph = grid_to_graph(grid)
        rebuilt = graph_to_grid(graph, len(grid), len(grid[0]))

        original = [list(row) for row in grid]
        ok = (rebuilt == original)

        print(f"\nTrain {i}: nodes={len(graph['nodes'])}, rebuild={ok}")

        print("  Nodes:")
        for node in graph["nodes"]:
            print(
                f"    Node {node['id']}: "
                f"colors={node['colors']}, "
                f"area={node['area']}, "
                f"centroid={node['centroid']}, "
                f"bbox={node['bbox']}, "
                f"width={node['width']}, "
                f"height={node['height']}"
            )

        print("  Sample edges:")
        for edge in graph["edges"][:5]:
            print(
                f"    {edge['src']}->{edge['dst']} | "
                f"dx={edge['dx']:.2f}, "
                f"dy={edge['dy']:.2f}, "
                f"dist={edge['dist']:.2f}, "
                f"touching={edge['touching']}, "
                f"same_color={edge['same_color']}"
            )

        # ACTUALLY CALL THE GRAPH PLOTTER
        plot_graph(graph, title=f"Train {i} Graph")


if __name__ == "__main__":
    file = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file)