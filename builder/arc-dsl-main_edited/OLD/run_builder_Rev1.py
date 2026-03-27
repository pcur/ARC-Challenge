import json
import os
from graph_builder_Rev1 import grid_to_graph, graph_to_grid

TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\data\training"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# 🔥 FIX: convert JSON grid (list) → DSL grid (tuple)
def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


def test_file(file_path):
    task = load_json(file_path)

    print(f"\nFILE: {os.path.basename(file_path)}")

    for i, pair in enumerate(task["train"]):

        # 🔥 FIX APPLIED HERE
        grid = to_dsl_grid(pair["input"])

        graph = grid_to_graph(grid)

        rebuilt = graph_to_grid(graph, len(grid), len(grid[0]))

        # convert DSL grid back to list for comparison
        original = [list(row) for row in grid]

        ok = (rebuilt == original)

        print(f"Train {i}: nodes={len(graph['nodes'])}, rebuild={ok}")


if __name__ == "__main__":
    file = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file)