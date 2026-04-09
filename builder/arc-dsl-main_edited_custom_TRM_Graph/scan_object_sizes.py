"""
scan_object_sizes.py

Scans all ARC training JSON files and reports statistics about object sizes
(number of cells per object). This tells us what max_cells_per_node should
be set to in the cell coordinate decoder.

Run with:
    python scan_object_sizes.py

Update TRAIN_PATH to point at your local ARC training data folder.
"""

import os
import json
from OLD2.custom_object2 import grid_to_graph, my_objects, mostcolor

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\tedhun\OneDrive - Wanzl GmbH & Co. KGaA\Microsoft Teams Chat Files\outputs\Documents\GS\ARC\Assignment10\Builder\data\training"

# If True, also scan output grids (recommended — VAE will train on both)
SCAN_OUTPUTS = True
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_dsl_grid(grid):
    return tuple(tuple(row) for row in grid)


def scan_file(fpath, scan_outputs=True):
    """
    Returns a list of (n_cells, file, split, pair_idx, grid_type) for every
    object found in every grid in the file.
    """
    task = load_json(fpath)
    fname = os.path.basename(fpath)
    results = []

    for split in ["train", "test"]:
        for i, pair in enumerate(task.get(split, [])):
            grids_to_check = [("input", pair["input"])]
            if scan_outputs and "output" in pair:
                grids_to_check.append(("output", pair["output"]))

            for grid_type, raw_grid in grids_to_check:
                grid = to_dsl_grid(raw_grid)
                objs = my_objects(grid)

                for obj in objs:
                    n_cells = len(obj)
                    results.append((n_cells, fname, split, i, grid_type))

    return results


def main():
    files = sorted([f for f in os.listdir(TRAIN_PATH) if f.endswith(".json")])
    print(f"Scanning {len(files)} files in: {TRAIN_PATH}")
    print(f"Scanning outputs: {SCAN_OUTPUTS}\n")

    all_results = []
    max_nodes_per_graph = []

    for fname in files:
        fpath = os.path.join(TRAIN_PATH, fname)
        task = load_json(fpath)

        file_results = scan_file(fpath, scan_outputs=SCAN_OUTPUTS)
        all_results.extend(file_results)

        # Also track max nodes per graph (for max_nodes config)
        for split in ["train", "test"]:
            for pair in task.get(split, []):
                grids = [pair["input"]]
                if SCAN_OUTPUTS and "output" in pair:
                    grids.append(pair["output"])
                for raw_grid in grids:
                    grid = to_dsl_grid(raw_grid)
                    objs = my_objects(grid)
                    max_nodes_per_graph.append(len(objs))

    if not all_results:
        print("No objects found. Check TRAIN_PATH.")
        return

    cell_counts = [r[0] for r in all_results]
    cell_counts.sort()

    n = len(cell_counts)
    total = sum(cell_counts)

    print("=" * 55)
    print("OBJECT SIZE STATISTICS (cells per object)")
    print("=" * 55)
    print(f"  Total objects scanned : {n:,}")
    print(f"  Min cells per object  : {min(cell_counts)}")
    print(f"  Max cells per object  : {max(cell_counts)}")
    print(f"  Mean cells per object : {total / n:.1f}")
    print(f"  Median                : {cell_counts[n // 2]}")
    print(f"  90th percentile       : {cell_counts[int(n * 0.90)]}")
    print(f"  95th percentile       : {cell_counts[int(n * 0.95)]}")
    print(f"  99th percentile       : {cell_counts[int(n * 0.99)]}")

    print()
    print("=" * 55)
    print("NODE COUNT STATISTICS (objects per graph)")
    print("=" * 55)
    max_nodes_per_graph.sort()
    m = len(max_nodes_per_graph)
    print(f"  Total graphs scanned  : {m:,}")
    print(f"  Min nodes per graph   : {min(max_nodes_per_graph)}")
    print(f"  Max nodes per graph   : {max(max_nodes_per_graph)}")
    print(f"  Mean nodes per graph  : {sum(max_nodes_per_graph) / m:.1f}")
    print(f"  90th percentile       : {max_nodes_per_graph[int(m * 0.90)]}")
    print(f"  95th percentile       : {max_nodes_per_graph[int(m * 0.95)]}")
    print(f"  99th percentile       : {max_nodes_per_graph[int(m * 0.99)]}")

    print()
    print("=" * 55)
    print("DISTRIBUTION (cells per object)")
    print("=" * 55)

    # Bucket distribution
    buckets = [1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]
    labels  = ["1", "2-5", "6-10", "11-20", "21-50", "51-100",
               "101-200", "201-500", "501+"]
    counts  = [0] * len(labels)

    for c in cell_counts:
        for idx, upper in enumerate(buckets[1:]):
            if c <= upper:
                counts[idx] += 1
                break

    for label, count in zip(labels, counts):
        pct = 100 * count / n
        bar = "█" * int(pct / 2)
        print(f"  {label:>8} cells : {count:5d} ({pct:5.1f}%) {bar}")

    print()

    # Top 10 largest objects
    all_results.sort(key=lambda x: -x[0])
    print("=" * 55)
    print("TOP 10 LARGEST OBJECTS")
    print("=" * 55)
    for cells, fname, split, idx, gtype in all_results[:10]:
        print(f"  {cells:4d} cells — {fname}  [{split}][{idx}][{gtype}]")

    print()
    print("=" * 55)
    print("RECOMMENDATIONS")
    print("=" * 55)
    p95 = cell_counts[int(n * 0.95)]
    p99 = cell_counts[int(n * 0.99)]
    max_c = max(cell_counts)
    max_n = max(max_nodes_per_graph)
    print(f"  max_cells_per_node (covers 95%) : {p95}")
    print(f"  max_cells_per_node (covers 99%) : {p99}")
    print(f"  max_cells_per_node (covers 100%): {max_c}")
    print(f"  max_nodes (covers 100%)         : {max_n}")
    print()
    print("  Suggested setting: use the 95th or 99th percentile for")
    print("  max_cells_per_node to keep the model size manageable,")
    print("  and accept that very large objects will be truncated.")
    print("  Use the 100th percentile for max_nodes (no truncation).")


if __name__ == "__main__":
    main()