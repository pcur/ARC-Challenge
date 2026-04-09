Only need to run the following files:

arc_trm_graph_hybrid.py (train with TRM using hybrid_object2.py)
visualise_predictions_trm_graph_hybrid.py (generates results in visualisations_trm_graph_hybrid folder)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Using builder hybrid_object2.py
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Test builder by running run_builder_hybrid.py

Currently testing json file# 00d62c1b.  To change the file being used, update specifally in main (see below)

if __name__ == "__main__":
    file_path = os.path.join(TRAIN_PATH, "00d62c1b.json")
    test_file(file_path)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
These files were initially built off the ARC-DSL-MAIN folder, but has since been updated and no longer uses those original files (inital and older files located in folders OLD and OLD2)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The difference between folders that include the word "custom" and "hybrid" are in regards to their node and edge features:

CUSTOM
Objects are represented using precise cell-level geometry. Each node stores the exact pixel coordinates of every cell in the object — where each pixel actually lives on the grid. The node feature vector is 22-dimensional: 10 color one-hot + 12 geometric features (area, centroid, bounding box corners, width, height, density, aspect ratio, single-pixel flag). Edge features are 12-dimensional and include distance, manhattan distance, same-color, touching, bbox overlap, same-row, same-col, same-area, and area ratios.

At reconstruction time, the decoder predicts per-cell coordinates and colors, then paints each predicted cell back onto a blank grid. This is precise but requires the decoder to predict up to 40 cell positions per object.

*These contain the same nubmer of features as the original DSL Object function 

Node features — describing one object:
    - Color — what color(s) the object is made of
    - Area — how many pixels it contains (is it tiny or large?)
    - Centroid row/col — where is the object's center of mass on the grid
    - Bounding box (4 values) — the top, bottom, left, right edges of the rectangle that encloses the object
    - Width / Height — how wide and tall that enclosing rectangle is
    - Density — how much of the bounding box is actually filled (a solid square = 1.0, a cross shape = less than 1.0, a diagonal line = very low)
    - Aspect ratio — is it wider than it is tall, or taller than wide
    - Is single pixel — is this just one dot
Edge features — describing the relationship between two objects A and B:
    - dx / dy — which direction is B from A (left/right, up/down)
    - Manhattan distance — how many steps apart are they on the grid
    - Euclidean distance — straight-line distance between their centers
    - Same color — do A and B share a color
    - Touching — are they physically adjacent (sharing a border pixel)
    - Bounding box overlap — do their enclosing rectangles intersect
    - Same row — are their centers on the same horizontal line
    - Same column — are their centers on the same vertical line
    - Same area — do they have exactly the same number of pixels
    - Area ratio (both directions) — how much bigger is A than B, and B than A


HYBRID
Objects are represented using a shape mask instead of explicit cell coordinates. Each node stores a flattened 10×10 binary mask showing the object's silhouette within its bounding box, plus a bounding box position. The node feature vector is 110-dimensional: 10 color one-hot + 100 flattened shape mask pixels. Edge features are only 5-dimensional: dx, dy, touching, same-row, same-col.

At reconstruction time, the decoder predicts a shape mask and a bounding box, then stamps the mask at the predicted location. This is coarser but simpler — the decoder doesn't need to predict individual cell coordinates.

Node features — describing one object:
    - Color — same as custom
    - Shape mask — a 10×10 pixel picture of exactly what the object looks like (its actual silhouette), scaled to fit in a 10×10 box
Edge features — describing the relationship between two objects A and B:
    - dx / dy — which direction is B from A
    - Touching — are they physically adjacent
    - Same row — are their centers on the same horizontal line
    - Same column — are their centers on the same vertical line
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

