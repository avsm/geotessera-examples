#!/usr/bin/env python3
# /// script
# dependencies = [
#   "geotessera",
#   "scikit-learn",
#   "umap-learn",
#   "matplotlib",
#   "rasterio",
#   "numpy",
# ]
# ///
"""Solar panel detection example using GeoTessera embeddings.

This example demonstrates:
1. Loading labeled training/test points from GeoJSON
2. Using sample_embeddings_at_points() API to extract embeddings
3. Training a simple logistic regression classifier
4. Generating pixel-level predictions across tiles

Usage:
    uv run python solarpanel/main.py --data-dir /path/to/data
"""

from geotessera import GeoTessera
import json
import rasterio
import numpy as np
import argparse
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solar panel detection using GeoTessera')
parser.add_argument('--data-dir', type=Path,
                    help='Directory containing data files (default: script directory)')
args = parser.parse_args()

# Set data directory (default to script's directory)
if args.data_dir:
    data_dir = args.data_dir
else:
    data_dir = Path(__file__).parent

# Add data_dir to path so we can import util
sys.path.insert(0, str(data_dir))
from util import load_fetch_collection, train_with_label_subset, visualize_embeddings

gt = GeoTessera(embeddings_dir="embeddings")

# load bounding box from bbox.json
bbox_file = data_dir / 'bbox.json'
if not bbox_file.exists():
    print(f"Error: bbox.json not found at {bbox_file}")
    print(f"Data directory: {data_dir}")
    sys.exit(1)

bounding_box = json.load(open(bbox_file))['bbox']

# load training and test sets
train_positive = [(a, True) for a in load_fetch_collection(str(data_dir / 'train_positive.geojson'))]
train_negative = [(a, False) for a in load_fetch_collection(str(data_dir / 'train_negative.geojson'))]
test_positive = [(a, True) for a in load_fetch_collection(str(data_dir / 'test_positive.geojson'))]
test_negative = [(a, False) for a in load_fetch_collection(str(data_dir / 'test_negative.geojson'))]

# concatenate to train and test sets
train = train_positive + train_negative
test = test_positive + test_negative

# Extract just the coordinates for sampling
train_points = [coord for coord, label in train]
test_points = [coord for coord, label in test]

print("Sampling training embeddings...")
train_embeddings = gt.sample_embeddings_at_points(train_points, year=2024)

print("Sampling test embeddings...")
test_embeddings = gt.sample_embeddings_at_points(test_points, year=2024)

# Extract labels
train_labels = np.array([label for coord, label in train], dtype=np.bool_)
test_labels = np.array([label for coord, label in test], dtype=np.bool_)

# Check for any NaN values (points outside coverage)
train_valid = ~np.any(np.isnan(train_embeddings), axis=1)
test_valid = ~np.any(np.isnan(test_embeddings), axis=1)

if not np.all(train_valid):
    print(f"Warning: {np.sum(~train_valid)} training points outside coverage")
    train_embeddings = train_embeddings[train_valid]
    train_labels = train_labels[train_valid]

if not np.all(test_valid):
    print(f"Warning: {np.sum(~test_valid)} test points outside coverage")
    test_embeddings = test_embeddings[test_valid]
    test_labels = test_labels[test_valid]

print(f"Found {len(train_embeddings)} training points and {len(test_embeddings)} test points.")

# visualize embeddings
visualize_embeddings(train_embeddings, train_labels, output_path=data_dir / 'train_embeddings_umap.png')

# simple training of a logistic regression model
model = LogisticRegression(max_iter=1000)

# analyze performance with different label subsets
label_subsets = [1, 5, 10, 20, 50, 100, len(train_embeddings) // 2]
train_with_label_subset(train_embeddings, train_labels, test_embeddings, test_labels, model, label_subsets, num_times=10)

# final training on all data
model.fit(train_embeddings, train_labels)

print("Training accuracy:", model.score(train_embeddings, train_labels))
print("Test accuracy:", model.score(test_embeddings, test_labels))

# we go through the embedding tiles and classify each pixel
# then we write out a GeoTIFF with the results
print("\nGenerating predictions for all tiles...")
embeddings = gt.fetch_embeddings(gt.registry.load_blocks_for_region(bounding_box, year=2024))

# Count tiles for progress reporting
tiles_processed = 0
for year, lon, lat, embedding, crs, transform in embeddings:
    # embedding is H,W,128 can reshape to (-1, 128) for prediction
    h, w, _ = embedding.shape
    reshaped = embedding.reshape(-1, 128)
    preds = model.predict(reshaped)
    # reshape back to H,W
    pred_image = 255 - (preds.reshape(h, w).astype(np.uint8) * 255)
    # write out GeoTIFF using rasterio
    out_meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
    }
    # create filename based on lon/lat (in data_dir/output/)
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    out_filename = output_dir / f"prediction_{lon:.4f}_{lat:.4f}.tif"
    with rasterio.open(out_filename, "w", **out_meta) as dest:
        dest.write(pred_image, 1)

    tiles_processed += 1
    print(f"Processed tile {tiles_processed}: ({lon:.2f}, {lat:.2f})")

print(f"\n✅ Predictions saved to {output_dir}/ ({tiles_processed} tiles)")
