import rasterio
from rasterio.warp import transform as rio_transform
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import binary_erosion

# Fixed random seed for reproducibility
np.random.seed(42)

# Read the GeoTIFF
src = rasterio.open('roi_1_clipped_gt_10m.tif')
data = src.read(1)
height, width = data.shape

# Split vertically: left half = train, right half = test
mid_col = width // 2
train_data = data[:, :mid_col]
test_data = data[:, mid_col:]

# Create binary masks of solar panels
train_solar_mask = train_data == 1
test_solar_mask = test_data == 1

# Erode solar panel regions to avoid sampling edge pixels
# This removes pixels near boundaries that may contain mixed land types
erosion_kernel = np.ones((5, 5))  # 5x5 kernel = ~2-3 pixel buffer from edges
train_solar_eroded = binary_erosion(train_solar_mask, structure=erosion_kernel)
test_solar_eroded = binary_erosion(test_solar_mask, structure=erosion_kernel)

# Get pixel coordinates for positive (solar) pixels from eroded regions
# Value 1 = solar panel, 255 = no solar panel
train_pos_indices = np.argwhere(train_solar_eroded)
test_pos_indices = np.argwhere(test_solar_eroded)

# Adjust test indices to account for the split
test_pos_indices[:, 1] += mid_col

# Get negative pixel coordinates (ensure they're not positive)
train_neg_indices = np.argwhere(train_data != 1)
test_neg_indices = np.argwhere(test_data != 1)
test_neg_indices[:, 1] += mid_col

# Sample same number of negative as positive
n_train_pos = len(train_pos_indices)
n_test_pos = len(test_pos_indices)

# Ensure we have enough negative samples
assert len(train_neg_indices) >= n_train_pos, "Not enough negative samples in training set"
assert len(test_neg_indices) >= n_test_pos, "Not enough negative samples in test set"

train_neg_sample = train_neg_indices[np.random.choice(len(train_neg_indices), n_train_pos, replace=False)]
test_neg_sample = test_neg_indices[np.random.choice(len(test_neg_indices), n_test_pos, replace=False)]

# Downsample to 25%
downsample_ratio = 0.25
train_pos_downsampled = train_pos_indices[np.random.choice(len(train_pos_indices), int(n_train_pos * downsample_ratio), replace=False)]
train_neg_downsampled = train_neg_sample[np.random.choice(len(train_neg_sample), int(n_train_pos * downsample_ratio), replace=False)]
test_pos_downsampled = test_pos_indices[np.random.choice(len(test_pos_indices), int(n_test_pos * downsample_ratio), replace=False)]
test_neg_downsampled = test_neg_sample[np.random.choice(len(test_neg_sample), int(n_test_pos * downsample_ratio), replace=False)]

# Function to convert pixel coordinates to lon/lat
def pixels_to_lonlat(indices, src):
    features = []
    for row, col in indices:
        # Get UTM coordinates from pixel coordinates
        x, y = src.xy(row, col)
        # Transform from EPSG:32630 to WGS84 (EPSG:4326)
        lon, lat = rio_transform(src.crs, 'EPSG:4326', [x], [y])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon[0], lat[0]]
            },
            "properties": {}
        })
    return features

# Create GeoJSON for each dataset (using downsampled data)
train_positive_geojson = {
    "type": "FeatureCollection",
    "features": pixels_to_lonlat(train_pos_downsampled, src)
}

train_negative_geojson = {
    "type": "FeatureCollection",
    "features": pixels_to_lonlat(train_neg_downsampled, src)
}

test_positive_geojson = {
    "type": "FeatureCollection",
    "features": pixels_to_lonlat(test_pos_downsampled, src)
}

test_negative_geojson = {
    "type": "FeatureCollection",
    "features": pixels_to_lonlat(test_neg_downsampled, src)
}

# Write to files
with open('train_positive.geojson', 'w') as f:
    json.dump(train_positive_geojson, f, indent=2)

with open('train_negative.geojson', 'w') as f:
    json.dump(train_negative_geojson, f, indent=2)

with open('test_positive.geojson', 'w') as f:
    json.dump(test_positive_geojson, f, indent=2)

with open('test_negative.geojson', 'w') as f:
    json.dump(test_negative_geojson, f, indent=2)

# Calculate bounding box for the entire GeoTIFF
bounds = src.bounds
# Transform corners to WGS84
min_lon, min_lat = rio_transform(src.crs, 'EPSG:4326', [bounds.left], [bounds.bottom])
max_lon, max_lat = rio_transform(src.crs, 'EPSG:4326', [bounds.right], [bounds.top])

bbox = {
    "bbox": [min_lon[0], min_lat[0], max_lon[0], max_lat[0]]
}

# pad bbox by 0.01 degrees
pad = 0.01
bbox['bbox'][0] -= pad
bbox['bbox'][1] -= pad
bbox['bbox'][2] += pad
bbox['bbox'][3] += pad

with open('bbox.json', 'w') as f:
    json.dump(bbox, f, indent=2)

src.close()

print(f"Train: {len(train_pos_downsampled)} positive samples, {len(train_neg_downsampled)} negative samples")
print(f"Test: {len(test_pos_downsampled)} positive samples, {len(test_neg_downsampled)} negative samples")
print("Files created: train_positive.geojson, train_negative.geojson, test_positive.geojson, test_negative.geojson, bbox.json")