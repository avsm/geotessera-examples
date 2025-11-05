from geotessera import GeoTessera
import json
import rasterio
import cleanlab
from tqdm import tqdm
import numpy as np
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
import umap

# loads and converts a geojson feature
# collection to a list of lon/lat tuples
def load_fetch_collection(f):
    data = json.load(open(f))
    features = data['features']
    coords = []
    for feature in features:
        lon, lat = feature['geometry']['coordinates']
        coords.append((lon, lat))
    return coords

# finds points from a set that fall within a tile
# and extracts their embeddings
def find_points_in_tile(tile_lon, tile_lat, transform, embedding, crs, points_set, embeddings_array, embeddings_count, labels_array):
    min_lon, max_lon = tile_lon - 0.05, tile_lon + 0.05
    min_lat, max_lat = tile_lat - 0.05, tile_lat + 0.05

    # Create transformer from WGS84 (EPSG:4326) to tile's CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    for (point_lon, point_lat), label in points_set:
        if min_lon <= point_lon < max_lon and min_lat <= point_lat < max_lat:
            # Transform from lat/lon to tile's projected coordinates
            x, y = transformer.transform(point_lon, point_lat)
            row, col = rasterio.transform.rowcol(transform, x, y)
            embeddings_array[embeddings_count] = embedding[row, col]
            labels_array[embeddings_count] = 1 if label else 0
            embeddings_count += 1

    return embeddings_count

def train_with_label_subset(X_train, y_train, X_test, y_test, model, label_subsets, num_times):
    X_train_true = X_train[y_train == 1]
    X_train_false = X_train[y_train == 0]
    rng = np.random.default_rng(seed=42)
    test_scores = []
    test_recalls = []
    test_precisions = []

    for num_labels in label_subsets:
        for _ in range(num_times):
            X = np.vstack([rng.choice(X_train_true, num_labels, replace=False), rng.choice(X_train_false, num_labels, replace=False)])
            y = np.hstack([np.ones(num_labels), np.zeros(num_labels)])
            # shuffle X and y in unison
            indices = np.arange(X.shape[0])
            rng.shuffle(indices)
            X = X[indices]
            y = y[indices]
            model.fit(X, y)
            train_score = model.score(X, y)
            test_score = model.score(X_test, y_test)
            test_scores.append(test_score)
            # compute precision and recall
            y_pred = model.predict(X_test)
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_precisions.append(precision)
            test_recalls.append(recall)

        avg_test_score = np.mean(test_scores[-num_times:])
        min_test_score = np.min(test_scores[-num_times:])
        max_test_score = np.max(test_scores[-num_times:])
        avg_precision = np.mean(test_precisions[-num_times:])
        avg_recall = np.mean(test_recalls[-num_times:])
        print(f"Labels: {num_labels*2}, Train accuracy: {train_score:.4f}, Test accuracy: {avg_test_score:.4f} (min: {min_test_score:.4f}, max: {max_test_score:.4f}) Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")


gt = GeoTessera()

# load bounding box from bbox.json
bounding_box = json.load(open('bbox.json'))['bbox']

# load training and test sets
train_positive = [(a, True) for a in load_fetch_collection('train_positive.geojson')]
train_negative = [(a, False) for a in load_fetch_collection('train_negative.geojson')]
test_positive = [(a, True) for a in load_fetch_collection('test_positive.geojson')]
test_negative = [(a, False) for a in load_fetch_collection('test_negative.geojson')]

# concatenate to train and test sets
train = train_positive + train_negative
test = test_positive + test_negative

# now we fetch the embeddings for the bounding box
# (this returns a generator, so we can iterate over it lazily)
embeddings = gt.fetch_embeddings(bounding_box)

# preallocate arrays for embeddings and labels
train_embeddings = np.zeros((len(train_positive) + len(train_negative), 128), dtype=np.float32)
train_labels = np.zeros((len(train_positive) + len(train_negative),), dtype=np.int32)
train_count = 0

test_embeddings = np.zeros((len(test_positive) + len(test_negative), 128), dtype=np.float32)
test_labels = np.zeros((len(test_positive) + len(test_negative),), dtype=np.int32)
test_count = 0

# count total tiles to process for progress bar
tile_count = gt.embeddings_count(bounding_box)

# now we iterate over the tiles and find points with labels
# we use this to populate our train and test embeddings and labels arrays
for lon, lat, embedding, crs, transform in tqdm(embeddings, total=tile_count):
    train_count = find_points_in_tile(lon, lat, transform, embedding, crs, train, train_embeddings, train_count, train_labels)
    test_count = find_points_in_tile(lon, lat, transform, embedding, crs, test, test_embeddings, test_count, test_labels)

print(f"Found {train_count} training points and {test_count} test points in embeddings.")
print(f"Training set: {len(train)} points, Test set: {len(test)} points.")

# check we found the number of points we expected
assert train_count == len(train)
assert test_count == len(test)

# simple training of a logistic regression model
model = LogisticRegression(max_iter=1000)

cl = cleanlab.classification.CleanLearning(model)

label_issues = cl.find_label_issues(
    train_embeddings, train_labels
)

print(label_issues[label_issues['label_quality'] < 0.9])

clean_labels = label_issues[label_issues['label_quality'] > 0.9].index.tolist()

# write train_clean_positive.geojson and train_clean_negative.geojson
# (only include training examples where label_quality is > 0.9)

train_clean_positive = [train[i][0] for i in clean_labels if train[i][1] == True]
train_clean_negative = [train[i][0] for i in clean_labels if train[i][1] == False]

# make sure train_clean_positive and train_clean_negative are the same size
sz = min(len(train_clean_positive), len(train_clean_negative))

train_clean_positive = train_clean_positive[:sz]
train_clean_negative = train_clean_negative[:sz]

with open('train_clean_positive.geojson', 'w') as f:
    json.dump({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {}
        } for lon, lat in train_clean_positive]
    }, f, indent=2)

with open('train_clean_negative.geojson', 'w') as f:
    json.dump({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {}
        } for lon, lat in train_clean_negative]
    }, f, indent=2)