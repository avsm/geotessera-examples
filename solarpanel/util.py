import json
import numpy as np
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

def visualize_embeddings(train_embeddings, train_labels, output_path='train_embeddings_umap.png'):
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(train_embeddings)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=train_labels, cmap='coolwarm', s=1)
    plt.colorbar(label='Label')
    plt.title('UMAP projection of training embeddings')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"UMAP visualization saved to {output_path}")
