import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn.cluster import DBSCAN as sklearn_DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from minisom import MiniSom
from pyclustering.cluster.clique import clique as pyclique


def plot_comparison(X, labels_a, labels_b, title_a, title_b, filename):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels_a)
    plt.title(title_a)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels_b)
    plt.title(title_b)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def map_labels_by_majority(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mapped = np.full_like(y_pred, -1)
    for cluster_id in np.unique(y_pred):
        if cluster_id == -1:
            continue
        mask = y_pred == cluster_id
        if not np.any(mask):
            continue
        true_labels, counts = np.unique(y_true[mask], return_counts=True)
        mapped_label = true_labels[np.argmax(counts)]
        mapped[mask] = mapped_label
    return mapped


def evaluate_clustering(y_true, y_pred):
    mapped = map_labels_by_majority(y_true, y_pred)
    valid_mask = mapped != -1
    coverage = float(np.mean(valid_mask))
    if not np.any(valid_mask):
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "coverage": coverage,
        }

    y_true_valid = y_true[valid_mask]
    mapped_valid = mapped[valid_mask]
    return {
        "accuracy": accuracy_score(y_true_valid, mapped_valid),
        "precision": precision_score(y_true_valid, mapped_valid, average="macro", zero_division=0),
        "recall": recall_score(y_true_valid, mapped_valid, average="macro", zero_division=0),
        "f1": f1_score(y_true_valid, mapped_valid, average="macro", zero_division=0),
        "coverage": coverage,
    }


def print_report(title, metrics, train_time, exec_time, note=None):
    note_text = f" | note: {note}" if note else ""
    print(
        f"{title}: "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"rec={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"coverage={metrics['coverage']:.2f}, "
        f"train={train_time:.4f}s, "
        f"exec={exec_time:.4f}s{note_text}"
    )


def format_metrics_table(rows):
    headers = [
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Coverage",
        "Train(s)",
        "Exec(s)",
        "Note",
    ]
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_line(cells):
        return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_line(headers), sep]
    for row in rows:
        lines.append(fmt_line(row))
    return "\n".join(lines)


def write_metrics_table(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    table_text = format_metrics_table(rows)
    table_path = os.path.join(output_dir, "metrics_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    csv_path = os.path.join(output_dir, "metrics_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Coverage",
                "TrainSeconds",
                "ExecSeconds",
                "Note",
            ]
        )
        writer.writerows(rows)

    print("\nDetailed metrics table:\n" + table_text)
    print(f"Saved metrics table to {table_path}")
    print(f"Saved metrics CSV to {csv_path}")


class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(42)
        random_idxs = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idxs]
        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = self.centroids.copy()
            for k in range(self.n_clusters):
                members = X[self.labels_ == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


class CustomCLIQUE:
    def __init__(self, intervals, threshold):
        self.intervals = intervals
        self.threshold = threshold
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0]) - 1
        min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
        min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
        x_bins = np.linspace(min_x, max_x, self.intervals + 1)
        y_bins = np.linspace(min_y, max_y, self.intervals + 1)

        grid = {}
        for i, point in enumerate(X):
            idx_x = np.digitize(point[0], x_bins) - 1
            idx_y = np.digitize(point[1], y_bins) - 1
            idx_x = min(max(idx_x, 0), self.intervals - 1)
            idx_y = min(max(idx_y, 0), self.intervals - 1)
            cell = (idx_x, idx_y)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        cluster_id = 0
        for points in grid.values():
            if len(points) >= self.threshold:
                for p in points:
                    self.labels_[p] = cluster_id
                cluster_id += 1
        return self


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.full(X.shape[0], -1)
        cluster_id = 0
        visited = np.zeros(X.shape[0], dtype=bool)

        for i in range(X.shape[0]):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = np.where(np.linalg.norm(X - X[i], axis=1) <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                continue

            self.labels_[i] = cluster_id
            seeds = list(neighbors)
            if i in seeds:
                seeds.remove(i)

            j = 0
            while j < len(seeds):
                neighbor_idx = seeds[j]
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    new_neighbors = np.where(
                        np.linalg.norm(X - X[neighbor_idx], axis=1) <= self.eps
                    )[0]
                    if len(new_neighbors) >= self.min_samples:
                        for n in new_neighbors:
                            if n not in seeds:
                                seeds.append(n)

                if self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id
                j += 1
            cluster_id += 1
        return self


class CustomSOM:
    def __init__(self, x=2, y=2, input_len=2, sigma=1.0, learning_rate=0.5, num_iteration=100):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.weights = None

    def _neighborhood(self, c, sigma):
        d = 2 * sigma * sigma
        if d == 0:
            d = 1e-6
        ax = np.arange(self.x)
        ay = np.arange(self.y)
        xx, yy = np.meshgrid(ax, ay, indexing="ij")
        return np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / d)

    def winner(self, x):
        diff = self.weights - x
        sq_dist = np.sum(diff ** 2, axis=2)
        return np.unravel_index(np.argmin(sq_dist), sq_dist.shape)

    def fit(self, X):
        self.weights = np.random.rand(self.x, self.y, X.shape[1]) * (
            np.max(X, axis=0) - np.min(X, axis=0)
        ) + np.min(X, axis=0)

        for t in range(self.num_iteration):
            lr = self.learning_rate * (1 - t / self.num_iteration)
            sig = self.sigma * (1 - t / self.num_iteration)
            for x in X:
                c = self.winner(x)
                nb = self._neighborhood(c, sig)
                self.weights += lr * nb[:, :, np.newaxis] * (x - self.weights)
        return self

    def predict(self, X):
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            c = self.winner(x)
            labels[i] = c[0] * self.y + c[1]
        return labels

def main():
    output_dir = os.path.join("result")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating dataset...")
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    print("Running KMeans...")
    start_t = time.time()
    custom_kmeans = CustomKMeans(n_clusters=3).fit(X)
    custom_kmeans_train = time.time() - start_t

    start_t = time.time()
    custom_kmeans_labels = custom_kmeans.predict(X)
    custom_kmeans_exec = time.time() - start_t
    custom_kmeans_metrics = evaluate_clustering(y, custom_kmeans_labels)

    start_t = time.time()
    sk_kmeans = sklearn_KMeans(n_clusters=3, random_state=42).fit(X)
    sk_kmeans_train = time.time() - start_t

    start_t = time.time()
    sk_kmeans_labels = sk_kmeans.predict(X)
    sk_kmeans_exec = time.time() - start_t
    sk_kmeans_metrics = evaluate_clustering(y, sk_kmeans_labels)

    plot_comparison(
        X,
        custom_kmeans_labels,
        sk_kmeans_labels,
        f"Custom KMeans ({custom_kmeans_train:.4f}s)",
        f"Sklearn KMeans ({sk_kmeans_train:.4f}s)",
        os.path.join(output_dir, "kmeans_comparison.png"),
    )
    print_report("Custom KMeans", custom_kmeans_metrics, custom_kmeans_train, custom_kmeans_exec)
    print_report("Sklearn KMeans", sk_kmeans_metrics, sk_kmeans_train, sk_kmeans_exec)

    print("Running CLIQUE...")
    start_t = time.time()
    custom_clique = CustomCLIQUE(intervals=10, threshold=3).fit(X)
    custom_clique_train = time.time() - start_t

    start_t = time.time()
    custom_clique_labels = np.copy(custom_clique.labels_)
    custom_clique_exec = time.time() - start_t
    custom_clique_metrics = evaluate_clustering(y, custom_clique_labels)

    start_t = time.time()
    try:
        clique_instance = pyclique(X.tolist(), 10, 3)
        clique_instance.process()
    except OSError:
        clique_instance = pyclique(X.tolist(), 10, 3, ccore=False)
        clique_instance.process()
    clique_clusters = clique_instance.get_clusters()
    sk_clique_train = time.time() - start_t

    start_t = time.time()
    clique_labels = np.zeros(X.shape[0]) - 1
    for cid, cluster in enumerate(clique_clusters):
        for idx in cluster:
            clique_labels[idx] = cid
    sk_clique_exec = time.time() - start_t
    sk_clique_metrics = evaluate_clustering(y, clique_labels)

    plot_comparison(
        X,
        custom_clique_labels,
        clique_labels,
        f"Custom CLIQUE ({custom_clique_train:.4f}s)",
        f"Pyclustering CLIQUE ({sk_clique_train:.4f}s)",
        os.path.join(output_dir, "clique_comparison.png"),
    )
    print_report(
        "Custom CLIQUE",
        custom_clique_metrics,
        custom_clique_train,
        custom_clique_exec,
        note="labels computed in fit",
    )
    print_report(
        "Pyclustering CLIQUE",
        sk_clique_metrics,
        sk_clique_train,
        sk_clique_exec,
        note="labels computed in process",
    )

    print("Running DBSCAN...")
    start_t = time.time()
    custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=5).fit(X)
    custom_dbscan_train = time.time() - start_t

    start_t = time.time()
    custom_dbscan_labels = np.copy(custom_dbscan.labels_)
    custom_dbscan_exec = time.time() - start_t
    custom_dbscan_metrics = evaluate_clustering(y, custom_dbscan_labels)

    start_t = time.time()
    sk_dbscan = sklearn_DBSCAN(eps=0.5, min_samples=5).fit(X)
    sk_dbscan_train = time.time() - start_t

    start_t = time.time()
    sk_dbscan_labels = np.copy(sk_dbscan.labels_)
    sk_dbscan_exec = time.time() - start_t
    sk_dbscan_metrics = evaluate_clustering(y, sk_dbscan_labels)

    plot_comparison(
        X,
        custom_dbscan_labels,
        sk_dbscan_labels,
        f"Custom DBSCAN ({custom_dbscan_train:.4f}s)",
        f"Sklearn DBSCAN ({sk_dbscan_train:.4f}s)",
        os.path.join(output_dir, "dbscan_comparison.png"),
    )
    print_report(
        "Custom DBSCAN",
        custom_dbscan_metrics,
        custom_dbscan_train,
        custom_dbscan_exec,
        note="labels computed in fit",
    )
    print_report(
        "Sklearn DBSCAN",
        sk_dbscan_metrics,
        sk_dbscan_train,
        sk_dbscan_exec,
        note="labels computed in fit",
    )

    print("Running SOM...")
    start_t = time.time()
    custom_som = CustomSOM(x=2, y=2, input_len=X.shape[1], num_iteration=100).fit(X)
    custom_som_train = time.time() - start_t

    start_t = time.time()
    custom_som_labels = custom_som.predict(X)
    custom_som_exec = time.time() - start_t
    custom_som_metrics = evaluate_clustering(y, custom_som_labels)

    start_t = time.time()
    som = MiniSom(2, 2, X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 100)
    sk_som_train = time.time() - start_t

    start_t = time.time()
    sk_som_labels = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        c = som.winner(x)
        sk_som_labels[i] = c[0] * 2 + c[1]
    sk_som_exec = time.time() - start_t
    sk_som_metrics = evaluate_clustering(y, sk_som_labels)

    plot_comparison(
        X,
        custom_som_labels,
        sk_som_labels,
        f"Custom SOM ({custom_som_train:.4f}s)",
        f"MiniSom ({sk_som_train:.4f}s)",
        os.path.join(output_dir, "som_comparison.png"),
    )
    print_report("Custom SOM", custom_som_metrics, custom_som_train, custom_som_exec)
    print_report("MiniSom", sk_som_metrics, sk_som_train, sk_som_exec)
    rows = [
        [
            "Custom KMeans",
            f"{custom_kmeans_metrics['accuracy']:.4f}",
            f"{custom_kmeans_metrics['precision']:.4f}",
            f"{custom_kmeans_metrics['recall']:.4f}",
            f"{custom_kmeans_metrics['f1']:.4f}",
            f"{custom_kmeans_metrics['coverage']:.2f}",
            f"{custom_kmeans_train:.4f}",
            f"{custom_kmeans_exec:.4f}",
            "",
        ],
        [
            "Sklearn KMeans",
            f"{sk_kmeans_metrics['accuracy']:.4f}",
            f"{sk_kmeans_metrics['precision']:.4f}",
            f"{sk_kmeans_metrics['recall']:.4f}",
            f"{sk_kmeans_metrics['f1']:.4f}",
            f"{sk_kmeans_metrics['coverage']:.2f}",
            f"{sk_kmeans_train:.4f}",
            f"{sk_kmeans_exec:.4f}",
            "",
        ],
        [
            "Custom CLIQUE",
            f"{custom_clique_metrics['accuracy']:.4f}",
            f"{custom_clique_metrics['precision']:.4f}",
            f"{custom_clique_metrics['recall']:.4f}",
            f"{custom_clique_metrics['f1']:.4f}",
            f"{custom_clique_metrics['coverage']:.2f}",
            f"{custom_clique_train:.4f}",
            f"{custom_clique_exec:.4f}",
            "labels computed in fit",
        ],
        [
            "Pyclustering CLIQUE",
            f"{sk_clique_metrics['accuracy']:.4f}",
            f"{sk_clique_metrics['precision']:.4f}",
            f"{sk_clique_metrics['recall']:.4f}",
            f"{sk_clique_metrics['f1']:.4f}",
            f"{sk_clique_metrics['coverage']:.2f}",
            f"{sk_clique_train:.4f}",
            f"{sk_clique_exec:.4f}",
            "labels computed in process",
        ],
        [
            "Custom DBSCAN",
            f"{custom_dbscan_metrics['accuracy']:.4f}",
            f"{custom_dbscan_metrics['precision']:.4f}",
            f"{custom_dbscan_metrics['recall']:.4f}",
            f"{custom_dbscan_metrics['f1']:.4f}",
            f"{custom_dbscan_metrics['coverage']:.2f}",
            f"{custom_dbscan_train:.4f}",
            f"{custom_dbscan_exec:.4f}",
            "labels computed in fit",
        ],
        [
            "Sklearn DBSCAN",
            f"{sk_dbscan_metrics['accuracy']:.4f}",
            f"{sk_dbscan_metrics['precision']:.4f}",
            f"{sk_dbscan_metrics['recall']:.4f}",
            f"{sk_dbscan_metrics['f1']:.4f}",
            f"{sk_dbscan_metrics['coverage']:.2f}",
            f"{sk_dbscan_train:.4f}",
            f"{sk_dbscan_exec:.4f}",
            "labels computed in fit",
        ],
        [
            "Custom SOM",
            f"{custom_som_metrics['accuracy']:.4f}",
            f"{custom_som_metrics['precision']:.4f}",
            f"{custom_som_metrics['recall']:.4f}",
            f"{custom_som_metrics['f1']:.4f}",
            f"{custom_som_metrics['coverage']:.2f}",
            f"{custom_som_train:.4f}",
            f"{custom_som_exec:.4f}",
            "",
        ],
        [
            "MiniSom",
            f"{sk_som_metrics['accuracy']:.4f}",
            f"{sk_som_metrics['precision']:.4f}",
            f"{sk_som_metrics['recall']:.4f}",
            f"{sk_som_metrics['f1']:.4f}",
            f"{sk_som_metrics['coverage']:.2f}",
            f"{sk_som_train:.4f}",
            f"{sk_som_exec:.4f}",
            "",
        ],
    ]
    write_metrics_table(rows, output_dir)

    print("Done! Check the result folder for outputs.")

if __name__ == "__main__":
    main()
