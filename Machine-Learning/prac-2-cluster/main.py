import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn.cluster import DBSCAN as sklearn_DBSCAN
from minisom import MiniSom
from pyclustering.cluster.clique import clique as pyclique

# Custom KMeans
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_idxs = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idxs]
        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self

# Generate Data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Custom vs Sklearn KMeans
start_t = time.time()
custom_kmeans = CustomKMeans(n_clusters=3).fit(X)
custom_time = time.time() - start_t

start_t = time.time()
sk_kmeans = sklearn_KMeans(n_clusters=3, random_state=42).fit(X)
sk_time = time.time() - start_t

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_)
plt.title(f'Custom KMeans ({custom_time:.4f}s)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sk_kmeans.labels_)
plt.title(f'Sklearn KMeans ({sk_time:.4f}s)')
plt.savefig('kmeans_comparison.png')

# Custom CLIQUE (Simplified Grid-based clustering)
class CustomCLIQUE:
    def __init__(self, intervals, threshold):
        self.intervals = intervals
        self.threshold = threshold
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0]) - 1
        # Simple 2D grid
        min_x, max_x = np.min(X[:,0]), np.max(X[:,0])
        min_y, max_y = np.min(X[:,1]), np.max(X[:,1])
        x_bins = np.linspace(min_x, max_x, self.intervals + 1)
        y_bins = np.linspace(min_y, max_y, self.intervals + 1)
        
        grid = {}
        for i, point in enumerate(X):
            idx_x = np.digitize(point[0], x_bins) - 1
            idx_y = np.digitize(point[1], y_bins) - 1
            cell = (idx_x, idx_y)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)
            
        cluster_id = 0
        for cell, points in grid.items():
            if len(points) >= self.threshold:
                for p in points:
                    self.labels_[p] = cluster_id
                cluster_id += 1
        return self

# Custom vs Pyclustering CLIQUE
start_t = time.time()
custom_clique = CustomCLIQUE(intervals=10, threshold=3).fit(X)
custom_clique_time = time.time() - start_t

start_t = time.time()
# pyclustering clique
clique_instance = pyclique(X.tolist(), 10, 3)
clique_instance.process()
clique_clusters = clique_instance.get_clusters()
sk_clique_time = time.time() - start_t

# Convert pyclustering output to labels
clique_labels = np.zeros(X.shape[0]) - 1
for cid, cluster in enumerate(clique_clusters):
    for idx in cluster:
        clique_labels[idx] = cid

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_clique.labels_)
plt.title(f'Custom CLIQUE ({custom_clique_time:.4f}s)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=clique_labels)
plt.title(f'Pyclustering CLIQUE ({sk_clique_time:.4f}s)')
plt.savefig('clique_comparison.png')

# Custom DBSCAN
class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels_ = np.full(X.shape[0], -1)  # -1 is noise
        cluster_id = 0
        visited = np.zeros(X.shape[0], dtype=bool)

        for i in range(X.shape[0]):
            if visited[i]:
                continue
            visited[i] = True
            
            # Find neighbors
            neighbors = np.where(np.linalg.norm(X - X[i], axis=1) <= self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                # Noise
                pass
            else:
                # Expand cluster
                self.labels_[i] = cluster_id
                
                # We need to iterate through neighbors (which can grow)
                seeds = list(neighbors)
                if i in seeds:
                    seeds.remove(i)
                
                j = 0
                while j < len(seeds):
                    neighbor_idx = seeds[j]
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        new_neighbors = np.where(np.linalg.norm(X - X[neighbor_idx], axis=1) <= self.eps)[0]
                        if len(new_neighbors) >= self.min_samples:
                            # Add neighbors without duplicates efficiently
                            for n in new_neighbors:
                                if n not in seeds:
                                    seeds.append(n)
                    
                    if self.labels_[neighbor_idx] == -1:
                        self.labels_[neighbor_idx] = cluster_id
                    j += 1
                cluster_id += 1
        return self

# Custom vs Sklearn DBSCAN
start_t = time.time()
custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=5).fit(X)
custom_dbscan_time = time.time() - start_t

start_t = time.time()
sk_dbscan = sklearn_DBSCAN(eps=0.5, min_samples=5).fit(X)
sk_dbscan_time = time.time() - start_t

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=custom_dbscan.labels_)
plt.title(f'Custom DBSCAN ({custom_dbscan_time:.4f}s)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=sk_dbscan.labels_)
plt.title(f'Sklearn DBSCAN ({sk_dbscan_time:.4f}s)')
plt.savefig('dbscan_comparison.png')

# Custom SOM
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
        xx, yy = np.meshgrid(ax, ay, indexing='ij')
        return np.exp(-((xx - c[0])**2 + (yy - c[1])**2) / d)
        
    def winner(self, x):
        diff = self.weights - x
        sq_dist = np.sum(diff**2, axis=2)
        return np.unravel_index(np.argmin(sq_dist), sq_dist.shape)
        
    def fit_predict(self, X):
        self.weights = np.random.rand(self.x, self.y, X.shape[1]) * \
                       (np.max(X, axis=0) - np.min(X, axis=0)) + np.min(X, axis=0)
        
        for t in range(self.num_iteration):
            lr = self.learning_rate * (1 - t / self.num_iteration)
            sig = self.sigma * (1 - t / self.num_iteration)
            
            for x in X:
                c = self.winner(x)
                nb = self._neighborhood(c, sig)
                self.weights += lr * nb[:, :, np.newaxis] * (x - self.weights)
                
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            c = self.winner(x)
            labels[i] = c[0] * self.y + c[1]
        return labels

def main():
    # 1. Generate Data consistently
    print("Generating dataset...")
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    
    # --- KMeans ---
    print("Running KMeans...")
    start_t = time.time()
    custom_kmeans = CustomKMeans(n_clusters=3).fit(X)
    custom_kmeans_time = time.time() - start_t

    start_t = time.time()
    sk_kmeans = sklearn_KMeans(n_clusters=3, random_state=42).fit(X)
    sk_kmeans_time = time.time() - start_t

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_)
    plt.title(f'Custom KMeans ({custom_kmeans_time:.4f}s)')
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sk_kmeans.labels_)
    plt.title(f'Sklearn KMeans ({sk_kmeans_time:.4f}s)')
    plt.savefig('kmeans_comparison.png')
    
    # --- CLIQUE ---
    print("Running CLIQUE...")
    start_t = time.time()
    custom_clique = CustomCLIQUE(intervals=10, threshold=3).fit(X)
    custom_clique_time = time.time() - start_t

    start_t = time.time()
    clique_instance = pyclique(X.tolist(), 10, 3)
    clique_instance.process()
    clique_clusters = clique_instance.get_clusters()
    sk_clique_time = time.time() - start_t

    clique_labels = np.zeros(X.shape[0]) - 1
    for cid, cluster in enumerate(clique_clusters):
        for idx in cluster:
            clique_labels[idx] = cid

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=custom_clique.labels_)
    plt.title(f'Custom CLIQUE ({custom_clique_time:.4f}s)')
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=clique_labels)
    plt.title(f'Pyclustering CLIQUE ({sk_clique_time:.4f}s)')
    plt.savefig('clique_comparison.png')
    
    # --- DBSCAN ---
    print("Running DBSCAN...")
    start_t = time.time()
    custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=5).fit(X)
    custom_dbscan_time = time.time() - start_t

    start_t = time.time()
    sk_dbscan = sklearn_DBSCAN(eps=0.5, min_samples=5).fit(X)
    sk_dbscan_time = time.time() - start_t

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=custom_dbscan.labels_)
    plt.title(f'Custom DBSCAN ({custom_dbscan_time:.4f}s)')
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sk_dbscan.labels_)
    plt.title(f'Sklearn DBSCAN ({sk_dbscan_time:.4f}s)')
    plt.savefig('dbscan_comparison.png')
    
    # --- SOM ---
    print("Running SOM...")
    start_t = time.time()
    custom_som_labels = CustomSOM(x=2, y=2, input_len=X.shape[1], num_iteration=100).fit_predict(X)
    custom_som_time = time.time() - start_t

    start_t = time.time()
    som = MiniSom(2, 2, X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 100)
    sk_som_labels = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        c = som.winner(x)
        sk_som_labels[i] = c[0] * 2 + c[1]
    sk_som_time = time.time() - start_t

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=custom_som_labels)
    plt.title(f'Custom SOM ({custom_som_time:.4f}s)')
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sk_som_labels)
    plt.title(f'MiniSom ({sk_som_time:.4f}s)')
    plt.savefig('som_comparison.png')
    
    print("Done! Check the *_comparison.png files.")

if __name__ == "__main__":
    main()
