# -*- coding: utf-8 -*-
from types import SimpleNamespace
import numpy as np
from sklearn.cluster import KMeans
import time


ModelName = "KMeans聚类算法"


def _manual_kmeans(data, n_clusters, init="random", max_iter=300, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError("n_clusters must be in [1, n_samples].")

    if init == "random":
        centers = data[rng.choice(n_samples, size=n_clusters, replace=False)].copy()
    else:
        raise ValueError("Only 'random' initialization is supported in manual KMeans.")

    labels = np.zeros(n_samples, dtype=np.int32)
    for it in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for i in range(n_clusters):
            members = data[new_labels == i]
            if members.size == 0:
                new_centers[i] = data[rng.integers(0, n_samples)]
            else:
                new_centers[i] = members.mean(axis=0)

        center_shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels
        if center_shift <= tol:
            break

    inertia = float(np.sum((data - centers[labels]) ** 2))
    return SimpleNamespace(
        labels_=labels,
        cluster_centers_=centers,
        n_iter_=it + 1,
        inertia_=inertia,
        implementation="manual",
    )


def train(data, *args):
    print("开始%s过程..." % ModelName)
    startT = time.time()

    n_clusters = args[0] if len(args) > 0 else 5
    init = args[1] if len(args) > 1 else "random"
    implementation = args[2].lower() if len(args) > 2 else "manual"
    max_iter = int(args[3]) if len(args) > 3 else 300
    tol = float(args[4]) if len(args) > 4 else 1e-4
    random_state = int(args[5]) if len(args) > 5 else 42

    if implementation == "manual":
        cluster = _manual_kmeans(
            data,
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
    elif implementation == "sklearn":
        cluster = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            n_init=10,
            random_state=random_state,
        ).fit(data)
        cluster.implementation = "sklearn"
    else:
        raise ValueError("implementation must be one of {'manual', 'sklearn'}")

    endT = time.time()
    print("%s过程结束。处理了%d个数据点，耗时%f秒。" % (ModelName, data.shape[0], (endT - startT)))
    return cluster
