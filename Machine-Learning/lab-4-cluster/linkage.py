# -*- coding: utf-8 -*-
from types import SimpleNamespace
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import time


ModelName = "凝聚法层次聚类算法"


def _slink_edges(data):
    n_samples = data.shape[0]
    pi = np.arange(n_samples, dtype=np.int32)
    lam = np.full(n_samples, np.inf, dtype=np.float64)

    for i in range(1, n_samples):
        m = np.linalg.norm(data[i] - data[:i], axis=1)
        pi[i] = i
        lam[i] = np.inf
        for j in range(i):
            pj = pi[j]
            if lam[j] >= m[j]:
                m[pj] = min(m[pj], lam[j])
                lam[j] = m[j]
                pi[j] = i
            else:
                m[pj] = min(m[pj], m[j])
        for j in range(i):
            if lam[j] >= lam[pi[j]]:
                pi[j] = i

    edges = [(i, int(pi[i]), float(lam[i])) for i in range(n_samples) if pi[i] != i]
    return edges


def _manual_slink(data, n_clusters):
    n_samples = data.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError("n_clusters must be in [1, n_samples].")

    edges = _slink_edges(data)
    edges.sort(key=lambda x: x[2])

    parent = np.arange(n_samples, dtype=np.int32)
    rank = np.zeros(n_samples, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Keep n_clusters components by adding only (n - n_clusters) shortest edges.
    for a, b, _dist in edges[: max(0, n_samples - n_clusters)]:
        union(a, b)

    roots = np.array([find(i) for i in range(n_samples)], dtype=np.int32)
    unique_roots, labels = np.unique(roots, return_inverse=True)

    return SimpleNamespace(
        labels_=labels,
        n_clusters_=unique_roots.size,
        implementation="manual",
        linkage="single",
    )


def train(data, *args):
    print("开始%s过程..." % ModelName)
    startT = time.time()

    n_clusters = args[0] if len(args) > 0 else 5
    linkage_name = args[1] if len(args) > 1 else "single"
    implementation = args[2].lower() if len(args) > 2 else "manual"

    if implementation == "manual":
        if linkage_name != "single":
            raise ValueError("Manual implementation only supports 'single' linkage (SLINK).")
        cluster = _manual_slink(data, n_clusters=n_clusters)
    elif implementation == "sklearn":
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_name).fit(data)
        cluster.implementation = "sklearn"
    else:
        raise ValueError("implementation must be one of {'manual', 'sklearn'}")

    endT = time.time()
    print("%s过程结束。处理了%d个数据点，耗时%f秒。" % (ModelName, data.shape[0], (endT - startT)))
    return cluster

