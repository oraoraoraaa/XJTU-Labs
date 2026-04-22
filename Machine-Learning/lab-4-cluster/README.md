# Clustering Lab (Manual + sklearn Comparison)

## What is implemented

This lab now supports clustering on three dataset types:

- `dot`: 2D point datasets (`dots/*.dat`)
- `timeseries`: labeled time-series dataset (`timeSeries/Plane_TRAIN.txt`)
- `uci`: UCI Wine Quality dataset (`wine+quality/`)

Algorithms:

- `kmeans`
  - manual implementation (from scratch)
  - sklearn baseline
- `linkage` (single-link / SLINK)
  - manual SLINK implementation (from scratch)
  - sklearn baseline
- `dbscan`
  - sklearn implementation

What the runner reports:

- clustering quality on labeled data: `ARI`, `NMI`
- runtime and peak memory usage
- clustered scatter plots for unlabeled dot data (`*.png`)

## Main files

- `clustering.py`: experiment entry, data loading, evaluation, compare/sweep
- `kmeans.py`: manual + sklearn KMeans
- `linkage.py`: manual SLINK + sklearn Agglomerative
- `dbscan.py`: DBSCAN wrapper
- `features.py`: dot/time-series feature processing

## How to run

Run from `Machine-Learning/lab-cluster`:

```bash
python clustering.py -d <data_path> -t <dot|timeseries|uci> -m <kmeans|linkage|dbscan|all>
```

### 1) 2D dot data

```bash
python clustering.py -d dots/clusterData1.10k.dat -t dot -m all
```

### 2) Time-series data

```bash
python clustering.py -d timeSeries/Plane_TRAIN.txt -t timeseries -m all
```

### 3) UCI wine quality data

```bash
python clustering.py -d wine+quality -t uci -m all
```

## Useful options

- custom settings:

```bash
python clustering.py -d timeSeries/Plane_TRAIN.txt -t timeseries -m kmeans -s '(5, "random")'
python clustering.py -d dots/clusterData2.8k.dat -t dot -m dbscan -s '(1.0, 12)'
```

- parameter sweep:

```bash
python clustering.py -d wine+quality -t uci -m all --sweep
```

- disable manual-vs-sklearn comparison (single run):

```bash
python clustering.py -d wine+quality -t uci -m kmeans --no-compare
```

- cap sample size for linkage (to keep runtime reasonable on large datasets):

```bash
python clustering.py -d dots/clusterData1.10k.dat -t dot -m linkage --max-linkage-samples 1200
```

## Notes

- Manual linkage currently supports `single` linkage only (SLINK), as required.
- For very large dot datasets, linkage is automatically sampled (configurable) to avoid excessive runtime.
