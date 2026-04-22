# -*- coding: utf-8 -*-
import argparse
import ast
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sklearn.preprocessing as skp

import dbscan as db
import features as ft
import kmeans as km
import linkage as lk


RESULT_DIR = "results"


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_time_series(path, separator=","):
    print("开始读入时序数据...")
    start_t = time.time()
    df = pd.read_table(path, header=None, sep=separator)
    data = df.to_numpy(dtype=np.float64)
    x = data[:, 1:]
    y = data[:, 0]
    x = skp.StandardScaler().fit_transform(x)
    end_t = time.time()
    print("时序数据读入结束。耗时%.3f秒。" % (end_t - start_t))
    return x, y


def load_dots(path, separator=r"\s+"):
    print("开始读入二维点数据...")
    start_t = time.time()
    df = pd.read_table(path, header=None, sep=separator)
    data = df.to_numpy(dtype=np.float64)
    end_t = time.time()
    print("二维点数据读入结束。耗时%.3f秒。" % (end_t - start_t))
    return data


def load_uci_wine(path):
    print("开始读入UCI wine质量数据...")
    start_t = time.time()

    if os.path.isdir(path):
        red_path = os.path.join(path, "winequality-red.csv")
        white_path = os.path.join(path, "winequality-white.csv")
        red = pd.read_csv(red_path, sep=";")
        white = pd.read_csv(white_path, sep=";")
        data = pd.concat([red, white], axis=0, ignore_index=True)
    else:
        data = pd.read_csv(path, sep=";")

    x = data.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = data.iloc[:, -1].to_numpy()
    x = skp.StandardScaler().fit_transform(x)

    end_t = time.time()
    print("UCI数据读入结束。耗时%.3f秒。" % (end_t - start_t))
    return x, y


def load_data(path, data_type):
    x_train = y_train = None
    if data_type == "dot":
        x_train = load_dots(path)
        x_train, _meta = ft.dotFeature(x_train, None)
    elif data_type == "timeseries":
        x_ts, y_train = load_time_series(path)
        x_train, _meta = ft.timeSeriesFeature(x_ts, None)
    elif data_type == "uci":
        x_train, y_train = load_uci_wine(path)
    else:
        raise ValueError("%s是未知数据类型。" % data_type)
    return x_train, y_train


def nmi(y_true, y_pred):
    y_true = np.squeeze(y_true)
    return skm.normalized_mutual_info_score(y_true, y_pred)


def ari(y_true, y_pred):
    y_true = np.squeeze(y_true)
    return skm.adjusted_rand_score(y_true, y_pred)


def show_dots(x, labels, output_path, plot_title, method_desc):
    unique_labels = np.unique(labels)
    fig, ax = plt.subplots(1, figsize=(9, 6))
    cmap = plt.cm.get_cmap("tab20", max(1, unique_labels.size))
    for idx, lb in enumerate(unique_labels):
        color = "#666666" if lb == -1 else cmap(idx)
        cluster_tag = "noise" if lb == -1 else "cluster"
        label_text = "%s=%s | %s" % (cluster_tag, lb, method_desc)
        ax.scatter(x[labels == lb, 0], x[labels == lb, 1], s=4, c=[color], label=label_text, alpha=0.85)
    ax.set_title(plot_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_with_profile(train_func, x, settings):
    tracemalloc.start()
    start_t = time.perf_counter()
    cluster = train_func(x, *settings)
    elapsed = time.perf_counter() - start_t
    _curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return cluster, elapsed, peak / (1024 * 1024)


def evaluate_result(x, y, cluster, model_name, run_name, data_type, data_path, settings):
    print("-" * 70)
    print("%s [%s]" % (model_name, run_name))
    if y is None or len(y) == 0:
        ensure_result_dir()
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        safe_settings = str(settings).replace(" ", "").replace("\"", "")
        output_name = "%s_%s_%s_%s.png" % (model_name, run_name, data_name, data_type)
        output_path = os.path.join(RESULT_DIR, output_name)
        method_desc = "%s/%s settings=%s data=%s" % (model_name, run_name, safe_settings, data_type)
        plot_title = "Cluster Result | %s | samples=%d | clusters=%d" % (
            method_desc,
            x.shape[0],
            np.unique(cluster.labels_).size,
        )
        show_dots(x, cluster.labels_, output_path, plot_title, method_desc)
        print("无标签数据。聚类可视化已保存到: %s" % output_path)
    else:
        pred_label = cluster.labels_
        print("ARI=%.4f, NMI=%.4f" % (ari(y, pred_label), nmi(y, pred_label)))


def run_model(model_name, x, y, settings, compare, data_type, data_path):
    if model_name == "kmeans":
        train_func = km.train
        base_settings = settings if settings is not None else (5, "random")
    elif model_name == "linkage":
        train_func = lk.train
        base_settings = settings if settings is not None else (5, "single")
    elif model_name == "dbscan":
        train_func = db.train
        base_settings = settings if settings is not None else (0.8, 10)
    else:
        raise ValueError("未知模型: %s" % model_name)

    results = []
    if model_name in ("kmeans", "linkage") and compare:
        manual_settings = tuple(base_settings) + ("manual",)
        skl_settings = tuple(base_settings) + ("sklearn",)

        manual_cluster, manual_time, manual_mem = run_with_profile(train_func, x, manual_settings)
        evaluate_result(x, y, manual_cluster, model_name, "manual", data_type, data_path, manual_settings)
        results.append(("manual", manual_time, manual_mem))

        skl_cluster, skl_time, skl_mem = run_with_profile(train_func, x, skl_settings)
        evaluate_result(x, y, skl_cluster, model_name, "sklearn", data_type, data_path, skl_settings)
        results.append(("sklearn", skl_time, skl_mem))
    else:
        if model_name == "dbscan" and len(base_settings) <= 2:
            base_settings = tuple(base_settings) + ("sklearn",)
        cluster, t_cost, mem_cost = run_with_profile(train_func, x, base_settings)
        run_name = getattr(cluster, "implementation", "single_run")
        evaluate_result(x, y, cluster, model_name, run_name, data_type, data_path, base_settings)
        results.append((run_name, t_cost, mem_cost))

    print("\n时间/内存统计:")
    for name, t_cost, mem_cost in results:
        print("  %s: time=%.4fs, peak_mem=%.2fMB" % (name, t_cost, mem_cost))


def run_sweep(model_name, x, y, param_grid, data_type, data_path):
    print("开始参数扫描，共%d组参数。" % len(param_grid))
    for settings in param_grid:
        print("\n参数: %s" % (settings,))
        run_model(model_name, x, y, settings=settings, compare=False, data_type=data_type, data_path=data_path)


def sample_if_needed(x, y, max_samples, random_state=42):
    if max_samples is None or x.shape[0] <= max_samples:
        return x, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    x_s = x[idx]
    y_s = y[idx] if y is not None else None
    print("数据量较大，已随机采样到%d条以便完成层次聚类实验。" % max_samples)
    return x_s, y_s


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering Lab Runner")
    parser.add_argument("-d", "--data", required=True, help="数据文件或目录路径")
    parser.add_argument("-t", "--type", required=True, choices=["dot", "timeseries", "uci"], help="数据类型")
    parser.add_argument("-m", "--model", default="all", choices=["kmeans", "linkage", "dbscan", "all"], help="聚类算法")
    parser.add_argument("-s", "--settings", default=None, help="算法参数，如 '(5, \"random\")'")
    parser.add_argument("--compare", action="store_true", default=True, help="对kmeans/linkage执行手写vs sklearn对比")
    parser.add_argument("--no-compare", dest="compare", action="store_false", help="关闭手写vs sklearn对比")
    parser.add_argument("--sweep", action="store_true", help="是否执行参数扫描")
    parser.add_argument("--max-linkage-samples", type=int, default=1200, help="linkage默认最大样本数")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = ast.literal_eval(args.settings) if args.settings else None
    ensure_result_dir()

    x_train, y_train = load_data(args.data, args.type)

    model_list = ["kmeans", "linkage", "dbscan"] if args.model == "all" else [args.model]
    for model_name in model_list:
        x_use, y_use = x_train, y_train
        if model_name == "linkage":
            x_use, y_use = sample_if_needed(x_train, y_train, args.max_linkage_samples)

        print("\n" + "=" * 70)
        print("当前算法: %s" % model_name)
        print("=" * 70)

        if args.sweep:
            if model_name == "kmeans":
                grid = [(3, "random", "manual"), (5, "random", "manual"), (8, "random", "manual")]
            elif model_name == "linkage":
                grid = [(3, "single", "manual"), (5, "single", "manual"), (8, "single", "manual")]
            else:
                grid = [(0.4, 5, "sklearn"), (0.8, 10, "sklearn"), (1.2, 15, "sklearn")]
            run_sweep(model_name, x_use, y_use, grid, data_type=args.type, data_path=args.data)
        else:
            run_model(
                model_name,
                x_use,
                y_use,
                settings=settings,
                compare=args.compare,
                data_type=args.type,
                data_path=args.data,
            )


if __name__ == "__main__":
    main()
