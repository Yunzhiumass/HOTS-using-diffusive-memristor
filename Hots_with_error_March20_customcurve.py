
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patched main that imports the custom-curve lib and passes per-layer curves.
Only the time-surface generation path is changed.
"""

from scipy import io
import numpy as np
import random, gc, pickle
from multiprocessing import cpu_count
import pandas as pd

# IMPORTANT: use the patched library
try:
    from Libs.Hotslib_with_error_customcurve import (
        n_mnist_rearranging, learn, infer, signature_gen,
        histogram_accuracy, dataset_resize, spac_downsample, build_layer_curve
    )
except ModuleNotFoundError:
    from Hotslib_with_error_customcurve import (
        n_mnist_rearranging, learn, infer, signature_gen,
        histogram_accuracy, dataset_resize, spac_downsample, build_layer_curve
    )

# --- Define two raw curves (copied from v8e example) ---
_raw_curve1 = np.array([
  1.0, 0.902993, 0.815396, 0.736406, 0.665259, 0.601229, 0.543636, 0.491846, 0.445275, 0.403381,
  0.365663, 0.331661, 0.300956, 0.273163, 0.24793, 0.224938, 0.203894, 0.184531, 0.166607, 0.149906,
  0.134233, 0.119413, 0.105284, 0.091699, 0.078525, 0.06564, 0.05293, 0.040287, 0.027608, 0.014791,
  0.001739, 0.112547, 0.099512, 0.087066, 0.075158, 0.063741, 0.052771, 0.042208, 0.032016, 0.02216,
  0.012611, 0.003343, 0.025753, 0.019222, 0.013184, 0.0076, 0.002437, 0.010134, 0.009151, 0.008263,
  0.007462, 0.006738
], dtype=np.float32)

_raw_curve2 = np.array([
  1.0, 0.960005, 0.92161, 0.884704, 0.849366, 0.815498, 0.783119, 0.752172, 0.722596, 0.694336,
  0.667344, 0.641569, 0.616963, 0.593478, 0.571071, 0.5497, 0.529323, 0.509902, 0.491399, 0.473779,
  0.45701, 0.441058, 0.425892, 0.411483, 0.397802, 0.384823, 0.372518, 0.360864, 0.349836, 0.339412,
  0.329571, 0.320293, 0.311559, 0.303351, 0.295652, 0.288446, 0.281718, 0.275453, 0.269637, 0.264258,
  0.259302, 0.254758, 0.250615, 0.246862, 0.24349, 0.240489, 0.237852, 0.23557, 0.233636,
  0.159337, 0.152964, 0.146847, 0.140973, 0.135335, 0.0076, 0.002437, 0.010134, 0.009151, 0.008263,
  0.007462, 0.006738
], dtype=np.float32)

# Build per-layer curves (you can change lengths to match desired temporal window)
custom_curve_layer = [
    build_layer_curve(_raw_curve1, target_len=50, smooth_win=5),
    build_layer_curve(_raw_curve2, target_len=100, smooth_win=7),
]
custom_dt_max = [1.0, 1.0]  # map delta_t ∈ [0,1] to full curve range

# %% Data loading and parameters setting
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train = [len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test = [len(test_set_orig[label]) for label in range(len(train_set_orig))]

# using a subset of N-MNIST to lower memory usage (unchanged defaults)
files_dataset_train = min(n_recording_labels_train) // 1
files_dataset_test = min(n_recording_labels_test) // 1
num_labels = len(test_set_orig)

# N-MNIST resolution
res_x = 28
res_y = 28

# Network parameters (unchanged, except we will pass custom curves)
layers = 2
surf_dim = [7, 3]
n_clusters = [72, 512]
n_jobs = cpu_count()
n_pol = [-1, 72]
n_batches = [10, 20]
n_batches_test = [10, 10]
u = 7
n_runs = 1
seeds = [1, 2, 3, 4, 5]

# tau is still needed when custom curve not used; here it's unused because we do pass the curves
tau_params = [
    {'mean': 1000, 'std': 1},
    {'mean': 2000, 'std': 1},
]

H_kmeansss = []
H_res = []

for run in range(n_runs):
    run_euc_res = []
    run_norm_res = []
    run_svc_res = []
    run_svc_norm_res = []
    run_kmeansss = []

    # shuffle & resize
    train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
    test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
    train_set_orig = dataset_resize(train_set_orig, res_x, res_y)
    test_set_orig = dataset_resize(test_set_orig, res_x, res_y)

    for label in range(num_labels):
        random.Random(seeds[run]).shuffle(train_set_orig[label])
        random.Random(seeds[run]).shuffle(test_set_orig[label])

    train_set = [train_set_orig[label][:files_dataset_train] for label in range(num_labels)]
    test_set = [test_set_orig[label][:files_dataset_test] for label in range(num_labels)]

    layer_res_x = res_x
    layer_res_y = res_y

    for layer in range(layers):
        print(f'########### LAYER_{layer} ###########')

        # Train (now with custom curve)
        train_set, kmeans = learn(
            train_set, surf_dim[layer], layer_res_x, layer_res_y,
            tau_params[layer], n_clusters[layer], n_pol[layer], n_batches[layer], n_jobs,
            custom_curve=custom_curve_layer[layer], custom_dt_max=custom_dt_max[layer]
        )
        run_kmeansss.append(kmeans)
        train_set = spac_downsample(train_set, u)

        # Test/infer (with the same custom curve)
        test_set = infer(
            test_set, surf_dim[layer], layer_res_x, layer_res_y,
            tau_params[layer], n_pol[layer], kmeans, n_batches_test[layer], n_jobs,
            custom_curve=custom_curve_layer[layer], custom_dt_max=custom_dt_max[layer]
        )
        test_set = spac_downsample(test_set, u)

        layer_res_x = layer_res_x // u
        layer_res_y = layer_res_y // u

        # signature generation & evaluation (unchanged)
        signatures, norm_signatures, svc, norm_svc = signature_gen(train_set, n_clusters[layer], n_jobs)
        test_signatures, test_norm_signatures, euc_accuracy, norm_euc_accuracy, euc_label, norm_euc_label = \
            histogram_accuracy(test_set, n_clusters[layer], signatures, norm_signatures, n_jobs)

        # Save histograms to Excel with labels (unchanged)
        train_hist_list, train_norm_hist_list, train_labels = [], [], []
        test_hist_list, test_norm_hist_list, test_labels = [], [], []

        for label_idx in range(len(train_set)):
            for rec in train_set[label_idx]:
                pols = rec[2]
                n_events = len(pols)
                hist_counts = np.bincount(pols, minlength=n_clusters[layer])
                norm_hist_counts = hist_counts / max(n_events, 1)
                train_hist_list.append(hist_counts)
                train_norm_hist_list.append(norm_hist_counts)
                train_labels.append(label_idx)

        for label_idx in range(len(test_set)):
            n_recs = len(test_set[label_idx])
            start_idx = sum([len(test_set[i]) for i in range(label_idx)])
            for i in range(n_recs):
                test_hist_list.append(test_signatures[start_idx + i])
                test_norm_hist_list.append(test_norm_signatures[start_idx + i])
                test_labels.append(label_idx)

        df_train_hist = pd.DataFrame(train_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_train_hist['Label'] = train_labels
        df_train_norm_hist = pd.DataFrame(train_norm_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_train_norm_hist['Label'] = train_labels
        df_test_hist = pd.DataFrame(test_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_test_hist['Label'] = test_labels
        df_test_norm_hist = pd.DataFrame(test_norm_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_test_norm_hist['Label'] = test_labels

        df_train_hist.to_excel(f"layer{layer + 1}_train_hist_clusters{n_clusters[layer]}.xlsx", index=False)
        df_train_norm_hist.to_excel(f"layer{layer + 1}_train_hist_norm_clusters{n_clusters[layer]}.xlsx", index=False)
        df_test_hist.to_excel(f"layer{layer + 1}_test_hist_clusters{n_clusters[layer]}.xlsx", index=False)
        df_test_norm_hist.to_excel(f"layer{layer + 1}_test_hist_norm_clusters{n_clusters[layer]}.xlsx", index=False)

        print(f'Layer {layer + 1} histograms (with labels) saved successfully in Excel format!')

        # record metrics
        run_euc_res.append(euc_accuracy)
        run_norm_res.append(norm_euc_accuracy)
        # svc metrics are computed in original code via recon_rates_svm; keep minimal here
        print('Euclidean accuracy: ' + str(euc_accuracy) + '%')
        print('Normalized euclidean accuracy: ' + str(norm_euc_accuracy) + '%')
        gc.collect()

    H_kmeansss.append(run_kmeansss)
    H_res.append(run_norm_res)

# Save minimal results (kmeans + accuracies)
filename = 'Results/test_result_new.pkl'
import os
os.makedirs('Results', exist_ok=True)
with open(filename, 'wb') as f:
    pickle.dump([H_kmeansss, H_res], f)

print("Done.")
