


from scipy import io
import numpy as np
import random, gc, pickle, sys, os
from multiprocessing import cpu_count
import pandas as pd
import copy


FILE_LAYER_1 = 'layer1.xlsx'
FILE_LAYER_2 = 'layer2.xlsx'
START_ROW_L1 = 353
START_ROW_L2 = 73




try:
    from Hotslib import (
        n_mnist_rearranging, learn, infer, signature_gen,
        histogram_accuracy, dataset_resize, spac_downsample
    )
except ModuleNotFoundError:
    print("Warning: 'Libs' folder not found. Trying to import from same directory...")
    from Hotslib import (
        n_mnist_rearranging, learn, infer, signature_gen,
        histogram_accuracy, dataset_resize, spac_downsample
    )



def load_curve_pool(filepath, start_row):
    print(f"Loading curve pool from: {filepath} (starting row {start_row})...")
    try:
        skip_rows_list = range(1, start_row - 1)
        df = pd.read_excel(filepath, header=0, skiprows=skip_rows_list)

        # NaN Check & Fix
        if df.isna().any().any():
            print(f"  Warning: NaN detected in {filepath}. Filling with 0.0.")
        df = df.fillna(0.0)

    except Exception as e:
        print(f"--- FATAL ERROR loading {filepath} --- \n{e}")
        sys.exit(1)

    # --- Process Time Axis ---
    time_x_raw_s = df.iloc[:, 0].values.astype(np.float32)
    time_relative_s = time_x_raw_s - time_x_raw_s[0]
    time_x_ms = time_relative_s * 1000.0  # Convert s -> ms
    dt_max_ms = np.max(time_x_ms)

    # --- Process Currents (Y-Axis) ---
    current_cols_df = df.iloc[:, 2:]
    all_values = current_cols_df.values.astype(np.float32)

    # Global Normalization
    GLOBAL_MIN = np.min(all_values)
    GLOBAL_MAX = np.max(all_values)
    GLOBAL_RANGE = GLOBAL_MAX - GLOBAL_MIN
    if GLOBAL_RANGE < 1e-9: GLOBAL_RANGE = 1e-9

    print(f"  Global Min: {GLOBAL_MIN:.2e}, Global Max: {GLOBAL_MAX:.2e}")

    # Vectorized Normalization
    all_curves_y = (current_cols_df.values.astype(np.float32).T - GLOBAL_MIN) / GLOBAL_RANGE
    all_curves_y = np.clip(all_curves_y, 0.0, 1.0)

    print(f"Loaded {all_curves_y.shape[0]} curves. Max time = {dt_max_ms:.2f} ms")
    return time_x_ms, all_curves_y, dt_max_ms






print("--- Initializing HOTS Experiment (Numba Accelerated) ---")

# Load Curves
l1_time, l1_curves, l1_dt_max = load_curve_pool(FILE_LAYER_1, START_ROW_L1)
l2_time, l2_curves, l2_dt_max = load_curve_pool(FILE_LAYER_2, START_ROW_L2)


custom_curve_layer = [
    (l1_time, l1_curves),
    (l2_time, l2_curves)
]
custom_dt_max = [l1_dt_max, l2_dt_max]

# Load N-MNIST
print("Loading N-MNIST dataset...")
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train = [len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test = [len(test_set_orig[label]) for label in range(len(train_set_orig))]


files_dataset_train = 6000
print(f"Train Set: Using {files_dataset_train} samples per class (Total ~{files_dataset_train * 10})")


num_labels = len(test_set_orig)

# Params
res_x_orig = 28
res_y_orig = 28

u = 1

layers = 2
surf_dim = [7, 3]
n_clusters = [32, 96]
n_jobs = cpu_count()
n_pol = [-1, 72]
n_batches = [10, 20]
n_batches_test = [2, 2]
n_runs = 1
seeds = [1, 2, 3, 4, 5]

# Dummy tau
tau_params = [{'mean': 1000, 'std': 1}, {'mean': 2000, 'std': 1}]

H_kmeansss = []
H_res = []


for run in range(n_runs):
    print(f"\n--- Starting Run {run + 1}/{n_runs} ---")
    run_euc_res = []
    run_norm_res = []
    run_kmeansss = []


    train_set = copy.deepcopy(train_set_orig)
    test_set = copy.deepcopy(test_set_orig)

    train_set = dataset_resize(train_set, res_x_orig, res_y_orig)
    test_set = dataset_resize(test_set, res_x_orig, res_y_orig)


    print("  [Info] Downsampling input from 28x28 to 14x14 (Factor 2)...")
    train_set = spac_downsample(train_set, 2)
    test_set = spac_downsample(test_set, 2)

    for label in range(num_labels):
        random.Random(seeds[run]).shuffle(train_set[label])
        random.Random(seeds[run]).shuffle(test_set[label])


    train_set = [train_set[label][:files_dataset_train] for label in range(num_labels)]


    print(f"  [Info] Train Set Size: {sum([len(l) for l in train_set])}")
    print(f"  [Info] Test Set Size:  {sum([len(l) for l in test_set])}")


    layer_res_x = 14
    layer_res_y = 14

    for layer in range(layers):
        print(f'########### LAYER_{layer} (Res: {layer_res_x}x{layer_res_y}) ###########')


        train_set, kmeans = learn(
            train_set, surf_dim[layer], layer_res_x, layer_res_y,
            tau_params[layer], n_clusters[layer], n_pol[layer], n_batches[layer], n_jobs,
            custom_curve=custom_curve_layer[layer],
            custom_dt_max=custom_dt_max[layer]
        )
        run_kmeansss.append(kmeans)


        train_set = spac_downsample(train_set, u)


        test_set = infer(
            test_set, surf_dim[layer], layer_res_x, layer_res_y,
            tau_params[layer], n_pol[layer], kmeans, n_batches_test[layer], n_jobs,
            custom_curve=custom_curve_layer[layer],
            custom_dt_max=custom_dt_max[layer]
        )
        test_set = spac_downsample(test_set, u)

        layer_res_x = layer_res_x // u
        layer_res_y = layer_res_y // u


        signatures, norm_signatures, svc, norm_svc = signature_gen(train_set, n_clusters[layer], n_jobs)
        test_signatures, test_norm_signatures, euc_accuracy, norm_euc_accuracy, euc_label, norm_euc_label = \
            histogram_accuracy(test_set, n_clusters[layer], signatures, norm_signatures, n_jobs)


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


        df_train = pd.DataFrame(train_hist_list)
        df_train['Label'] = train_labels

        df_train_norm = pd.DataFrame(train_norm_hist_list)
        df_train_norm['Label'] = train_labels

        df_test = pd.DataFrame(test_hist_list)
        df_test['Label'] = test_labels

        df_test_norm = pd.DataFrame(test_norm_hist_list)
        df_test_norm['Label'] = test_labels


        output_dir = "Features_14x14"
        os.makedirs(output_dir, exist_ok=True)

        df_train.to_excel(f"{output_dir}/layer{layer + 1}_train_hist_14x14_73_u1_32_96.xlsx", index=False)
        df_train_norm.to_excel(f"{output_dir}/layer{layer + 1}_train_hist_norm_14x14_73_u1_32_96.xlsx", index=False)
        df_test.to_excel(f"{output_dir}/layer{layer + 1}_test_hist_14x14_73_u1_32_96.xlsx", index=False)
        df_test_norm.to_excel(f"{output_dir}/layer{layer + 1}_test_hist_norm_14x14_73_u1_32_96.xlsx", index=False)

        print(f"Saved Raw and Norm histograms with Labels to {output_dir}/")

        run_euc_res.append(euc_accuracy)
        run_norm_res.append(norm_euc_accuracy)
        print('Euclidean accuracy: ' + str(euc_accuracy) + '%')
        print('Normalized euclidean accuracy: ' + str(norm_euc_accuracy) + '%')
        gc.collect()

    H_kmeansss.append(run_kmeansss)
    H_res.append(run_norm_res)

filename = 'Results/test_result_14x14_73_u1_32_96.pkl'
os.makedirs('Results', exist_ok=True)
with open(filename, 'wb') as f:
    pickle.dump([H_kmeansss, H_res], f)

print("Done.")