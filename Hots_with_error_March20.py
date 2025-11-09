#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:28:11 2020

@author: marcorax93

Script used for HOTS on N-MNIST,
this version uses batched-kmeans as a clustering algorithm, a subsampling layer
and two different classifiers (normalized histograms distance as in the original paper,
and a support vector machine trained on the histograms)

"""

from scipy import io
import numpy as np
import random, gc, pickle
from multiprocessing import cpu_count

from Libs.Hotslib_with_error import n_mnist_rearranging, learn, infer, signature_gen, \
    histogram_accuracy, dataset_resize, spac_downsample, recon_rates_svm

# %% Data loading and parameters setting

## Data loading
train_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/train_set.mat')['train_set'])
test_set_orig = n_mnist_rearranging(io.loadmat('N-MNIST/test_set.mat')['test_set'])
n_recording_labels_train = [len(train_set_orig[label]) for label in range(len(train_set_orig))]
n_recording_labels_test = [len(test_set_orig[label]) for label in range(len(train_set_orig))]

# using a subset of N-MNIST to lower memory usage
files_dataset_train = min(n_recording_labels_train) // 1
files_dataset_test = min(n_recording_labels_test) // 1
num_labels = len(test_set_orig)

# N-MNIST resolution
res_x = 28
res_y = 28

# Network parameters
layers = 2
surf_dim = [7, 3]  # lateral dimension of surfaces
n_clusters = [72, 512]
n_jobs = cpu_count()
n_pol = [-1, 72]  # input polarities of each layer (if -1 polarity is discarded.)
n_batches = [10, 20]  # batches of data for minibatchkmeans
n_batches_test = [10, 10]
u = 7  # Spatial downsample factor
n_runs = 1  # run the code multiple times on reshuffled data to better assest performance
seeds = [1, 2, 3, 4, 5]

# HOTS tau for first and second layer.
tau_params = [
    {'mean': 1000, 'std': 1},
    {'mean': 2000, 'std': 1},
]

# %%% BENCH HOTS

H_kmeansss = []  # save the networks layer for every run
H_res = []  # save the networks layer for every run

import pandas as pd
import numpy as np
from Libs.Hotslib_with_error import histogram_accuracy, signature_gen, learn, infer, spac_downsample

for run in range(n_runs):
    run_euc_res = []
    run_norm_res = []
    run_svc_res = []
    run_svc_norm_res = []
    run_kmeansss = []

    # Random data shuffling
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

        # 训练过程
        [train_set, kmeans] = learn(train_set, surf_dim[layer], layer_res_x,
                                    layer_res_y, tau_params[layer], n_clusters[layer],
                                    n_pol[layer], n_batches[layer], n_jobs)
        run_kmeansss.append(kmeans)
        train_set = spac_downsample(train_set, u)

        # 测试过程
        test_set = infer(test_set, surf_dim[layer], layer_res_x, layer_res_y,
                         tau_params[layer], n_pol[layer], kmeans, n_batches_test[layer], n_jobs)
        test_set = spac_downsample(test_set, u)

        layer_res_x = layer_res_x // u
        layer_res_y = layer_res_y // u

        # signature generation
        [signatures, norm_signatures, svc, norm_svc] = signature_gen(train_set, n_clusters[layer], n_jobs)
        [test_signatures, test_norm_signatures,
         euc_accuracy, norm_euc_accuracy,
         euc_label, norm_euc_label] = histogram_accuracy(test_set, n_clusters[layer],
                                                         signatures, norm_signatures, n_jobs)

        # 存放histograms和labels
        train_hist_list, train_norm_hist_list, train_labels = [], [], []
        test_hist_list, test_norm_hist_list, test_labels = [], [], []

        # 遍历训练集，生成每条记录的histogram（未归一化和归一化）
        for label_idx in range(len(train_set)):
            for rec in train_set[label_idx]:
                pols = rec[2]
                n_events = len(pols)

                hist_counts = np.bincount(pols, minlength=n_clusters[layer])
                norm_hist_counts = hist_counts / n_events

                train_hist_list.append(hist_counts)
                train_norm_hist_list.append(norm_hist_counts)
                train_labels.append(label_idx)  # 保存类别标签

        # 遍历测试集，生成每条记录的histogram
        for label_idx in range(len(test_set)):
            n_recs = len(test_set[label_idx])
            start_idx = sum([len(test_set[i]) for i in range(label_idx)])
            for i in range(n_recs):
                test_hist_list.append(test_signatures[start_idx + i])
                test_norm_hist_list.append(test_norm_signatures[start_idx + i])
                test_labels.append(label_idx)  # 保存类别标签

        # 转换为DataFrame，加上label列
        df_train_hist = pd.DataFrame(train_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_train_hist['Label'] = train_labels

        df_train_norm_hist = pd.DataFrame(train_norm_hist_list,
                                          columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_train_norm_hist['Label'] = train_labels

        df_test_hist = pd.DataFrame(test_hist_list, columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_test_hist['Label'] = test_labels

        df_test_norm_hist = pd.DataFrame(test_norm_hist_list,
                                         columns=[f'Cluster_{i}' for i in range(n_clusters[layer])])
        df_test_norm_hist['Label'] = test_labels

        # 保存为Excel文件（含label信息）
        df_train_hist.to_excel(f"layer{layer + 1}_train_hist_clusters{n_clusters[layer]}.xlsx", index=False)
        df_train_norm_hist.to_excel(f"layer{layer + 1}_train_hist_norm_clusters{n_clusters[layer]}.xlsx",
                                    index=False)
        df_test_hist.to_excel(f"layer{layer + 1}_test_hist_clusters{n_clusters[layer]}.xlsx", index=False)
        df_test_norm_hist.to_excel(f"layer{layer + 1}_test_hist_norm_clusters{n_clusters[layer]}.xlsx", index=False)

        print(f'Layer {layer + 1} histograms (with labels) saved successfully in Excel format!')







        run_euc_res.append(euc_accuracy)
        run_norm_res.append(norm_euc_accuracy)
        rec_rate_svc, rec_rate_norm_svc = recon_rates_svm(svc, norm_svc, test_signatures, test_norm_signatures,
                                                          test_set)

        run_svc_res.append(rec_rate_svc)
        run_svc_norm_res.append(rec_rate_norm_svc)

        print(run)
        print('Euclidean accuracy: ' + str(euc_accuracy) + '%')
        print('Normalized euclidean accuracy: ' + str(norm_euc_accuracy) + '%')
        print('Svc accuracy: ' + str(rec_rate_svc) + '%')
        print('Normalized Svc accuracy: ' + str(rec_rate_norm_svc) + '%')
        gc.collect()

    H_kmeansss.append(run_kmeansss)
    H_res.append(run_svc_res)




# %% Save run (Uncomment all code to save)
filename = 'Results/test_result_new.pkl'
with open(filename, 'wb') as f:
    pickle.dump([H_kmeansss, H_res], f)



# %% Load previous results
# filename='Results/test_result_new.pkl'
# with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
#     H_kmeansss, H_res = pickle.load(f)

# %% Results:
# Layer 1 mean:
H1 = np.mean(np.array(H_res)[:, 0])
# Layer 2 mean:
H2 = np.mean(np.array(H_res)[:, 1])

# Layer 1 Standard Deviation:
H1_sd = np.std(np.array(H_res)[:, 0])
# Layer 2 Standard Deviations:
H2_sd = np.std(np.array(H_res)[:, 1])

print("Layer1 HOTS: " + str(H1) + "+-" + str(H1_sd))  # Mean result +- std
print("Layer2 HOTS: " + str(H2) + "+-" + str(H2_sd))  # Mean result +- std

