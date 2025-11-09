#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:28:55 2020

@author: marcorax93
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from scipy.spatial.distance import cdist
import time, gc


def surfaces(data_recording, res_x, res_y, surf_dim, tau_params, n_pol):
    """
    Modified surfaces function with probabilistic tau.

    Arguments:
        data_recording: data of a single recording [x, y, p, t]
        res_x, res_y: pixel resolution of the dataset
        surf_dim: lateral dimension of the squared time context
        tau_params: dictionary containing mean and std of tau {'mean': float, 'std': float}
        n_pol: number of polarities (-1 if polarity info should be discarded)
    """
    import numpy as np

    dl = surf_dim // 2
    n_events = len(data_recording[3])

    # Generate a single tau value for this recording from the distribution
    tau = np.random.normal(tau_params['mean'], tau_params['std'])
    # Ensure tau is positive
    tau = max(tau, 1e-6)

    # Allocate memory
    if n_pol == -1:
        surface = np.zeros([res_y + 2 * dl, res_x + 2 * dl], dtype=np.float32)
        surfs = np.zeros([n_events, surf_dim, surf_dim], dtype=np.float32)
        timestamp_table = np.ones([res_y + 2 * dl, res_x + 2 * dl]) * -1
    else:
        surface = np.zeros([n_pol, res_y + 2 * dl, res_x + 2 * dl], dtype=np.float32)
        surfs = np.zeros([n_events, n_pol, surf_dim, surf_dim], dtype=np.float32)
        timestamp_table = np.ones([n_pol, res_y + 2 * dl, res_x + 2 * dl]) * -1

    for event in range(n_events):
        new_ts = data_recording[3][event]
        new_x = int(data_recording[0][event] + dl)
        new_y = int(data_recording[1][event] + dl)

        decay_map = ((new_ts - timestamp_table) > 0) * (timestamp_table > 0)
        surface = np.exp(((timestamp_table - new_ts) * decay_map) / tau) * decay_map

        if n_pol == -1:
            timestamp_table[new_y, new_x] = new_ts
            surfs[event, :, :] = surface[new_y - dl:new_y + dl + 1,
                                 new_x - dl:new_x + dl + 1]
        else:
            new_pol = int(data_recording[2][event])
            timestamp_table[new_pol, new_y, new_x] = new_ts
            surfs[event, :, :, :] = surface[:, new_y - dl:new_y + dl + 1,
                                    new_x - dl:new_x + dl + 1]

    return surfs


def learn(dataset, surf_dim, res_x, res_y, tau_params, n_clusters, n_pol,
          num_batches, n_jobs):
    """
    Memory-optimized learn function with probabilistic tau.
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    import gc
    import time
    from joblib import Parallel, delayed

    num_labels = len(dataset)

    n_total_events = 0
    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    n_events_map = np.zeros([num_labels, max_recording_per_label])
    for label in range(num_labels):
        num_recordings_label = len(dataset[label])
        for recording in range(num_recordings_label):
            n_events = len(dataset[label][recording][3])
            n_total_events += n_events
            n_events_map[label, recording] = n_events

    batch_recording = max_recording_per_label // num_batches

    # 优化的批处理大小
    batch_size = min(1000, n_total_events // num_batches)

    # 优化的KMeans配置
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        compute_labels=True,
        random_state=42,
        verbose=0,
        max_no_improvement=10,
        init_size=min(3 * n_clusters, n_total_events),
        n_init=3
    )

    print('Generating Time Surfaces and Clustering')
    start_time = time.time()

    # 使用生成器来减少内存占用
    def generate_surfaces():
        for batch in range(num_batches):
            if batch == num_batches - 1:
                batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
                n_events_batch_map = n_events_map[:, batch * batch_recording:]
            else:
                batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                                 range(num_labels)]
                n_events_batch_map = n_events_map[:, batch * batch_recording:(batch + 1) * batch_recording]

            for label in range(num_labels):
                num_recordings_label = len(batch_dataset[label])
                surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(
                    batch_dataset[label][recording],
                    res_x,
                    res_y,
                    surf_dim,
                    tau_params,
                    n_pol) for recording in range(num_recordings_label))

                for surf in surf_label:
                    if n_pol == -1:
                        yield surf.reshape(-1, surf_dim ** 2).astype('float32')
                    else:
                        yield surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')

                gc.collect()

    # 使用生成器进行增量训练
    surface_generator = generate_surfaces()
    for i, surf_batch in enumerate(surface_generator):
        if i % 100 == 0:
            print(f"\rProgress: {(i / (n_total_events / surf_batch.shape[0]) * 100):.2f}%", end="")
        kmeans.partial_fit(surf_batch)

    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + 'seconds')

    # 使用相同的生成器进行预测
    print('Generating Time Surfaces and Infering')
    start_time = time.time()

    current_idx = 0
    for batch in range(num_batches):
        print(f"\rProgress: {(batch / num_batches * 100):.2f}%", end="")

        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:, batch * batch_recording:]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                             range(num_labels)]
            n_events_batch_map = n_events_map[:, batch * batch_recording:(batch + 1) * batch_recording]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            for recording in range(num_recordings_label):
                surf = surfaces(batch_dataset[label][recording], res_x, res_y,
                                surf_dim, tau_params, n_pol)

                if n_pol == -1:
                    surf = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    surf = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')

                new_pols = kmeans.predict(surf)
                dataset[label][batch * batch_recording + recording][2] = new_pols

                current_idx += len(new_pols)
                gc.collect()

    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + 'seconds')

    return dataset, kmeans


def infer(dataset, surf_dim, res_x, res_y, tau_params, n_pol, kmeans, num_batches,
          n_jobs):
    """
    Memory-optimized infer function with probabilistic tau.
    """
    import numpy as np
    import gc
    import time
    from joblib import Parallel, delayed

    num_labels = len(dataset)
    n_total_events = 0
    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    n_events_map = np.zeros([num_labels, max_recording_per_label])
    for label in range(num_labels):
        num_recordings_label = len(dataset[label])
        for recording in range(num_recordings_label):
            n_events = len(dataset[label][recording][3])
            n_total_events += n_events
            n_events_map[label, recording] = n_events

    batch_recording = max_recording_per_label // num_batches

    print('Generating Time Surfaces and Infering')
    start_time = time.time()

    # 使用批处理和生成器来优化内存使用
    for batch in range(num_batches):
        print(f"\rProgress: {(batch / num_batches * 100):.2f}%", end="")

        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
            n_events_batch_map = n_events_map[:, batch * batch_recording:]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                             range(num_labels)]
            n_events_batch_map = n_events_map[:, batch * batch_recording:(batch + 1) * batch_recording]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            for recording in range(num_recordings_label):
                surf = surfaces(batch_dataset[label][recording], res_x, res_y,
                                surf_dim, tau_params, n_pol)

                if n_pol == -1:
                    surf = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    surf = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')

                new_pols = kmeans.predict(surf)
                dataset[label][batch * batch_recording + recording][2] = new_pols

                gc.collect()

    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + 'seconds')

    return dataset


def signature_gen(dataset, n_clusters, n_jobs):
    """
    This function is used generate signatures from the histogram as in the original
    paper and also train a Support Vector Classifier (machine) to compare classifiers.

    Arguments :

        dataset: list containing the data_recording for every recording of the
                 training dataset sorted by label.

        num_clusters: number of clusters extracted by the layer of the network.

        n_jobs: hist generation can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking

    Returns:

        signatures: The histogram signatures (a 2d array [labels,num_clusters]) for
                    training dataset

        norm_signatures: The normalized histogram signatures
                         (a 2d array [labels,num_clusters]) for training dataset

        svc: The support vector classifier (sklearn svm) trained on histograms

        norm_svc: The support vector classifier (sklearn svm) trained on
                  normalized histograms


    """

    def hists_gen(label, n_recordings, pols, n_clusters):
        n_events = len(pols)
        hist = np.array([sum(pols == cluster) for cluster in range(n_clusters)]) / n_recordings
        norm_hist = hist / n_events

        return hist, norm_hist

    n_labels = len(dataset)
    signatures = np.zeros([n_labels, n_clusters])
    norm_signatures = np.zeros([n_labels, n_clusters])
    all_hists = []
    all_norm_hists = []
    labels = []
    for label in range(n_labels):
        n_recordings = len(dataset[label])
        hists, norm_hists = zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(label,
                                                                            n_recordings, dataset[label][recording][2],
                                                                            n_clusters) for recording in
                                                         range(n_recordings)))
        all_hists += hists
        all_norm_hists += norm_hists
        labels += [label for recording in range(n_recordings)]
        signatures[label, :] = sum(hists)
        norm_signatures[label, :] = sum(norm_hists)

    svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    svc.fit(all_hists, labels)

    norm_svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    norm_svc.fit(all_norm_hists, labels)

    return signatures, norm_signatures, svc, norm_svc


def histogram_accuracy(dataset, n_clusters, signatures, norm_signatures, n_jobs):
    """
    This function is used to test the histogram classifier accuracy, with both
    signatures and normalized signatures obtained with signature_gen.

    Arguments :

        dataset: list containing the data_recording for every recording of the
                 training dataset sorted by label.

        num_clusters: number of clusters extracted by the layer of the network.

        signatures: The histogram signatures (a 2d array [labels,num_clusters]) for
                    training dataset

        norm_signatures: The normalized histogram signatures
                         (a 2d array [labels,num_clusters]) for training dataset

        n_jobs: hist generation can run on multiple threads.
                It CAN be an higher value than the number of threads, but use less
                if you like multitasking

    Returns:


        test_signatures: The histogram signatures (a 2d array [n_toral_recordings,num_clusters])
                         of the test dataset (per each recording)

        test_norm_signatures: The normalized histogram signatures
                              (a 2d array [n_toral_recordings,num_clusters]) of
                              the test dataset (per each recording)

        euc_accuracy: percent of correctly guessed recordings of test set using
                      uclidean distance of histograms

        euc_accuracy: percent of correctly guessed recordings of test set using
                      uclidean distance of normalized histograms

        euc_labels: predicted labels of test set using uclidean distance of
                    histograms

        norm_euc_labels: predicted labels of test set using uclidean distance of
                         normalized histograms

    """

    def hists_gen(label, n_recordings, pols, n_clusters, signatures, norm_signatures):
        n_events = len(pols)
        hist = np.array([sum(pols == cluster) for cluster in range(n_clusters)])
        norm_hist = hist / n_events
        euc_label = np.argmin(np.linalg.norm(signatures - hist, axis=1))
        norm_euc_label = np.argmin(np.linalg.norm(norm_signatures - norm_hist, axis=1))

        return hist, norm_hist, euc_label, norm_euc_label

    n_labels = len(dataset)
    n_toral_recordings = sum([len(dataset[label]) for label in range(n_labels)])
    test_signatures = np.zeros([n_toral_recordings, n_clusters])
    test_norm_signatures = np.zeros([n_toral_recordings, n_clusters])
    recording_idx = 0
    euc_labels = np.zeros(n_toral_recordings)
    norm_euc_labels = np.zeros(n_toral_recordings)
    euc_accuracy = 0
    norm_euc_accuracy = 0

    for label in range(n_labels):
        n_recordings = len(dataset[label])
        hists, norm_hists, rec_euc_label, rec_norm_euc_label = zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(label,
                                                                                                               n_recordings,
                                                                                                               dataset[
                                                                                                                   label][
                                                                                                                   recording][
                                                                                                                   2],
                                                                                                               n_clusters,
                                                                                                               signatures,
                                                                                                               norm_signatures)
                                                                                            for recording in
                                                                                            range(n_recordings)))
        test_signatures[recording_idx:recording_idx + n_recordings, :] = np.asarray(hists)
        test_norm_signatures[recording_idx:recording_idx + n_recordings, :] = np.asarray(norm_hists)
        euc_labels[recording_idx:recording_idx + n_recordings] = np.asarray(rec_euc_label)
        norm_euc_labels[recording_idx:recording_idx + n_recordings] = np.asarray(rec_norm_euc_label)

        euc_accuracy += sum(np.asarray(rec_euc_label) == label) * (100 / n_toral_recordings)
        norm_euc_accuracy += sum(np.asarray(rec_norm_euc_label) == label) * (100 / n_toral_recordings)

        recording_idx += n_recordings

    return test_signatures, test_norm_signatures, euc_accuracy, norm_euc_accuracy, euc_labels, norm_euc_labels


def recon_rates_svm(svc, norm_svc, test_signatures, test_norm_signatures, test_set):
    """
    This function is used to test the SVC (support vector classifier) accuracy,
    with both signatures and normalized signatures obtained with the histogram_accuracy
    (it takes some time to generate them so I only calculate them once).

    Arguments :

        svc: The support vector classifier (sklearn svm) trained on histograms

        norm_svc: The support vector classifier (sklearn svm) trained on
                  normalized histograms

        test_signatures: The histogram signatures (a 2d array [n_toral_recordings,num_clusters])
                         of the test dataset (per each recording)

        test_norm_signatures: The normalized histogram signatures
                              (a 2d array [n_toral_recordings,num_clusters]) of
                              the test dataset (per each recording)

        test_set: list containing the data_recording for every recording of the
                 test dataset sorted by label.

    Returns:


        rec_rate_svc: percent of correctly guessed recordings of test set using
                      the SVC on histograms

        rec_rate_norm_svc: percent of correctly guessed recordings of test set using
                           the SVC on normalized histograms


    """
    svc_labels = svc.predict(test_signatures)
    norm_svc_labels = norm_svc.predict(test_norm_signatures)
    rec_rate_svc = 0
    rec_rate_norm_svc = 0
    label_idx = 0
    for label in range(len(test_set)):
        for recording in range(len(test_set[label])):
            if svc_labels[label_idx] == label:
                rec_rate_svc += 1 / len(svc_labels) * 100
            if norm_svc_labels[label_idx] == label:
                rec_rate_norm_svc += 1 / len(svc_labels) * 100
            label_idx += 1
    return rec_rate_svc, rec_rate_norm_svc


def n_mnist_rearranging(dataset):
    """
    A function used to re-arrange n-mnist to a format more fitting for hots
    calculation.

    Arguments:

        dataset: original n-mnist dataset

    Returns:

        rearranged_dataset: list containing the data_recording for every
                            recording of the dataset sorted by label.
                            Data of a single recording is a list of 4 arrays [x, y, p, t]
                            containing spatial coordinates of events (x,y),
                            polarities (p) and timestamps (t).
    """
    rearranged_dataset = []
    n_labels = len(dataset)
    for label in range(n_labels):
        n_recordings = len(dataset[label][0][0])
        dataset_recording = []
        for recording in range(n_recordings):
            x = dataset[label][0][0][recording][0] - 1  # pixel index starts from 1 in N-MNIST
            y = dataset[label][0][0][recording][1] - 1  # pixel index starts from 1 in N-MNIST
            p = dataset[label][0][0][recording][2] - 1  # polarity index starts from 1 in N-MNIST
            ts = dataset[label][0][0][recording][3]
            dataset_recording.append([x, y, p, ts])
        rearranged_dataset.append(dataset_recording)

    return rearranged_dataset


def dataset_resize(dataset, res_x, res_y):
    """
    A function used to cut edge pixels of n-mnist and reduce its size

    Arguments:

        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label.

        res_x, res_y: the x and y pixel resolution of the dataset.

    Returns:

        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label, cut to res_x,res_y.

    """
    for label in range(len(dataset)):
        for recording in range((len(dataset[label]))):
            idx = dataset[label][recording][0] < res_x
            idy = dataset[label][recording][1] < res_y
            dataset[label][recording][0] = dataset[label][recording][0][idx * idy]
            dataset[label][recording][1] = dataset[label][recording][1][idx * idy]
            dataset[label][recording][2] = dataset[label][recording][2][idx * idy]
            dataset[label][recording][3] = dataset[label][recording][3][idx * idy]

    return dataset


def spac_downsample(dataset, ldim):
    """
    A function used to spatially undersample the dataset, it takes x and y
    coordinates of events and devides them by ldim to the closest integer (floor
    operation). This allows to increase spatial integration without scaling up
    the dimensionality of each layer (memrory intensive for clustering)

    Arguments:

        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label.

        ldim: the downsampling factor.

    Returns:

        dataset: list containing the data_recording for every
                 recording of the dataset sorted by label, scaled by ldim.

    """
    for label in range(len(dataset)):
        for recording in range((len(dataset[label]))):
            dataset[label][recording][0] = dataset[label][recording][0] // ldim
            dataset[label][recording][1] = dataset[label][recording][1] // ldim

    return dataset



