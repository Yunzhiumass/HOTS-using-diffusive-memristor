
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patched HOTS lib: adds custom-curve time surface generation (borrowed from v8e).
Only the time-surface path is changed; all other behaviors are kept.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
import time, gc

# -------------------------
# Helpers for custom curves
# -------------------------

def _moving_average(v, win=5):
    if win <= 1:
        return np.asarray(v, dtype=np.float32).copy()
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    vpad = np.pad(v, (pad, pad), mode='edge')
    return np.convolve(vpad, kernel, mode='valid').astype(np.float32)

def _resample_curve(curve, target_len):
    src = np.asarray(curve, dtype=np.float32)
    if len(src) == target_len:
        return src.copy()
    x_src = np.linspace(0.0, 1.0, len(src), dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    y_dst = np.interp(x_dst, x_src, src).astype(np.float32)
    return y_dst

def _monotone_decreasing(v):
    out = np.asarray(v, dtype=np.float32).copy()
    for i in range(1, len(out)):
        if out[i] > out[i-1]:
            out[i] = out[i-1]
    out[out < 0.0] = 0.0
    out[out > 1.0] = 1.0
    return out

def build_layer_curve(raw_curve, target_len, smooth_win=5):
    """Sanitize → resample → smooth → force non-increasing within [0,1]."""
    c = np.nan_to_num(np.asarray(raw_curve, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    c = _resample_curve(c, target_len)
    c = _moving_average(c, win=smooth_win)
    c = _monotone_decreasing(c)
    if len(c) > 0:
        if c[0] < 1.0:
            c[0] = min(1.0, max(c[0], 0.999))
        if c[-1] < 0.0:
            c[-1] = 0.0
    return c.astype(np.float32)

def _apply_curve(delta_t, curve, dt_max):
    """Map delta_t ∈ [0, dt_max] to indices of curve."""
    L = len(curve)
    if L <= 1:
        return np.ones_like(delta_t, dtype=np.float32)
    idx_f = (np.clip(delta_t, 0.0, dt_max) / max(dt_max, 1e-9)) * (L - 1)
    idx = np.clip(np.round(idx_f), 0, L - 1).astype(np.int32)
    return curve[idx]

# -------------------------
# Time surfaces
# -------------------------

def surfaces(data_recording, res_x, res_y, surf_dim, tau_params, n_pol,
             custom_curve=None, custom_dt_max=None):
    """
    Time-surface generator.
    If custom_curve & custom_dt_max are provided, uses lookup-curve decay.
    Otherwise falls back to exp((t_prev - t_now)/tau) with per-recording tau.
    """
    import numpy as _np

    dl = surf_dim // 2
    n_events = len(data_recording[3])

    # probabilistic tau (kept for backwards compatibility when not using custom curve)
    mean_tau = float(tau_params.get('mean', 1000))
    std_tau = float(tau_params.get('std', 0))
    tau = max(_np.random.normal(mean_tau, std_tau), 1e-9)

    # allocate
    if n_pol == -1:
        surface = _np.zeros([res_y + 2 * dl, res_x + 2 * dl], dtype=_np.float32)
        surfs = _np.zeros([n_events, surf_dim, surf_dim], dtype=_np.float32)
        timestamp_table = _np.ones([res_y + 2 * dl, res_x + 2 * dl], dtype=_np.float32) * -1.0
    else:
        surface = _np.zeros([n_pol, res_y + 2 * dl, res_x + 2 * dl], dtype=_np.float32)
        surfs = _np.zeros([n_events, n_pol, surf_dim, surf_dim], dtype=_np.float32)
        timestamp_table = _np.ones([n_pol, res_y + 2 * dl, res_x + 2 * dl], dtype=_np.float32) * -1.0

    use_custom = (custom_curve is not None) and (custom_dt_max is not None) and (len(custom_curve) >= 2)
    if use_custom:
        curve = _np.asarray(custom_curve, dtype=_np.float32)
        dt_max = float(custom_dt_max)

    for event in range(n_events):
        new_ts = float(data_recording[3][event])
        new_x = int(data_recording[0][event] + dl)
        new_y = int(data_recording[1][event] + dl)

        valid = (timestamp_table > 0)
        delta = new_ts - timestamp_table[valid]

        if use_custom:
            vals = _apply_curve(delta, curve, dt_max)
            surface[...] = 0.0
            surface[valid] = vals
        else:
            surface[...] = 0.0
            surface[valid] = _np.exp((timestamp_table[valid] - new_ts) / tau)

        if n_pol == -1:
            timestamp_table[new_y, new_x] = new_ts
            surfs[event, :, :] = surface[new_y - dl:new_y + dl + 1, new_x - dl:new_x + dl + 1]
        else:
            new_pol = int(data_recording[2][event])
            timestamp_table[new_pol, new_y, new_x] = new_ts
            surfs[event, :, :, :] = surface[:, new_y - dl:new_y + dl + 1, new_x - dl:new_x + dl + 1]

    return surfs

# -------------------------
# Learn / Infer (pass-through custom_curve)
# -------------------------

def learn(dataset, surf_dim, res_x, res_y, tau_params, n_clusters, n_pol,
          num_batches, n_jobs, custom_curve=None, custom_dt_max=None):
    """
    Same API as before, but accepts optional custom_curve/custom_dt_max and
    forwards them to `surfaces`.
    """
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

    batch_recording = max_recording_per_label // num_batches if num_batches>0 else max_recording_per_label
    batch_size = min(1000, max(1, n_total_events // max(1, num_batches)))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        compute_labels=True,
        random_state=42,
        verbose=0,
        max_no_improvement=10,
        init_size=min(3 * n_clusters, max(3 * n_clusters, batch_size)),
        n_init=3
    )

    print('Generating Time Surfaces and Clustering')
    start_time = time.time()

    # incremental fit
    for batch in range(max(1, num_batches)):
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in range(num_labels)]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])
            surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(
                batch_dataset[label][recording],
                res_x, res_y, surf_dim, tau_params, n_pol,
                custom_curve=custom_curve, custom_dt_max=custom_dt_max
            ) for recording in range(num_recordings_label))

            for surf in surf_label:
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')
                if X.size > 0:
                    kmeans.partial_fit(X)
            gc.collect()

    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + ' seconds')

    # infer to replace pols with cluster ids (same as original flow)
    print('Generating Time Surfaces and Infering')
    start_time = time.time()
    for batch in range(max(1, num_batches)):
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in range(num_labels)]

        for label in range(num_labels):
            for r, rec in enumerate(batch_dataset[label]):
                surf = surfaces(rec, res_x, res_y, surf_dim, tau_params, n_pol,
                                custom_curve=custom_curve, custom_dt_max=custom_dt_max)
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')
                if X.size == 0:
                    new_pols = np.array([], dtype=int)
                else:
                    new_pols = kmeans.predict(X)
                dataset[label][batch * batch_recording + r][2] = new_pols
                gc.collect()
    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + ' seconds')

    return dataset, kmeans


def infer(dataset, surf_dim, res_x, res_y, tau_params, n_pol, kmeans, num_batches,
          n_jobs, custom_curve=None, custom_dt_max=None):
    """
    As above, pass custom_curve/custom_dt_max through to `surfaces`.
    """
    num_labels = len(dataset)
    print('Generating Time Surfaces and Infering')
    start_time = time.time()

    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    batch_recording = max_recording_per_label // num_batches if num_batches>0 else max_recording_per_label

    for batch in range(max(1, num_batches)):
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in range(num_labels)]

        for label in range(num_labels):
            for r, rec in enumerate(batch_dataset[label]):
                surf = surfaces(rec, res_x, res_y, surf_dim, tau_params, n_pol,
                                custom_curve=custom_curve, custom_dt_max=custom_dt_max)
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')
                if X.size == 0:
                    new_pols = np.array([], dtype=int)
                else:
                    new_pols = kmeans.predict(X)
                dataset[label][batch * batch_recording + r][2] = new_pols
                gc.collect()

    print('\rProgress 100%. Completed in: ' + str(time.time() - start_time) + ' seconds')
    return dataset


# -------------- (unchanged below) --------------

def signature_gen(dataset, n_clusters, n_jobs):
    def hists_gen(label, n_recordings, pols, n_clusters):
        n_events = len(pols)
        hist = np.array([sum(pols == cluster) for cluster in range(n_clusters)]) / n_recordings
        norm_hist = hist / max(n_events, 1)
        return hist, norm_hist

    n_labels = len(dataset)
    signatures = np.zeros([n_labels, n_clusters])
    norm_signatures = np.zeros([n_labels, n_clusters])
    all_hists = []
    all_norm_hists = []
    labels = []
    for label in range(n_labels):
        n_recordings = len(dataset[label])
        hists, norm_hists = zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(
            label, n_recordings, dataset[label][recording][2], n_clusters
        ) for recording in range(n_recordings)))
        all_hists += hists
        all_norm_hists += norm_hists
        labels += [label for _ in range(n_recordings)]
        signatures[label, :] = sum(hists)
        norm_signatures[label, :] = sum(norm_hists)

    svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    svc.fit(all_hists, labels)

    norm_svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
    norm_svc.fit(all_norm_hists, labels)

    return signatures, norm_signatures, svc, norm_svc


def histogram_accuracy(dataset, n_clusters, signatures, norm_signatures, n_jobs):
    def hists_gen(label, n_recordings, pols, n_clusters, signatures, norm_signatures):
        n_events = len(pols)
        hist = np.array([sum(pols == cluster) for cluster in range(n_clusters)])
        norm_hist = hist / max(n_events, 1)
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
        hists, norm_hists, rec_euc_label, rec_norm_euc_label = zip(*Parallel(n_jobs=n_jobs)(delayed(hists_gen)(
            label, n_recordings, dataset[label][recording][2], n_clusters, signatures, norm_signatures
        ) for recording in range(n_recordings)))
        test_signatures[recording_idx:recording_idx + n_recordings, :] = np.asarray(hists)
        test_norm_signatures[recording_idx:recording_idx + n_recordings, :] = np.asarray(norm_hists)
        euc_labels[recording_idx:recording_idx + n_recordings] = np.asarray(rec_euc_label)
        norm_euc_labels[recording_idx:recording_idx + n_recordings] = np.asarray(rec_norm_euc_label)

        euc_accuracy += sum(np.asarray(rec_euc_label) == label) * (100 / n_toral_recordings)
        norm_euc_accuracy += sum(np.asarray(rec_norm_euc_label) == label) * (100 / n_toral_recordings)

        recording_idx += n_recordings

    return test_signatures, test_norm_signatures, euc_accuracy, norm_euc_accuracy, euc_labels, norm_euc_labels


def n_mnist_rearranging(dataset):
    rearranged_dataset = []
    n_labels = len(dataset)
    for label in range(n_labels):
        n_recordings = len(dataset[label][0][0])
        dataset_recording = []
        for recording in range(n_recordings):
            x = dataset[label][0][0][recording][0] - 1
            y = dataset[label][0][0][recording][1] - 1
            p = dataset[label][0][0][recording][2] - 1
            ts = dataset[label][0][0][recording][3]
            dataset_recording.append([x, y, p, ts])
        rearranged_dataset.append(dataset_recording)
    return rearranged_dataset


def dataset_resize(dataset, res_x, res_y):
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
    for label in range(len(dataset)):
        for recording in range((len(dataset[label]))):
            dataset[label][recording][0] = dataset[label][recording][0] // ldim
            dataset[label][recording][1] = dataset[label][recording][1] // ldim
    return dataset
