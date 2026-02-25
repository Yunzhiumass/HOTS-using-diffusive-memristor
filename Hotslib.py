

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
import time, gc
from numba import jit



@jit(nopython=True, fastmath=True)
def _numba_kernel_2d(
        new_ts, new_y, new_x, dl,
        timestamp_table,
        curve_assignment_table,
        curve_pool_time_x,
        curve_pool_currents_y,
        res_y_padded,
        res_x_padded
):

    surface = np.zeros((res_y_padded, res_x_padded), dtype=np.float32)


    for y in range(new_y - dl, new_y + dl + 1):
        if y < 0 or y >= res_y_padded: continue

        for x in range(new_x - dl, new_x + dl + 1):
            if x < 0 or x >= res_x_padded: continue

            t_start = timestamp_table[y, x]
            if t_start <= 0: continue

            curve_idx = curve_assignment_table[y, x]
            time_x = curve_pool_time_x
            current_y = curve_pool_currents_y[curve_idx]

            delta_ms = (new_ts - t_start) / 1000.0

            val = np.interp(delta_ms, time_x, current_y)

            if delta_ms > time_x[-1]:
                val = 0.0
            elif delta_ms < time_x[0]:
                val = current_y[0]

            surface[y, x] = val

    return surface



@jit(nopython=True, fastmath=True)
def _numba_kernel_3d(
        new_ts, new_y, new_x, dl,
        timestamp_table,
        curve_assignment_table,
        curve_pool_time_x,
        curve_pool_currents_y,
        n_pol,
        res_y_padded,
        res_x_padded
):

    surface = np.zeros((n_pol, res_y_padded, res_x_padded), dtype=np.float32)


    for y in range(new_y - dl, new_y + dl + 1):
        if y < 0 or y >= res_y_padded: continue

        for x in range(new_x - dl, new_x + dl + 1):
            if x < 0 or x >= res_x_padded: continue

            for p in range(n_pol):
                t_start = timestamp_table[p, y, x]
                if t_start <= 0: continue

                curve_idx = curve_assignment_table[p, y, x]
                time_x = curve_pool_time_x
                current_y = curve_pool_currents_y[curve_idx]

                delta_ms = (new_ts - t_start) / 1000.0

                val = np.interp(delta_ms, time_x, current_y)

                if delta_ms > time_x[-1]:
                    val = 0.0
                elif delta_ms < time_x[0]:
                    val = current_y[0]

                surface[p, y, x] = val

    return surface



def surfaces(data_recording, res_x, res_y, surf_dim, tau_params, n_pol,
             custom_curve=None, custom_dt_max=None, seed=0):
    import numpy as _np

    dl = surf_dim // 2
    n_events = len(data_recording[3])
    res_y_padded = res_y + 2 * dl
    res_x_padded = res_x + 2 * dl


    mean_tau = float(tau_params.get('mean', 1000))
    tau = max(_np.random.normal(mean_tau, 0), 1e-9)


    if n_pol == -1:
        surfs = _np.zeros([n_events, surf_dim, surf_dim], dtype=_np.float32)
        timestamp_table = _np.ones([res_y_padded, res_x_padded], dtype=_np.float32) * -1.0
    else:
        surfs = _np.zeros([n_events, n_pol, surf_dim, surf_dim], dtype=_np.float32)
        timestamp_table = _np.ones([n_pol, res_y_padded, res_x_padded], dtype=_np.float32) * -1.0


    use_custom = (custom_curve is not None)
    curve_assignment_table = None
    curve_pool_time_x = None
    curve_pool_currents_y = None

    if use_custom:
        curve_pool_time_x, curve_pool_currents_y = custom_curve
        n_curves = len(curve_pool_currents_y)

        _np.random.seed(seed)
        if n_pol == -1:
            curve_assignment_table = _np.random.randint(
                0, n_curves, size=(res_y_padded, res_x_padded), dtype=np.int32
            )
        else:
            curve_assignment_table = _np.random.randint(
                0, n_curves, size=(n_pol, res_y_padded, res_x_padded), dtype=np.int32
            )


    for event in range(n_events):
        new_ts = float(data_recording[3][event])
        new_x = int(data_recording[0][event] + dl)
        new_y = int(data_recording[1][event] + dl)

        if use_custom:

            if n_pol == -1:
                # Call 2D Kernel
                surface = _numba_kernel_2d(
                    new_ts, new_y, new_x, dl,
                    timestamp_table,
                    curve_assignment_table,
                    curve_pool_time_x,
                    curve_pool_currents_y,
                    res_y_padded,
                    res_x_padded
                )
            else:

                surface = _numba_kernel_3d(
                    new_ts, new_y, new_x, dl,
                    timestamp_table,
                    curve_assignment_table,
                    curve_pool_time_x,
                    curve_pool_currents_y,
                    n_pol,
                    res_y_padded,
                    res_x_padded
                )
        else:

            valid = (timestamp_table > 0)
            delta_us = new_ts - timestamp_table[valid]
            surface = _np.zeros_like(timestamp_table, dtype=_np.float32)
            surface[valid] = _np.exp(-delta_us / tau)


        if n_pol == -1:
            timestamp_table[new_y, new_x] = new_ts
            surfs[event, :, :] = surface[new_y - dl:new_y + dl + 1, new_x - dl:new_x + dl + 1]
        else:
            new_pol = int(data_recording[2][event])
            timestamp_table[new_pol, new_y, new_x] = new_ts
            surfs[event, :, :, :] = surface[:, new_y - dl:new_y + dl + 1, new_x - dl:new_x + dl + 1]

    return surfs




def learn(dataset, surf_dim, res_x, res_y, tau_params, n_clusters, n_pol,
          num_batches, n_jobs, custom_curve=None, custom_dt_max=None):
    num_labels = len(dataset)
    # Calculate total events for BatchKMeans init
    n_total_events = 0
    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    for label in range(num_labels):
        for recording in range(len(dataset[label])):
            n_total_events += len(dataset[label][recording][3])

    batch_recording = max_recording_per_label // num_batches if num_batches > 0 else max_recording_per_label
    batch_size = min(1000, max(1, n_total_events // max(1, num_batches)))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, max_iter=100,
        compute_labels=True, random_state=42, verbose=0,
        max_no_improvement=10, init_size=min(3 * n_clusters, max(3 * n_clusters, batch_size)), n_init=3
    )

    print('Generating Time Surfaces and Clustering (Numba Accelerated)')
    start_time = time.time()


    for batch in range(max(1, num_batches)):
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                             range(num_labels)]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])

            seeds = []
            for recording in range(num_recordings_label):
                rec_abs_idx = (batch * batch_recording) + recording
                seeds.append(label * 100000 + rec_abs_idx)


            surf_label = Parallel(n_jobs=n_jobs)(delayed(surfaces)(
                batch_dataset[label][recording],
                res_x, res_y, surf_dim,
                tau_params, n_pol,
                custom_curve=custom_curve,
                custom_dt_max=custom_dt_max,
                seed=seeds[recording]
            ) for recording in range(num_recordings_label))

            for surf in surf_label:
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')
                if X.size > 0:
                    kmeans.partial_fit(X)
            gc.collect()

    print(f'\rClustering Completed in: {time.time() - start_time:.2f} seconds')


    print('Generating Time Surfaces and Infering (Numba Accelerated & Parallelized)')
    start_time = time.time()
    for batch in range(max(1, num_batches)):
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                             range(num_labels)]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])


            seeds = []
            for recording in range(num_recordings_label):
                rec_abs_idx = (batch * batch_recording) + recording
                seeds.append(label * 100000 + rec_abs_idx)


            surf_list = Parallel(n_jobs=n_jobs)(delayed(surfaces)(
                batch_dataset[label][recording],
                res_x, res_y, surf_dim,
                tau_params, n_pol,
                custom_curve=custom_curve,
                custom_dt_max=custom_dt_max,
                seed=seeds[recording]
            ) for recording in range(num_recordings_label))


            for r, surf in enumerate(surf_list):
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')

                if X.size == 0:
                    new_pols = np.array([], dtype=int)
                else:
                    new_pols = kmeans.predict(X)

                dataset[label][batch * batch_recording + r][2] = new_pols

            del surf_list
            gc.collect()

    print(f'\rInference Completed in: {time.time() - start_time:.2f} seconds')
    return dataset, kmeans


def infer(dataset, surf_dim, res_x, res_y, tau_params, n_pol, kmeans, num_batches,
          n_jobs, custom_curve=None, custom_dt_max=None):
    num_labels = len(dataset)
    print('Generating Time Surfaces and Infering (Numba Accelerated & Parallelized)')
    start_time = time.time()


    max_recording_per_label = max([len(dataset[label]) for label in range(num_labels)])
    batch_recording = max_recording_per_label // num_batches if num_batches > 0 else max_recording_per_label

    for batch in range(max(1, num_batches)):
        # Prepare batch dataset
        if batch == num_batches - 1:
            batch_dataset = [dataset[label][batch * batch_recording:] for label in range(num_labels)]
        else:
            batch_dataset = [dataset[label][batch * batch_recording:(batch + 1) * batch_recording] for label in
                             range(num_labels)]

        for label in range(num_labels):
            num_recordings_label = len(batch_dataset[label])

            seeds = []
            for recording in range(num_recordings_label):
                rec_abs_idx = (batch * batch_recording) + recording
                seeds.append(label * 100000 + rec_abs_idx)


            surf_list = Parallel(n_jobs=n_jobs)(delayed(surfaces)(
                batch_dataset[label][recording],
                res_x, res_y, surf_dim,
                tau_params, n_pol,
                custom_curve=custom_curve,
                custom_dt_max=custom_dt_max,
                seed=seeds[recording]
            ) for recording in range(num_recordings_label))


            for r, surf in enumerate(surf_list):
                if n_pol == -1:
                    X = surf.reshape(-1, surf_dim ** 2).astype('float32')
                else:
                    X = surf.reshape(-1, n_pol * surf_dim ** 2).astype('float32')

                if X.size == 0:
                    new_pols = np.array([], dtype=int)
                else:
                    new_pols = kmeans.predict(X)


                dataset[label][batch * batch_recording + r][2] = new_pols

            # Clear memory
            del surf_list
            gc.collect()

    print(f'\rInference Completed in: {time.time() - start_time:.2f} seconds')
    return dataset


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