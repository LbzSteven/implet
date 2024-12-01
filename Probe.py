import os

import numpy as np
from sklearn.model_selection import train_test_split

import utils
import pickle
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC


def probe_shapelet(dataset, labels, pdata, model, shapelet, pos, device='cuda', save_path='probe/simu_implet_random',
                   shapelet_labels=None, is_threshold_info_gain=False):
    path_only = save_path
    if path_only and not os.path.isdir(path_only):
        os.makedirs(path_only, exist_ok=True)

    length = len(shapelet)
    num = pdata.shape[0]
    pdata_s_distances = np.zeros(num)
    for i in range(num):
        pdata_s_distances[i], _ = utils.compute_shapelet_distance(pdata[i], shapelet, length=length, position=pos)

    dataset_s_distances, _, best_threshold = utils.get_distances_info_gain(dataset, shapelet, length, pos, labels)

    data = pdata_s_distances.reshape(-1, 1)

    if not is_threshold_info_gain:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        centroids = kmeans.cluster_centers_.flatten()
        threshold = np.mean(centroids)
    else:
        threshold = best_threshold

    dataset_s_label = shapelet_labels[0] if shapelet_labels[0] is not None else [i >= threshold for i in
                                                                                 dataset_s_distances]
    pdata_s_label = shapelet_labels[1] if shapelet_labels[1] is not None else [i >= threshold for i in
                                                                               pdata_s_distances]
    print(pdata_s_label)
    dataset_latent = utils.get_hidden_layers(model=model, hook_block=None, data=dataset, device=device)
    pdata_latent = utils.get_hidden_layers(model=model, hook_block=None, data=pdata, device=device)

    dataset_latent = dataset_latent.reshape(dataset_latent.shape[0], -1)
    pdata_latent = pdata_latent.reshape(pdata_latent.shape[0], -1)
    print(dataset_latent.shape, pdata_latent.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        pdata_latent, pdata_s_label, test_size=0.2, random_state=42)

    probe_latent_label = {
        'dataset_s_distances': dataset_s_distances,
        'pdata_s_distances': pdata_s_distances,
        'dataset_s_label': dataset_s_label,
        'dataset_latent': dataset_latent,
        'pdata_s_label': pdata_s_label,
        'pdata_latent': pdata_latent,
        'latent_train': X_train,
        'latent_test': X_test,
        'label_train': y_train,
        'label_test': y_test,
    }
    with open(os.path.join(save_path, 'probe_latent_label.pkl'), 'wb') as f:
        pickle.dump(probe_latent_label, f)
    # with open('probe/simu_implet_random/probe_test_data.pkl', 'wb') as f:
    #     pickle.dump(probe_latent_label, f)

    classifier = LogisticRegression()
    # classifier = SVC(kernel='rbf')
    # classifier.fit(dataset_latent, dataset_s_label)
    classifier.fit(X_train, y_train)
    # dataset_s_pred = classifier.predict(dataset_latent)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    # dataset_s_pred[dataset_s_pred < 0.5] = 0
    # dataset_s_pred[dataset_s_pred >= 0.5] = 1

    dataset_s_pred = classifier.predict(dataset_latent)
    # pdata_s_pred[pdata_s_pred < 0.5] = 0
    # pdata_s_pred[pdata_s_pred >= 0.5] = 1

    print(f'Training Accuracy = {accuracy_score(y_train, pred_train):.2f}')

    print(f'Testing Accuracy = {accuracy_score(y_test, pred_test):.2f}')

    accuracy = accuracy_score(dataset_s_label, dataset_s_pred)
    # precision = precision_score(pdata_s_label, pdata_s_pred)
    # recall = recall_score(pdata_s_label, pdata_s_pred)
    # f1score = f1_score(pdata_s_label, pdata_s_pred)
    print(f"----------------------")
    print(f"Accuracy = {accuracy:.2f}")
    # print(f"Precision = {precision:.2f}")
    # print(f"Recall = {recall:.2f}")
    # print(f"F1 Score = {f1score:.2f}")

    results = {
        # 'pdata_s_pred': pdata_s_pred,
        'pred_train': pred_train,
        'pred_test': pred_test,
        'dataset_s_pred': dataset_s_pred,
        'accuracy': accuracy,
        'classifier': classifier
    }

    with open(os.path.join(save_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results


def get_pdata(shapelet, selected_datasets, inst_length, num_shapelet=1, is_add=False, repeat_max=None,
              is_z_norm=True, save_dir='probe/GunPoint'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    length = len(shapelet)
    pdata_ws = np.empty((0, 1, inst_length))
    pdata_wos = np.empty((0, 1, inst_length))
    startings = []
    for ds in selected_datasets:
        if ds == 'GunPoint':
            continue
        train_ds, test_x, train_y, test_y, enc1 = utils.read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)

        if repeat_max is None:
            repeat_max = len(train_ds)
        print(f'dealing with {ds}:', train_ds.shape)
        train_ds = utils.interpolate_along_last_axis(train_ds[:repeat_max], inst_length=inst_length)
        if is_z_norm:
            train_ds = utils.z_normalization(train_ds)
        pdata_wos = np.concatenate((pdata_wos, train_ds), axis=0)
        train_ds, startings_data = utils.data_given_env_multiple_shapelet(train_ds, shape1=shapelet,
                                                                          num_shapelet=num_shapelet, is_add=is_add)
        if is_z_norm:
            train_ds = utils.z_normalization(train_ds)
        pdata_ws = np.concatenate((pdata_ws, train_ds), axis=0)
        startings += startings
    pdata = {
        'pdata_ws': pdata_ws,
        'pdata_wos': pdata_wos,
        'startings': startings,
        'shapelet': shapelet,
        'length': length,
    }

    with open(os.path.join(save_dir, 'pdata.pkl'), 'wb') as f:
        pickle.dump(pdata, f)

    return pdata
