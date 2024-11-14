import os

import numpy as np
import utils
import pickle
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LinearRegression


def probe_shapelet(dataset, labels, pdata, model, shapelet, pos, device='cuda'):

    length = len(shapelet)
    dataset_s_distances, info_gain, best_threshold = utils.get_distances_info_gain(dataset, shapelet, length, pos,
                                                                                  labels)

    dataset_s_label = [1 if i > best_threshold else 0 for i in dataset_s_distances]

    num = pdata.shape[0]
    pdata_s_distances = np.zeros(num)

    for i in range(num):
        pdata_s_distances[i], _ = utils.compute_shapelet_distance(pdata[i], shapelet, length=length, position=pos)
    pdata_s_label = [1 if i > best_threshold else 0 for i in pdata_s_distances]

    dataset_latent = utils.get_hidden_layers(model=model, hook_block=None, data=dataset, device=device)
    pdata_latent = utils.get_hidden_layers(model=model, hook_block=None, data=pdata, device=device)

    dataset_latent = dataset_latent.reshape(dataset_latent.shape[0], -1)
    pdata_latent = pdata_latent.reshape( pdata_latent.shape[0], -1)
    print(dataset_latent.shape, pdata_latent.shape)
    probe_latent_label = {
        'dataset_s_label': dataset_s_label,
        'pdata_s_label': pdata_s_label,
        'dataset_latent': dataset_latent,
        'pdata_latent': pdata_latent,
    }
    with open('probe/GunPoint/probe_latent_label.pkl', 'wb') as f:
        pickle.dump(probe_latent_label, f)

    classifier = LinearRegression()
    classifier.fit(dataset_latent, dataset_s_label)

    dataset_s_pred = classifier.predict(dataset_latent)
    dataset_s_pred[dataset_s_pred < 0.5] = 0
    dataset_s_pred[dataset_s_pred >= 0.5] = 1

    pdata_s_pred = classifier.predict(pdata_latent)
    pdata_s_pred[pdata_s_pred < 0.5] = 0
    pdata_s_pred[pdata_s_pred >= 0.5] = 1

    print(f'Training Accuracy = {accuracy_score(dataset_s_label, dataset_s_pred):.2f}')


    accuracy = accuracy_score(pdata_s_label, pdata_s_pred)
    precision = precision_score(pdata_s_label, pdata_s_pred)
    recall = recall_score(pdata_s_label, pdata_s_pred)
    f1score = f1_score(pdata_s_label, pdata_s_pred)
    print(f"----------------------")
    print(f"Accuracy = {accuracy:.2f}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print(f"F1 Score = {f1score:.2f}")

    results = {
        'pdata_s_pred': pdata_s_pred,
        'dataset_s_pred': dataset_s_pred,
        'accuracy': accuracy
    }

    with open('probe/GunPoint/results.pkl', 'wb') as f:
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
        train_ds, test_x, train_y, test_y, enc1 = utils.read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)

        if repeat_max is None:
            repeat_max = len(train_ds)
        print(f'dealing with {ds}:', train_ds.shape)
        train_ds = utils.interpolate_along_last_axis(train_ds[:repeat_max], inst_length=inst_length)
        if is_z_norm:
            train_ds = utils.z_normalization(train_ds)
        pdata_wos = np.concatenate((pdata_wos, train_ds), axis=0)
        train_ds, startings_data = utils.data_given_env_multiple_shapelet(train_ds, shape1=shapelet, num_shapelet=num_shapelet,is_add=is_add)
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

