import argparse
import time

import numpy as np
import warnings
import torch
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from utils import *

from tsai.models.FCN import FCN
from tsai.models.RNN import RNN
from tsai.models.MLP import MLP


def main(dataset, model_path, k=None, lamb=0.1, model_type='FCN', xai_name='DeepLift',
         attr_class=None, sample_filter_class=None, sample_filter_by_pred_y=True):
    if model_path is None:
        model_path = f'../../PretrainModels/TimeSeriesClassifications/FCN/{dataset}'
        train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset, None)

        test_y = np.argmax(test_y, axis=1)
        train_y = np.argmax(train_y, axis=1)
        num_class = len(np.unique(train_y))

    else:
        with open(f'{model_path}/data.pkl', 'rb') as f:
            data = pickle.load(f)
        train_x, test_x, train_y, test_y = data['train_x'], data['test_x'], data['train_y'], data['test_y']
        test_y = np.argmax(test_y, axis=1)
        train_y = np.argmax(train_y, axis=1)
        num_class = len(np.unique(train_y))

    if model_type == 'RNN':
        model = RNN(c_in=1, c_out=num_class)
    elif model_type == 'FCN':
        model = FCN(c_in=1, c_out=num_class)
    elif model_type == 'MLP':
        model = MLP(c_in=1, c_out=num_class, seq_len=train_x.shape[-1])
    else:
        raise ValueError('Wrong model pick')
    state_dict = torch.load(f'{model_path}/weight.pt', map_location='cuda:1')
    model.load_state_dict(state_dict)
    model.train()
    test_y_pred = np.load(f'{model_path}/test_preds.npy')

    attr_save_dir = f'attributions/{dataset}_class_{str(attr_class)}/{xai_name}/exp.pkl'
    if not os.path.isfile(attr_save_dir):
        attr_gp, _ = get_gt_attr(model, test_x, None, None,
                                 save_dir=attr_save_dir,
                                 xai_name=xai_name, target_class=attr_class)
    else:
        with open(attr_save_dir, 'rb') as f:
            attr = pickle.load(f)
            attr_gp = attr['attributions']

    for i in range(num_class):
        if sample_filter_class is None:
            sample_filter_type = 'all_samples'
        else:
            sample_filter_type = f'sample_{"pred" if sample_filter_by_pred_y else "true"}_{sample_filter_class}'
        implets_save_dir = f'figure/{dataset}/attr_{str(attr_class)}_{sample_filter_type}/{xai_name}/c{i}/'
        if sample_filter_class is not None and sample_filter_by_pred_y:
            y = test_y_pred
        else:
            y = test_y
        implets_class_i = implet_extractor(test_x, y, attr_gp, target_class=sample_filter_class, lamb=lamb)
        num_implets = len(implets_class_i)

        path_only = os.path.split(implets_save_dir)[0]
        if path_only and not os.path.isdir(path_only):
            os.makedirs(path_only, exist_ok=True)

        with open(os.path.join(implets_save_dir, 'implets.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'implets': implets_class_i,
                    'num_implets': num_implets,
                }, f
            )

        plot_multiple_images_with_attribution(
            np.array([imp[1] for imp in implets_class_i[:50]]),
            np.array([0 for _ in implets_class_i[:50]]),
            min(num_implets, 50),
            use_attribution=True,
            attributions=np.array([imp[2] for imp in implets_class_i[:50]]),
            save_path=os.path.join(implets_save_dir, 'implets.png')
        )
        # dependent DTW
        implet_with_attr = [np.vstack((imp[1], imp[2])).T for imp in implets_class_i]
        if k is None:
            best_k_dep, best_indices_dep, best_centroids_dep = implet_cluster_auto(implet_with_attr, None)
        else:
            best_k_dep = k
            best_indices_dep, best_centroids_dep = implet_cluster(implet_with_attr, k)
        plot_implet_clusters(implet_with_attr, best_indices_dep, best_centroids_dep,
                             save_path=os.path.join(implets_save_dir, 'cluster_depDTW.png'))

        # 1d DTW
        implet_itself = [imp[1] for imp in implets_class_i]
        if k is None:
            best_k_1d, best_indices_1d, best_centroids_1d = implet_cluster_auto(implet_itself, None)
        else:
            best_k_1d = k
            best_indices_1d, best_centroids_1d = implet_cluster(implet_itself, k)
        plot_implet_clusters(implet_with_attr, best_indices_1d, best_centroids_1d,
                             save_path=os.path.join(implets_save_dir, 'cluster_1dDTW.png'))

        with open(os.path.join(implets_save_dir, 'implets_cluster_results.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'implets': implets_class_i,
                    'num_implets': num_implets,
                    'best_k_dep': best_k_dep,
                    'best_indices_dep': best_indices_dep,
                    'best_centroids_dep': best_centroids_dep,
                    'best_k_1d': best_k_1d,
                    'best_indices_1d': best_indices_1d,
                    'best_centroids_1d': best_centroids_1d,
                }, f
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify datasets')

    parser.add_argument('--dataset_name', type=str, default=None, help='dataset_name')
    parser.add_argument('--model_path', type=str, default=None, help='model_path')
    parser.add_argument('--dataset_choice', type=str, default='uni', help='dataset choice')
    parser.add_argument('--num_clusters', type=int, default=None, help='num_clusters')
    parser.add_argument('--implet_lambda', type=float, default=0.1, help='hyperparameter during implet extraction')
    parser.add_argument('--model_type', type=str, default='FCN', help='type of the models')
    parser.add_argument('--xai_name', type=str, default='DeepLift', help='type of explainer')
    parser.add_argument('--attr_class', type=str, default='None',
                        help='attribution is computed w.r.t. attr_class')
    parser.add_argument('--sample_filter_class', type=str, default='None',
                        help='only compute implets of the given class')
    parser.add_argument('--sample_filter_by_pred_y', default=False, action='store_true',
                        help='Ignored when sample_filter_class is None.'
                             'If true, only keep test_y == sample_filter_class;'
                             'otherwise, only keep test_y_pred == sample_filter_class.')

    args = parser.parse_args()
    if args.dataset_choice == 'uni':
        datasets = selected_uni
    else:
        datasets = [args.dataset_choice]
    dataset_name = args.dataset_name
    model_path = args.model_path
    k = args.num_clusters
    lamb = args.implet_lambda
    model_type = args.model_type
    xai_name = args.xai_name
    attr_class = None if args.attr_class == 'None' else int(args.attr_class)
    sample_filter_class = None if args.sample_filter_class == 'None' else int(args.sample_filter_class)
    sample_filter_by_pred_y = args.sample_filter_by_pred_y
    if dataset_name is None:
        print(datasets)
        for dataset in datasets:
            main(dataset, None, k, lamb=lamb, model_type=model_type, xai_name=xai_name,
                 attr_class=attr_class,
                 sample_filter_class=sample_filter_class, sample_filter_by_pred_y=sample_filter_by_pred_y)
    else:
        main(dataset_name, model_path, k, lamb=lamb, model_type=model_type, xai_name=xai_name,
             attr_class=attr_class,
             sample_filter_class=sample_filter_class, sample_filter_by_pred_y=sample_filter_by_pred_y)