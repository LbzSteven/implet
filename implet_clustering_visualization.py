import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime

from utils import pickle_save_to_file, pickle_load_from_file
from utils.data_utils import read_UCR_UEA
from utils.implet_extactor import implet_extractor, implet_cluster_auto, implet_cluster
from utils.visualization import plot_implet_clusters_with_instances



device = torch.device("cpu")

model_names = ['FCN']
tasks = ['GunPoint'] # , "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes", "Strawberry"

xai_names = ['Saliency', ] # , 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion', 'Saliency'



for model_name in model_names:
    print(f'======== {model_name} ========')

    for task in tasks:
        print(f'-------- {task} --------')

        for explainer in xai_names:
            _, test_x, _, test_y, _ = read_UCR_UEA(task, None)
            # load attributions
            # with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
            #     attr = pickle.load(f)
            # attr_test = attr['attributions']

            # compute implets
            # implets_class0 = implet_extractor(test_x, test_y, attr_test, target_class=0)
            # implets_class1 = implet_extractor(test_x, test_y, attr_test, target_class=1)
            # implets = implets_class0 + implets_class1

            # implets_list = {
            #     'implets_class0': implets_class0,
            #     'implets_class1': implets_class1,
            # }
            implets_save_dir = f'./output/half_implet/{model_name}/{task}/{explainer}'
            implets_list = pickle_load_from_file(os.path.join(implets_save_dir, 'implets.pkl'))

            implets_class0 = implets_list['implets_class0']
            implets_class1 = implets_list['implets_class1']


            for implate_name, implets_class_i in implets_list.items():
                cluster_path = os.path.join(implets_save_dir, f'{implate_name}_cluster_results.pkl')
                print(cluster_path)
                implet_cluster_results = pickle_load_from_file(cluster_path)
                best_indices_dep = implet_cluster_results['best_indices_dep']
                best_centroids_dep = implet_cluster_results['best_centroids_dep']
                best_k_dep = implet_cluster_results['best_k_dep']

                # print(best_indices_dep,best_centroids_dep,best_k_dep)
                # print(implets_save_dir, implate_name, best_k_dep)
                # print(implet_cluster_results['best_centroids_dep'])
                if best_k_dep is None:
                    continue

                for clsuster_i in range(best_k_dep):


                    implet_indexes = list(best_indices_dep[clsuster_i])
                    # print(implet_indexes)
                    implet_cluster = [implets_class_i[idx] for idx in implet_indexes]

                    # implet_with_attr = [np.vstack((imp[1], imp[2])).T for imp in implet_cluster]

                    instances_num = list(set([imp[0] for imp in implet_cluster]))
                    instances_num.sort()
                    # print(implet_with_attr, instances_num)
                    save_path = f'./figure/cluster_result/half_cluster/{model_name}/{explainer}/{task}/{implate_name}_clsuter{clsuster_i}.png'
                    plot_implet_clusters_with_instances(implet_cluster, test_x[instances_num], save_path=save_path)