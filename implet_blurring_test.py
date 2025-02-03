import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tsai.models.FCN import FCN
from tsai.models.InceptionTime import InceptionTime

from utils import pickle_save_to_file
from utils.data_utils import read_UCR_UEA
from utils.implet_extactor import implet_extractor
from utils.insert_shapelet import insert_random, overwrite_shaplet_random

device = torch.device("cpu")

model_names = ['InceptionTime']
tasks = ['GunPoint', "ECG200", "DistalPhalanxOutlineCorrect", "PowerCons", "Earthquakes",
         "Strawberry"]
xai_names = ['GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion',
             'Saliency']

# each row is [model_name, task_name, xai_name, method, acc_score]
# method is in ['ori', 'repl_implet', 'repl_random_loc']
result_path = 'output/blurring_test.csv'
if os.path.isfile(result_path):
    result = pd.read_csv(result_path).values.tolist()
else:
    result = []

n_trials = 10

for model_name in model_names:
    print(f'======== {model_name} ========')

    for task in tasks:
        print(f'-------- {task} --------')

        # load model
        if model_name == 'FCN':
            model = FCN(c_in=1, c_out=2)
        elif model_name == 'InceptionTime':
            model = InceptionTime(c_in=1, c_out=2)
        else:
            raise ValueError

        model_path = f'models/{model_name}/{task}/weight.pt'
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()


        def predict(x):
            if len(x.shape) == 1:
                x = x[np.newaxis, np.newaxis, :]
            elif len(x.shape) == 2:
                x = x[np.newaxis]

            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x).to(device)

            logits = model(x).detach().cpu().numpy()
            return np.argmax(logits, axis=-1)


        # load dataset
        _, test_x, _, test_y, _ = read_UCR_UEA(task, None)
        test_y = np.argmax(test_y, axis=1)

        for explainer in xai_names:
            # load attributions
            with open(f'attributions/{model_name}/{task}/{explainer}/test_exp.pkl', 'rb') as f:
                attr = pickle.load(f)
            attr_test = attr['attributions']

            # compute implets
            implets_class0 = implet_extractor(test_x, test_y, attr_test, target_class=0)
            implets_class1 = implet_extractor(test_x, test_y, attr_test, target_class=1)
            implets = implets_class0 + implets_class1

            # implets_list = {
            #     'implets': implets,
            #     'implets_class0': implets_class0,
            #     'implets_class1': implets_class1,
            # }
            # implets_save_dir = f'./output/{model_name}/{task}/{explainer}'
            # pickle_save_to_file(data=implets,
            #                     file_path=os.path.join(implets_save_dir, 'implets.pkl'))


            # modify samples
            test_x_overwritten = []
            test_x_rand_insert = []
            for implet in implets:
                i, _, _, _, start_loc, end_loc = implet
                implet_len = end_loc - start_loc + 1
                sample = test_x[i].flatten()
                for _ in range(n_trials):
                    sample_overwritten = overwrite_shaplet_random(sample, start_loc, implet_len)
                    sample_overwritten = sample_overwritten[np.newaxis]
                    test_x_overwritten.append(sample_overwritten)

                    sample_rand_insert = insert_random(sample, implet_len)
                    sample_rand_insert = sample_rand_insert[np.newaxis]
                    test_x_rand_insert.append(sample_rand_insert)

            test_x_overwritten = np.array(test_x_overwritten)
            test_x_rand_insert = np.array(test_x_rand_insert)

            y_true = np.array([test_y[i] for i, _, _, _, _, _ in implets])
            y_true = np.repeat(y_true, n_trials)

            # compute accuracies
            for method, x in zip(['ori', 'repl_implet', 'repl_random_loc'],
                                 [test_x, test_x_overwritten, test_x_rand_insert]):
                y_pred = predict(x)
                if y_pred.shape == test_y.shape:
                    acc = accuracy_score(test_y, y_pred)
                else:
                    acc = accuracy_score(y_true, y_pred)
                entry = [model_name, task, explainer, method, acc]
                print(entry)
                result.append(entry)

        df = pd.DataFrame(result, columns=['model_name', 'task_name', 'xai_name', 'method', 'acc_score'])
        df.to_csv(result_path, index=False)
