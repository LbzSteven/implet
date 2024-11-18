import os.path
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from utils.data_utils import read_UCR_UEA, z_normalization
from utils.explaination_utils import explain,get_xai_ref
from utils.visualization import plot_multiple_images_with_attribution

def insert_shapelet(data, shape1=None):
    shapelet_length = len(shape1)

    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    startings = []
    for i in range(num):
        starting = np.random.randint(length - shapelet_length)
        a = data[i].flatten()
        # print(a.shape)
        # a[starting:starting + shapelet_length] = shape1
        a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
        c1[i] = a.reshape(1, length)
        startings.append(starting)
    return c1, startings

def multiple_insert_shapelet(length, num_shapelet, shapelet_length):
    if length < (num_shapelet - 1) * shapelet_length:
        raise ValueError("Not enough space")
    startings = []
    while len(startings) < num_shapelet:
        _starting = np.random.randint(length - shapelet_length)
        if all(abs(_starting - val) >= shapelet_length for val in startings):
            startings.append(_starting)
    return sorted(startings)

def data_given_env_multiple_shapelet(data, shape1=None, num_shapelet=1,is_add=True):

    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    shapelet_length = len(shape1)
    startings = []
    for i in range(num):
        instance_startings = multiple_insert_shapelet(length, num_shapelet, shapelet_length)
        a = data[i].flatten()
        for starting in instance_startings:
            if is_add:
                a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
            else:
                a[starting:starting + shapelet_length] = shape1
        c1[i] = a.reshape(1, length)
        startings.append(instance_startings)
    return c1, startings

def interpolate_along_last_axis(data, inst_length=150):
    data = data.reshape(-1, data.shape[-1])
    input_shape = data.shape
    input_length = input_shape[-1]
    output_shape = input_shape[:-1] + (inst_length,)

    interpolated_data = np.zeros(output_shape)
    original_indices = np.linspace(0, 1, input_length)
    target_indices = np.linspace(0, 1, inst_length)

    for i in range(input_shape[0]):
        interpolated_data[i] = np.interp(target_indices, original_indices, data[i])
    interpolated_data = interpolated_data.reshape(-1, 1, inst_length)
    return interpolated_data

def closest_gt_mse(shapelet_attr, gt_attr):
    mses = []
    for i in range(len(gt_attr)):
        mses.append(mean_squared_error(shapelet_attr, gt_attr[i], squared=True))
    return np.min(mses)


def insert_data_to_env(model, shapelet, selected_datasets, inst_length, num_shapelet=1, is_add=True,
                       xai_name='DeepLift', num_attribution_each=None,
                       target_class=0, repeat_max=100,
                       is_z_norm=True, is_plot=False, is_verbose=False,
                       img_path=None,
                       save_dir=None, device='cuda'):
    insert_shapelet_percentage = {}
    dataset_attr = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if (xai_name is not None) and (not os.path.exists(os.path.join(save_dir, xai_name))):
        os.makedirs(os.path.join(save_dir, xai_name))

    length = len(shapelet)

    for ds in selected_datasets:
        if not os.path.exists(os.path.join(save_dir, f'{ds}.pkl')):
            train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)
            if is_z_norm:
                train_x = z_normalization(train_x)
            train_x = interpolate_along_last_axis(train_x[:repeat_max], inst_length=inst_length)

            train_x, startings = data_given_env_multiple_shapelet(train_x, shape1=shapelet, num_shapelet=num_shapelet,is_add=is_add)
            if is_z_norm:
                train_x = z_normalization(train_x)
            model.to(device)
            # Now it is predicted as 0, could be further changed to look at specific dataset
            preds = model(torch.from_numpy(train_x).float().to(device)).detach().cpu().numpy()
            predict_as_target_class = np.count_nonzero(np.argmax(preds, axis=1) == target_class)
            percentage = predict_as_target_class / len(train_x)

            insert_shapelet_percentage[ds] = percentage
            # model.cpu()

            if is_verbose:
                print(f'----------------------------')
                # print(f'{ds} mean: {train_x.mean():.2f}')
                print(f'{ds} as class 0', predict_as_target_class)
                print(f'percentage {percentage:.2f}')

            data_save = {
                'ds_name': ds,
                'shapelet': shapelet,
                'length': length,
                'train_x': train_x,
                'startings': startings,
                'num_sample': len(train_x),
                'preds': preds,
                'target_class': target_class,
                'percentage': percentage
            }
            with open(os.path.join(save_dir, f'{ds}.pkl'), 'wb') as f:
                pickle.dump(data_save, f)

        else:
            with open(os.path.join(save_dir, f'{ds}.pkl'), 'rb') as f:
                data_save = pickle.load(f)

            _, _, _, train_x, startings, _, preds, _, percentage = data_save.values()
            insert_shapelet_percentage[ds] = percentage

        if xai_name is None:
            continue

        attributions = np.zeros((train_x.shape[0], train_x.shape[-1]))
        xai_ref = get_xai_ref(xai_name)
        xai = xai_ref(model)

        if not num_attribution_each:
            num_attribution = len(train_x)
        else:
            num_attribution = num_attribution_each
        attribution_shapelets = np.zeros((num_attribution, length))
        model.to(device)

        for i in range(num_attribution):
            if is_verbose:
                print(f'{i + 1}/{num_attribution} using {xai_name} on {ds}')
            target_sample = torch.from_numpy(train_x[i].reshape(1, -1, train_x.shape[-1])).float().to(device)
            targeted_label = target_class
            exp = explain(xai, xai_name, target_sample, targeted_label, sliding_window=(1, 5), baselines=None)
            exp = exp.detach().cpu().clone().numpy().flatten()
            attributions[i] = exp
            starting = startings[i]
            if isinstance(starting, list):
                starting_instance = starting.copy()
                if len(starting_instance)>0:
                    starting = starting_instance[0]
                else:
                    continue
            attribution_shapelets[i] = exp[starting:starting+length]

        if is_plot:
            plot_multiple_images_with_attribution(train_x, preds, 4, (2, 2), (12, 6), use_attribution=True,
                                                  attributions=attributions,
                                                  normalize_attribution=True,
                                                  save_path=f'{img_path}_{ds}')
        attribution_save = {
            'xai_name': xai_name,
            'attributions': attributions,
            'attribution_shapelet': attribution_shapelets,
        }
        dataset_attr[ds] = attributions
        with open(os.path.join(save_dir, xai_name, f'{ds}.pkl'), 'wb') as f:
            pickle.dump(attribution_save, f)

    return dataset_attr, insert_shapelet_percentage


def get_bg_pred(model, selected_datasets, inst_length, target_class, device='cuda'):
    bg_per = {}
    model.to(device)
    for ds in selected_datasets:
        train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)
        train_x = interpolate_along_last_axis(train_x[:100], inst_length=inst_length)
        GP_preds_c0 = model(torch.from_numpy(train_x).float().to(device)).detach().cpu().numpy()
        predict_as_0 = np.count_nonzero(np.argmax(GP_preds_c0, axis=1) == target_class)
        percentage = predict_as_0 / len(train_x)
        bg_per[ds] = percentage

    return bg_per


def get_gt_attr(model, train_data, startings, length, save_dir, xai_name='DeepLift', target_class=None, repeats=None, device='cuda'):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if repeats is None:
        repeats = len(train_data)
    if startings is not None:
        attribution_shapelet = np.zeros((repeats, length))
    else:
        attribution_shapelet = None
    attributions = np.zeros((repeats, train_data.shape[-1]))
    xai_ref = get_xai_ref(xai_name)
    xai = xai_ref(model)
    model.to(device)
    target_class_marker = target_class if target_class is not None else 'pred'
    for i in range(repeats):
        target_sample = torch.from_numpy(train_data[i].reshape(1, -1, train_data.shape[-1])).float().to(device)

        if target_class is None:
            predicted_label = torch.argmax(model(target_sample)).item()
            target_class = predicted_label
        exp = explain(xai, xai_name, target_sample, target_class, sliding_window=(1, 5), baselines=None)
        exp = exp.detach().cpu().clone().numpy().flatten()

        attributions[i] = exp
        if startings is not None:
            starting = startings[i]
            attribution_shapelet[i] = exp[starting:starting + length]

    if save_dir is not None:
        gt_attr = {
            'target_class': target_class_marker,
            'xai_name': xai_name,
            'attributions': attributions,
            'attribution_shapelet': attribution_shapelet
        }
        with open(os.path.join(save_dir, 'train_exp.pkl'), 'wb') as f:
            pickle.dump(gt_attr, f)
    return attributions, attribution_shapelet  # gt_attr,
