import os
import os.path
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import utils
from utils.data_utils import read_UCR_UEA, z_normalization
from utils.explaination_utils import explain, get_xai_ref
from utils.visualization import plot_multiple_images_with_attribution


def insert_blender(instance, shape1, starting, blend_length):
    """
    blend the shapelet into instance
    :param instance:
    :param shape1:
    :param starting:
    :param blend_length:
    :return:
    """
    shapelet_length = len(shape1)
    length = len(instance)

    ending = starting + shapelet_length

    # Create blending weights
    blend_end = np.linspace(0, 1, blend_length)
    blend_start = np.linspace(1, 0, blend_length)
    # shape1
    # Insert sequence with blending
    result_sequence = instance.copy()
    result_sequence[starting:ending] = shape1

    # Apply blending at the start
    result_sequence[starting - blend_length:starting] \
        = instance[starting - blend_length] * blend_start + shape1[0] * blend_end

    # Apply blending at the end
    result_sequence[ending:ending + blend_length] = \
        instance[ending: ending + blend_length] * blend_end + shape1[-1] * blend_start

    return result_sequence


def insert_best(instance, shape1, quantile: float = None):
    """
    Insert the shapelet into the best location where casue the least steep changes
    if with quantile, then randomly choose a potition within the least percentiles else choose the least one

    :param instance:
    :param shape1:
    :param quantile:
    :return: inserted instance
    """

    shapelet_length = len(shape1)
    length = len(instance)
    shapelet_diff = shape1[0] - shape1[-1]
    instance_diff = instance[:-(shapelet_length)] - instance[shapelet_length:]
    # print(shapelet_diff, instance_diff)
    insert_diff = np.abs(instance_diff - shapelet_diff)
    if quantile is None:
        starting = np.argmin(insert_diff)

    else:
        threshold = np.quantile(insert_diff, quantile)
        indices = np.where(insert_diff <= threshold)[0]
        starting = np.random.choice(indices)
        # print(indices, insert_diff[indices], insert_diff[indices])
    best_diff = instance_diff[starting] - shapelet_diff
    # print(instance[starting] - shape1[0])
    instance[starting:starting + shapelet_length] = shape1 + (instance[starting] - shape1[0] - best_diff/2)

    return instance, starting


def insert_shapelet(data, shape1, is_add=False, is_blending=False, blend_length=5, is_best_insert=False):
    shapelet_length = len(shape1)

    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    startings = []
    for i in range(num):
        a = data[i].flatten()
        if is_add:
            a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
        elif is_blending:
            starting = np.random.randint(blend_length, length - shapelet_length - blend_length)
            a = insert_blender(a, shape1, starting, blend_length)
        elif is_best_insert:
            a, starting = insert_best(a, shape1, quantile=0.05)
        else:
            starting = np.random.randint(length - shapelet_length)
            a[starting:starting + shapelet_length] = shape1
        c1[i] = a.reshape(1, length)
        startings.append([starting])
    return c1, startings


def multiple_insert_shapelet(length, num_shapelet, shapelet_length, is_blending=False, blend_length=5):
    if is_blending:
        if length < (num_shapelet - 1) * (shapelet_length + 2 * blend_length):
            raise ValueError("Not enough space")
        startings = []
        while len(startings) < num_shapelet:
            _starting = np.random.randint(blend_length, length - shapelet_length - blend_length)

            if all(abs(_starting - val) >= (shapelet_length + 2 * blend_length) for val in startings):
                startings.append(_starting)
    else:
        if length < (num_shapelet - 1) * shapelet_length:
            raise ValueError("Not enough space")
        startings = []
        while len(startings) < num_shapelet:
            _starting = np.random.randint(length - shapelet_length)
            if all(abs(_starting - val) >= shapelet_length for val in startings):
                startings.append(_starting)
    return sorted(startings)


def data_given_env_multiple_shapelet(data, shape1=None, num_shapelet=1, is_add=True, is_blending=False, blend_length=5,
                                     is_best_insert=False):
    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    shapelet_length = len(shape1)

    if num_shapelet == 1 or is_best_insert:
        c1, startings = insert_shapelet(data, shape1, is_add=is_add, is_blending=is_blending, blend_length=5,
                                        is_best_insert=is_best_insert)

    else:
        c1 = np.zeros((num, 1, length))
        startings = []
        shapelet_length = len(shape1)
        for i in range(num):
            instance_startings = multiple_insert_shapelet(length, num_shapelet, shapelet_length, is_blending,
                                                          blend_length)
            a = data[i].flatten()

            for starting in instance_startings:

                if is_add:
                    a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
                elif is_blending:

                    a = insert_blender(a, shape1, starting, blend_length)
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
                       is_z_norm=True,
                       is_blending=False, blend_length=5,
                       is_best_insert=False,
                       img_path=None,
                       save_dir=None, device='cuda', is_plot=False, is_verbose=False):
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

            train_x, startings = data_given_env_multiple_shapelet(train_x, shape1=shapelet, num_shapelet=num_shapelet,
                                                                  is_add=is_add, is_blending=is_blending,
                                                                  is_best_insert=is_best_insert,
                                                                  blend_length=blend_length)
            # if is_z_norm:
            #     train_x = z_normalization(train_x)
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
                if len(starting_instance) > 0:
                    starting = starting_instance[0]
                else:
                    continue
            attribution_shapelets[i] = exp[starting:starting + length]

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


def get_bg_pred(model, selected_datasets, inst_length, target_class, device='cuda', is_z_norm=True):
    bg_per = {}
    model.to(device)
    for ds in selected_datasets:
        train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)
        train_x = interpolate_along_last_axis(train_x[:100], inst_length=inst_length)
        if is_z_norm:
            train_x = z_normalization(train_x)
        GP_preds_c0 = model(torch.from_numpy(train_x).float().to(device)).detach().cpu().numpy()
        predict_as_0 = np.count_nonzero(np.argmax(GP_preds_c0, axis=1) == target_class)
        percentage = predict_as_0 / len(train_x)
        bg_per[ds] = percentage

    return bg_per


def get_attr(model, data, startings, length, save_dir, xai_name='DeepLift', target_class=None, repeats=None,
             device='cuda'):
    if save_dir is not None:
        path_only = os.path.split(save_dir)[0]
        if path_only and not os.path.isdir(path_only):
            os.makedirs(path_only, exist_ok=True)
    if repeats is None:
        repeats = len(data)
    if startings is not None:
        attribution_shapelet = np.zeros((repeats, length))
    else:
        attribution_shapelet = None
    attributions = np.zeros((repeats, data.shape[-1]))
    xai_ref = get_xai_ref(xai_name)
    xai = xai_ref(model)
    model.to(device)
    target_class_marker = target_class if target_class is not None else 'pred'
    for i in range(repeats):
        target_sample = torch.from_numpy(data[i].reshape(1, -1, data.shape[-1])).float().to(device)

        if target_class is not None:
            target_label = target_class
        else:
            target_label = torch.argmax(model(target_sample)).item()
        exp = explain(xai, xai_name, target_sample, target_label, sliding_window=(1, 5), baselines=None)
        exp = exp.detach().cpu().clone().numpy().flatten()

        attributions[i] = exp
        if startings is not None:
            starting = startings[i]
            attribution_shapelet[i] = exp[starting:starting + length]

    if save_dir is not None:
        attr = {
            'target_class': target_class_marker,
            'xai_name': xai_name,
            'attributions': attributions,
            'attribution_shapelet': attribution_shapelet
        }
        with open(os.path.join(save_dir), 'wb') as f:
            pickle.dump(attr, f)
    return attributions, attribution_shapelet #,  attr,


def get_pdata(shapelet, selected_datasets, inst_length, num_shapelet=1, is_add=False, repeat_max=None,
              is_z_norm=True,
              is_blending=False, blend_length=5,
              is_best_insert=False,
              save_dir='probe/GunPoint'):
    """
    This function insert shapelet into selected datasets' training data, and create two datasets,
        one is insertion with shapelet
        another one is without insertion
        support multiple insertion.
        selected datasets where first interpolate into inst_length
    :param shapelet: inserted shapelet
    :param selected_datasets: the datasets that selected to insert
    :param inst_length: dataset instance length that interpolate into
    :param num_shapelet: number of shapelet
    :param is_add: are we adding the shapelet on the top of instance or insert the shapelet
    :param repeat_max: each dataset's maximum instance
    :param is_z_norm: do instance or not
    :param save_dir: the position we want to save the data
    :return: a dictionary that contains with shapelet and without shapelet.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    length = len(shapelet)
    pdata_ws = np.empty((0, 1, inst_length))
    pdata_wos = np.empty((0, 1, inst_length))
    startings = []
    for ds in selected_datasets:
        # if ds == 'GunPoint':
        #     continue
        train_ds, test_x, train_y, test_y, enc1 = utils.read_UCR_UEA(dataset=ds, UCR_UEA_dataloader=None)

        if repeat_max is None:
            repeat_max = len(train_ds)
        print(f'dealing with {ds}:', train_ds.shape)
        train_ds = utils.interpolate_along_last_axis(train_ds[:repeat_max], inst_length=inst_length)
        if is_z_norm:
            train_ds = utils.z_normalization(train_ds)
        pdata_wos = np.concatenate((pdata_wos, train_ds), axis=0)
        train_ds, startings_data = data_given_env_multiple_shapelet(train_ds, shape1=shapelet,
                                                                    num_shapelet=num_shapelet, is_add=is_add,
                                                                    is_blending=is_blending,
                                                                    is_best_insert=is_best_insert,
                                                                    blend_length=blend_length)
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
