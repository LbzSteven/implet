import numpy as np


def max_score_subsequence(arr, left, lamb, threshold):
    n = len(arr)
    max_score = float('-inf')
    best_start, best_end = -1, -1
    current_sum = 0

    if arr[left] < threshold:
        return None, max_score, best_start, best_end

    for right in range(left, n):
        current_sum += arr[right]
        length = right - left + 1
        current_score = (current_sum / length) + lamb * length
        if current_score > max_score and (current_sum / length) > threshold:
            max_score = current_score
            best_start, best_end = left, right
        # while left <= right and (current_sum / length + lamb*length) < max_score:
        #    current_sum -= arr[left]
        #    left += 1
    return arr[best_start:best_end + 1], max_score, best_start, best_end


def implet_extractor(train_x, train_y, attr, target_class=0):
    implets = []
    num = len(train_x)
    for i in range(num):
        inst = train_x[i].flatten()
        if train_y[i] != target_class:
            continue
        starting = 0
        abs_attr = np.abs(attr[i].flatten())
        avg = np.mean(abs_attr)
        while starting < len(abs_attr):
            sub_attr, max_score, best_start, best_end = max_score_subsequence(arr=abs_attr, left=starting, lamb=0.1,
                                                                              threshold=avg * 2)
            if best_end != -1:
                implets.append([i, inst[best_start:best_end + 1], sub_attr, max_score, best_start, best_end])
                starting = best_end + 1
            else:
                starting += 1
    return implets
