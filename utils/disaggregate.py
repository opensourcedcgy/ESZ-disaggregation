# -*- coding: utf-8 -*-
"""Disaggregation related utilities"""
import logging
import numpy as np


def mains_to_batches(mains, num_seq_per_batch, seq_length, pad=True, stride=1):
    """
    In the disaggregation step this is used
    convert mains into batches. E.g. [0,1,2,3,4,5] with num_seq_per_batch=4, stride=1, seq_length=3
    will be converted into [[[[0],[1],[2]],[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]]]]
    Parameters
    ----------
    :param mains : 1D np.ndarray
        And it is highly advisable to pad `mains` with `seq_length` elements
        at both ends so the net can slide over the very start and end.
    :param num_seq_per_batch: number of sequences in a batch
    :param seq_length: int, sequence length
    :param pad: boolean, padding for the mains
    :param stride : int, optional
    :return: batches : list of 3D (num_sequences, seq_length, 1) np.ndarray
    """
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    assert mains.ndim == 1
    if pad:
        mains = np.pad(np.copy(mains), seq_length - 1, 'constant')
    n_mains_samples = len(mains)
    # Divide mains data into batches
    n_batches = ((float(n_mains_samples) / stride) / num_seq_per_batch)
    n_batches = np.ceil(n_batches).astype(int)
    logging.debug("Number of batches {}".format(n_batches))

    batches = []
    seq_starts = []
    seq_indexes = list(range(0, n_mains_samples - seq_length, stride))
    for batch_i in range(n_batches):
        selected_indexes = seq_indexes[batch_i*num_seq_per_batch:(batch_i+1)*num_seq_per_batch]
        batch = []
        for inx in selected_indexes:
            seq = mains[inx:inx+seq_length]
            if len(seq) == seq_length:
                batch.append(seq)
                seq_starts.append(inx)
        batch = np.reshape(np.array(batch), (len(batch), seq_length, 1)).astype(np.float32)
        batches.append(batch)

    return batches


def disaggregate(model, mains, model_name, num_seq_per_batch, seq_len,
                    appliance, target_scale, stride=1):
    """
    Disaggregation function to predict all results for whole time series mains.
    :param model: tf model object
    :param mains: numpy.ndarray, shape(-1,)
    :param model_name: name of the used model
    :param num_seq_per_batch: int, number of sequences to have in the batch
    :param seq_len: int, length of the sequence
    :param appliance: str, name of the appliance
    :param target_scale: int, scaling factor of predicted value
    :param stride: int, stride of moving window
    :return:
            p: np.ndarray,  shape(-1,), disaggregated power of the appliance
            metrics = dict containing the metrics
    """
    # Converting mains array into batches for prediction
    mains = mains.reshape(-1,)
    agg_batches = mains_to_batches(mains, num_seq_per_batch, seq_len, stride=stride, pad=True)

    if (appliance == 'fridge') or (appliance == 'Refrigerator') or (appliance == 'REFRIGERATOR'):
        if target_scale:
            target_max = target_scale
        else:
            target_max = 313
        target_min = 0
        input_max = 7879
        input_min = 80
    elif (appliance == 'washing machine') or (appliance == 'Washing_Machine') or (appliance == 'WASHING_MACHINE'):
        if target_scale:
            target_max = target_scale
        else:
            target_max = 3999
        target_min = 0
        input_max = 7879
        input_min = 80
    elif (appliance == 'dishwasher') or (appliance == 'Dishwasher') or (appliance == 'DISHWASHER'):
        if target_scale:
            target_max = target_scale
        else:
            target_max = 500
        target_min = 0
        input_max = 7879
        input_min = 80

    elif (appliance == 'Electric_Vehicle') or (appliance == 'electric vehicle') or (appliance=='ELECTRIC_VEHICLE'):
        if target_scale:
            target_max = target_scale
        else:
            target_max = 6000
        target_min = 0
        input_max = 7879
        input_min = 80

    elif (appliance == 'DRYER'):
        if target_scale:
            target_max = target_scale
        else:
            target_max = 2500
        target_min = 0
        input_max = 7879
        input_min = 80

    # list to store predictions
    y_net = []
    for id, batch in enumerate(agg_batches):
        X_pred = np.copy(batch.reshape(-1, seq_len, 1))
        X_pred /= (input_max-input_min)
        X_pred = X_pred * 10
        y_net.append(model.predict(X_pred))

    # converting the predictions to rectangles
    rectangles = pred_to_rectangles(y_net, num_seq_per_batch, seq_len, stride)

    return rectangles


def ensemble_rectangles(rectangles_dict, target_scale, seq_len, stride, probability_threshold, sample_period_seconds, mains):
    """
    Ensemble the predicted activities into time series data.
    :param rectangles_dict: dict, activity dict from the prediction
    :param target_scale: int, scaling factor to convert predicted value to real value
    :param seq_len: int
    :param stride: int
    :param probability_threshold: float
    :param sample_period_seconds: str, sampling period in 'xs'.
    :param mains: 1D np.ndarray
    :return: disaggregated_power: np.ndarray,  time series data
    """

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    # Scaling the predictions
    for model, _ in rectangles_dict.items():
        rectangles_dict[model][:, 2] *= target_scale  # scaling the y power estimates
    # Merging the rectanles
    rectangles = np.array([rect for rectangles in rectangles_dict.values() for rect in rectangles])

    # thresholding predictions
    num_samples = len(mains) + 2 * (seq_len - 1)
    matrix = np.ones(shape=(num_samples, 2), dtype=np.float32)
    # TODO: Remove those rectangels with less than miniumum threshold on power
    # TODO: This assumes padding always
    for rect in rectangles:  # gettting one rectangle
        start_point = int(min(rect[0], num_samples))
        end_point = int(min(rect[1], num_samples))
        matrix[start_point:end_point, 0] += 1
        matrix[start_point:end_point, 1] += rect[2]
    matrix[:, 1] = (matrix[:, 1] - 1) / matrix[:, 0]
    # the lowest predicted number of power at a same time point must exceed the pro_threshold
    prob = (matrix[:, 0] - 1) / matrix[:, 0].max()
    matrix[prob < probability_threshold] = 0
    matrix[matrix[:, 0] < 2] = 0
    power_vector = matrix[:, 1]
    # basically removing the padding from the starting and controlling it using len(mains_arr) at the end
    disaggregated_power = power_vector[seq_len - 1:seq_len + mains.shape[0] - 1]
    return disaggregated_power


def pred_to_rectangles(pred, num_seq_per_batch, seq_len, stride):
    """
    Convert prediction result to rectangle values for displaying in charts
    :param pred: list, list of predictions from network
    :param num_seq_per_batch: int, number of seqs per batch
    :param seq_len: int
    :param stride: int
    :return: np.ndarray with batches of [start, end, height] as rectangles
    """
    rectangles = []
    for id, batch in enumerate(pred):
        start_batch = id * num_seq_per_batch * stride
        if batch.shape[0]:
            for i in range(batch.shape[0]):
                start_seq = start_batch + stride * i
                start_rec = start_seq + seq_len * batch[i][0]
                end_rec = start_seq + seq_len * batch[i][1]
                height_rec = batch[i][2]
                if ~np.isnan(start_rec) and ~np.isnan(end_rec):
                    start_rec = int(round(start_rec))
                    end_rec = int(round(end_rec))
                else:
                    # print np.nan
                    start_rec = 0
                    end_rec = 0
                    height_rec = 0
                rectangles.append([start_rec, end_rec, height_rec])
    return np.array(rectangles)


