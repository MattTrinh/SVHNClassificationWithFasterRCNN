import numpy
import datetime
import tensorflow as tf
from tensorflow import keras
def intersection_over_union(y_true, y_pred):
    """
    :param y_pred: [minimum_r, minimum_c, maximum_r, maximum_c]
    :param y_true: [minimum_r, minimum_c, maximum_r, maximum_c]

    :return:

    """
    print("Calculating IoU")
    
    y_true_minimum_r, y_true_minimum_c, y_true_maximum_r, y_true_maximum_c = y_true
    y_pred_minimum_r, y_pred_minimum_c, y_pred_maximum_r, y_pred_maximum_c = y_pred

    if y_true_maximum_r < y_pred_minimum_r:
        return 0.0

    if y_pred_maximum_r < y_true_minimum_r:
        return 0.0

    if y_true_maximum_c < y_pred_minimum_c:
        return 0.0

    if y_pred_maximum_c < y_true_minimum_c:
        return 0.0

    minimum_r = numpy.maximum(y_true_minimum_r, y_pred_minimum_r)
    minimum_c = numpy.maximum(y_true_minimum_c, y_pred_minimum_c)

    maximum_r = numpy.minimum(y_true_maximum_r, y_pred_maximum_r)
    maximum_c = numpy.minimum(y_true_maximum_c, y_pred_maximum_c)

    intersection = (maximum_r - minimum_r + 1) * (maximum_c - minimum_c + 1)

    y_true_area = (y_true_maximum_r - y_true_minimum_r + 1) * (y_true_maximum_c - y_true_minimum_c + 1)
    y_pred_area = (y_pred_maximum_r - y_pred_minimum_r + 1) * (y_pred_maximum_c - y_pred_minimum_c + 1)

    union = y_true_area + y_pred_area - intersection
    print("Calculated IoU = %5.2f"%(intersection/union))
    return intersection / union


def evaluate(y_true, y_pred, threshold=0.5):
    print("Mean-average precision calculation entered")
    y_true_indices = []
    y_pred_indices = []

    scores = []

    for y_pred_detection_index, y_pred_detection in enumerate(y_pred):
        for y_true_detection_index, y_true_detection in enumerate(y_true):
            score = intersection_over_union(y_true_detection, y_pred_detection)

            if score > threshold:
                y_true_indices += [y_true_detection_index]
                y_pred_indices += [y_pred_detection_index]

                scores += [score]

    scores = numpy.argsort(scores)[::-1]

    if scores.size == 0:
        tp = 0
        fp = len(y_pred)
        fn = len(y_true)
    else:
        y_true_matched_indices = []
        y_pred_matched_indices = []

        for score in scores:
            y_true_index = y_true_indices[score]
            y_pred_index = y_pred_indices[score]

            if (y_true_index not in y_true_matched_indices) and (y_pred_index not in y_pred_matched_indices):
                y_true_matched_indices += [y_true_index]
                y_pred_matched_indices += [y_pred_index]

        tp = len(y_true_matched_indices)
        fp = len(y_pred) - len(y_pred_matched_indices)
        fn = len(y_true) - len(y_true_matched_indices)

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    try:
        accuracy = (tp + fp) / scores.size
    except ZeroDivisionError:
        accuracy = 0.0

    print("Finished avg_precision calculation. Metrics: FN: %d, FP: %d, TP: %d" %(fn, fp, tp))
    print("Precision: %5.2f, Recall: %5.2f, Threshold: %5.2f, Accuracy: %5.2f"%(precision, recall, threshold, accuracy))
    

    return {
        "false negatives": fn,
        "false positives": fp,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "true positives": tp
    }
