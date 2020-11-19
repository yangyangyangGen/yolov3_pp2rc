import numpy as np


def iou_bboxes_xywh(bbox1: np.ndarray, bbox2: np.ndarray):
    """
        x y is of center point.
        1: N、 N: N、 1:1
    """
    assert bbox1.shape[-1] == 4 and bbox2.shape[-1] == 4

    bbox1_xmin, bbox1_xmax = bbox1[:, 0] - \
        bbox1[:, 2] / 2., bbox1[:, 0] + bbox1[:, 2] / 2.
    bbox1_ymin, bbox1_ymax = bbox1[:, 1] - \
        bbox1[:, 3] / 2., bbox1[:, 1] + bbox1[:, 3] / 2.

    bbox2_xmin, bbox2_xmax = bbox2[:, 0] - \
        bbox2[:, 2] / 2., bbox2[:, 0] + bbox2[:, 2] / 2.
    bbox2_ymin, bbox2_ymax = bbox2[:, 1] - \
        bbox2[:, 3] / 2., bbox2[:, 1] + bbox2[:, 3] / 2.

    bbox1_area = (bbox1_xmax - bbox1_xmin + 1.) * \
        (bbox1_ymax - bbox1_ymin + 1.)
    bbox2_area = (bbox2_xmax - bbox2_xmin + 1.) * \
        (bbox2_ymax - bbox2_ymin + 1.)

    inter_xmin = np.maximum(bbox1_xmin, bbox2_xmin)
    inter_ymin = np.maximum(bbox1_ymin, bbox2_ymin)
    inter_xmax = np.minimum(bbox2_xmax, bbox2_xmax)
    inter_ymax = np.minimum(bbox1_ymax, bbox2_ymax)

    inter_h = (inter_ymax - inter_ymin + 1.)
    inter_w = (inter_xmax - inter_xmin + 1.)
    #
    inter_h = np.clip(inter_h, a_min=0, a_max=None)
    inter_w = np.clip(inter_w, a_min=0, a_max=None)

    inter_area = inter_h * inter_w

    return inter_area / (bbox1_area + bbox2_area - inter_area).astype("float")
