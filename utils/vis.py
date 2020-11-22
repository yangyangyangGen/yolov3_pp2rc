import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 150


def show_image2xywh_list(image2gtxywh_list: list, rows: int, cols: int):
    N = len(image2gtxywh_list)
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col + 1
            plt.subplot(rows, cols, idx)
            image, gt_xywh = image2gtxywh_list[idx-1]
            image = cv2.UMat(image).get()
            for xywh in gt_xywh:
                x, y, w, h = map(int, xywh)
                x1, y1 = (x - w//2) + 1, (y - h//2) + 1
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(image, (x1, y1), (x2, y2),  (0, 0, 255), 2)
            plt.imshow(image)


def show_image2xywh_one(hwc_image: np.ndarray,
                        gt_xywh: list):

    for xywh in gt_xywh:
        x, y, w, h = map(int, xywh)
        x1, y1 = (x - w//2) + 1, (y - h//2) + 1
        x2, y2 = x1 + w, y1 + h
        print((x1, y1), (x2, y2))
        cv2.rectangle(hwc_image, (x1, y1), (x2, y2),  (0, 0, 255), 2)
    plt.imshow(hwc_image)
