
from math import sqrt
from PIL import Image, ImageEnhance
from box import iou_bboxes_xywh
from abc import ABCMeta, abstractmethod
from collections import Iterable

import cv2
import random
import numpy as np

# __all__ = ["image_augment"]


class Compose(object):
    def __init__(self, *transforms):
        super(Compose, self).__init__()
        for t in transforms:
            assert isinstance(t, Transform), \
                f"{t.__class__.__name__} must extend Transform."
        self.transforms = transforms

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        for t in self.transforms:
            hwc_img, gt_xywh, gt_cls = t(hwc_img, gt_xywh, gt_cls)
        return hwc_img, gt_xywh, gt_cls


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, hwc_img, gt_xywh=None, gt_clas=None):
        raise NotImplementedError(
            f"'__call__' not implement in class {self.__class__.__name__}")


class RandomDestory(Transform):
    """Random改变 亮度、对比度、颜色."""

    def __init__(self):
        super(RandomDestory, self).__init__()
        self.ops = [self.random_brightness,
                    self.random_contrast, self.random_color]

    def random_brightness(self, im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(im).enhance(e)

    def random_contrast(self, im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(im).enhance(e)

    def random_color(self, im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(im).enhance(e)

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        np.random.shuffle(self.ops)
        pil_im = Image.fromarray(hwc_img)
        for op in self.ops:
            pil_im = op(pil_im)
        return np.array(pil_im, dtype=hwc_img.dtype), gt_xywh, gt_cls


class RandomExpand(Transform):
    """
    随机填充. 对原图的外围进行填充 fill, 如果fill为None则填充0.
    Create a Large Background and do fill.

    :param gt_xywh: must normalized & dtype is float.

    :param thresh: 做随机填充比例
    :param max_ratio: 原图在填充图中的最大缩小比例，或称填充图放大比例.
    :param fill_values: 填充的像素值. type: tuple or list.
    :param xy_ratio_same: 长宽最大比例是否相同
    :return: hwc_image, gt_xywh, gt_cls
    """

    def __init__(self,
                 thresh=.5,
                 max_ratio=4.,
                 fill_values=None,
                 xy_ratio_same=True):
        super(RandomExpand, self).__init__()
        assert 0. <= thresh <= 1.
        assert max_ratio > 1.  # 填充生成的画布必须比原图大.
        if fill_values is not None:
            assert type(fill_values) in (tuple, list)

        self.thresh = thresh
        self.max_ratio = max_ratio
        self.fill_values = fill_values
        self.xy_ratio_same = xy_ratio_same

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        if random.random() < self.thresh:
            return hwc_img, gt_xywh, gt_cls

        assert gt_xywh is not None
        assert "float" in str(gt_xywh.dtype), \
            f"`gt_xywh` must do normalize and so it's dtype should be float, and now it's dtype is {gt_xywh.dtype}."

        h, w, c = hwc_img.shape
        ratio_x = np.random.uniform(1, self.max_ratio)
        ratio_y = ratio_x \
            if self.xy_ratio_same else np.random.uniform(1, self.max_ratio)

        np.random.uniform(1, self.max_ratio)

        oh = round(h * ratio_y)
        ow = round(w * ratio_x)

        offset_x = random.randint(0, ow - w)
        offset_y = random.randint(0, oh - h)

        out_img = np.zeros((oh, ow, c), dtype=hwc_img.dtype)

        if self.fill_values is not None and len(self.fill_values) == c:
            for i in range(len(self.fill_values)):
                out_img[..., i] = self.fill_values[i]

        out_img[offset_y: offset_y+h, offset_x: offset_x+w, :] = hwc_img

        gt_xywh[:, 0] = gt_xywh[:, 0] * w + offset_x / float(ow)
        gt_xywh[:, 1] = gt_xywh[:, 1] * h + offset_y / float(oh)
        gt_xywh[:, 2] = gt_xywh[:, 2] / float(ratio_x)
        gt_xywh[:, 3] = gt_xywh[:, 3] / float(ratio_y)

        return out_img, gt_xywh, gt_cls


class RandomCrop(Transform):
    """
        随机裁剪

    @params:
        constraints: [(min_iou1, max_iou1), (min_iou2, max_iou2), .... (min_iou_n, max_iou_n)]
    """

    def __init__(self,
                 min_life_box: int = 1, max_trial: int = 50,
                 scales: tuple = (.3, 1.), max_ratio: float = 2.,
                 constraints: tuple = ((0.1, 1.0), (0.3, 1.0), (0.5, 1.0),
                                       (0.7, 1.0), (0.9, 1.0), (0.0, 1.0))):
        super(RandomCrop, self).__init__()
        assert min_life_box >= 1
        assert max_trial >= 1
        assert isinstance(scales, Iterable) and len(scales) == 2

        self.scales = scales
        self.max_ratio = max_ratio
        self.max_trial = max_trial
        self.constraints = constraints
        self.min_life_box = min_life_box

    def crop_bbox_x1y1wh(self,
                         boxes_xywh: np.ndarray,
                         labels: np.ndarray,
                         crop_x1y1wh,
                         im_hw):
        """For Data aug.
        删除(置0)不在crop区域的box.
        """

        xmin, ymin, w, h = map(float, crop_x1y1wh)
        im_h, im_w = map(float, im_hw)

        boxes = boxes_xywh.copy()
        # xywh -> xyxy
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2.) * im_w, \
            (boxes[:, 0] + boxes[:, 2] / 2) * im_w

        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2.) * im_h, \
            (boxes[:, 1] + boxes[:, 3] / 2) * im_h

        crop_xyxy = np.array([xmin, ymin, xmin + w, ymin + h])
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2.

        # get [N, ] 代表crop的框是否包含了此物体
        mask = np.logical_and(
            crop_xyxy[:2] <= boxes_center, boxes_center <= crop_xyxy[2:]).all(axis=1)

        # align
        boxes[:, :2] = np.maximum(boxes[:, :2], crop_xyxy[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_xyxy[2:])
        # 左上角平移， 改变box坐标
        boxes[:, :2] -= crop_xyxy[:2]
        boxes[:, 2:] -= crop_xyxy[:2]
        # 判断坐标是否正常， 有点鸡肋.
        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))

        # mask: [N,] -> [N, 1]
        boxes = boxes * np.expand_dims(mask.astype(np.float32), axis=1)
        labels = labels * mask.astype(np.float32)

        # xyxy -> xywh
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2. / w, \
            (boxes[:, 2] - boxes[:, 0]) / w

        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2. / h, \
            (boxes[:, 3] - boxes[:, 1]) / h

        return boxes, labels, mask.sum()

    def __call__(self, hwc_img, gt_xywh, gt_cls):
        pil_img = Image.fromarray(hwc_img)
        w, h = pil_img.size

        crop_boxes = []
        for min_iou, max_iou in self.constraints:
            for _ in range(self.max_trial):
                # generator crop box.
                scale = random.uniform(self.scales[0], self.scales[1])
                aspect_ratio = random.uniform(max(1. / self.max_ratio, scale * scale),
                                              min(self.max_ratio, 1. / scale / scale))
                crop_h = int(h * scale / sqrt(aspect_ratio))
                crop_w = int(w * scale * sqrt(aspect_ratio))
                # crop_x, crop_y of left bottom.
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                crop_xywh_normed = np.array([[(crop_x + crop_w / 2.) / w,
                                              (crop_y + crop_h / 2.) / h,
                                              crop_w / float(w), crop_h / float(h)]], dtype=np.float)

                iou_ndarray = iou_bboxes_xywh(crop_xywh_normed, gt_xywh)
                if min_iou <= iou_ndarray.min() and max_iou >= iou_ndarray.max():
                    crop_boxes.append((crop_x, crop_y, crop_w, crop_h))
                    break

        # crop
        while crop_boxes:
            crop_box = crop_boxes.pop(random.randint(0, len(crop_boxes) - 1))
            out_crop_boxes, out_crop_labels, life_box_num = \
                self.crop_bbox_x1y1wh(gt_xywh, gt_cls, crop_box, (h, w))

            if life_box_num < self.min_life_box:
                continue

            # at this bbox is relative coord.
            pil_img = pil_img.crop((crop_box[0], crop_box[1],
                                    crop_box[0] + crop_box[2],
                                    crop_box[1] + crop_box[3])).resize(pil_img.size, Image.LANCZOS)

            return np.asarray(pil_img), out_crop_boxes, out_crop_labels

        return hwc_img, gt_xywh, gt_cls


class RandomInterpolationZoom(Transform):
    def __init__(self, size,
                 inter_method_tuple=(cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                     cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)):
        super(RandomInterpolationZoom, self).__init__()
        assert inter_method_tuple
        self.inter_method_tuple = inter_method_tuple
        if isinstance(size, tuple):
            assert len(size) == 2
            self.hw = tuple(size)
        else:
            self.hw = (size, size)

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        inter_method = self.inter_method_tuple[
            random.randint(0, len(self.inter_method_tuple) - 1)]

        h, w = hwc_img.shape[:2]
        scale_x = self.hw[1] / float(w)
        scale_y = self.hw[0] / float(h)

        hwc_img = cv2.resize(hwc_img, None, None,
                             scale_x, scale_y, interpolation=inter_method)

        return hwc_img, gt_xywh, gt_cls


class RandomHorizontalFlip(Transform):
    def __init__(self, thresh=.5):
        super(RandomHorizontalFlip, self).__init__()
        assert 0. <= thresh <= 1.
        self.thresh = thresh

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        if random.random() > self.thresh:
            hwc_img = hwc_img[:, ::-1, :]
            gt_xywh[:, 0] = 1. - gt_xywh[:, 0]
        return hwc_img, gt_xywh, gt_cls


class RandomVerticalFlip(Transform):
    def __init__(self, thresh=.5):
        super(RandomVerticalFlip, self).__init__()
        assert 0. <= thresh <= 1.
        self.thresh = thresh

    def __call__(self, hwc_img, gt_xywh=None, gt_cls=None):
        if random.random() > self.thresh:
            hwc_img = hwc_img[::-1, ::, :]
            gt_xywh[:, 1] = 1. - gt_xywh[:, 1]
        return hwc_img, gt_xywh, gt_cls


if __name__ == "__main__":
    pass
