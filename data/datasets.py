import os
import logging
import numpy as np
import paddle as pp

from PIL import Image
# from .annotation import get_cname2cid_dict_from_txt, voc_parse
from annotation import get_cname2cid_dict_from_txt, voc_parse


class VOCDataset(pp.io.Dataset):
    """
          {
            "im_path": im_file,
            "im_id": im_id,
            "im_h": im_h,
            "im_w": im_w,
            "im_d": im_d,
            "is_crowd": gt_is_crowd,
            "difficult": gt_difficult,
            "gt_class": gt_class,
            "gt_xywh": gt_bbox_xywh,
            "gt_poly": []
        }
    """

    def __init__(self,
                 cfg,
                 root_dir: str,
                 list_path: str,
                 label_path: str,
                 label_dir: str = "",
                 transform=None):
        super(VOCDataset, self).__init__()
        assert cfg is not None

        self.cfg = cfg
        self.image_dir = root_dir
        self.label_dir = root_dir if "" == label_dir else label_dir

        self._cname2cid_dict = get_cname2cid_dict_from_txt(label_path)
        self.record_info_list = voc_parse(self.image_dir, list_path,
                                          self.cname2cid_dict,
                                          label_dir=self.label_dir)
        self.transform = transform

    def random_shuffle_boxes(self, gt_boxes, gt_cls):
        assert len(gt_boxes) == len(gt_cls)
        random_idx = np.random.permutation(len(gt_boxes))
        return gt_boxes[random_idx], gt_cls[random_idx]

    def __getitem__(self, idx):
        record_info = self.record_info_list[idx]
        im_path = record_info["im_path"]
        h = record_info["im_h"]
        w = record_info["im_w"]
        gt_cls = record_info["gt_class"]
        gt_xywh = record_info["gt_xywh"]

        hwc_im = np.array(
            Image.open(os.path.join(self.image_dir, im_path)).convert("RGB"), dtype=np.uint8)
        gt_xywh = gt_xywh.astype(np.float32)
        gt_xywh[..., 0::2] /= float(w)
        gt_xywh[..., 1::2] /= float(h)
        gt_cls = gt_cls.astype(np.int64)

        if self.transform is not None:
            hwc_im, gt_xywh, gt_cls = self.transform(hwc_im, gt_xywh, gt_cls)
        gt_xywh, gt_cls = self.random_shuffle_boxes(gt_xywh, gt_cls)
        return hwc_im, gt_xywh, gt_cls, (h, w)

    def __len__(self):
        return len(self.record_info_list)

    @property
    def cname2cid_dict(self):
        return self._cname2cid_dict


if __name__ == "__main__":
    import transforms as T

    root_dir = r"D:\workspace\DataSets\det\Insect"
    list_path = r"D:\workspace\DataSets\det\Insect\ImageSets\train_list.txt"
    label_path = r"D:\workspace\DataSets\det\Insect\ImageSets\label_list.txt"
    transforms = T.Compose(
        T.RandomDestory(),
        T.RandomExpand(),
        T.RandomCrop(),
        T.RandomHorizontalFlip(),
        T.RandomInterpolationZoom(608)
    )

    voc_dataset = VOCDataset(
        [], root_dir, list_path, label_path, transform=transforms)

    for i in range(100):
        hwc_im, gt_xywh, gt_cls, scale = voc_dataset.__getitem__(i)
        print(i, hwc_im.shape, gt_xywh.shape, gt_cls.shape, scale)
