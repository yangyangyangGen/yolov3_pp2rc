import os
import logging
import numpy as np

from xml.etree import ElementTree as ET

__all__ = ["get_cname2cid_dict_from_txt", "voc_parse"]


def get_cname2cid_dict_from_txt(label_list_txt: str,
                                start_idx: int = 0) -> dict:
    assert os.path.exists(label_list_txt), f"{label_list_txt} not exists."

    with open(label_list_txt, "r", encoding="utf-8") as fr:
        content = [line.strip() for line in fr.readlines() if len(line)]
    # content_splited = filter(lambda line: len(line), content_splited)  # filter space.
    return {cname: start_idx + idx for idx, cname in enumerate(content)}


def voc_parse(root_dir: str,
              list_path: str,
              cname2cid_map: dict,
              label_dir: str = "",
              split_character: str = " ") -> list:
    """
        {
            "im_path":
            "im_id":
            "im_h":
            "im_w":
            "im_d":
            "is_crowd":
            "difficult":
            "gt_class":
            "gt_xywh":
            "gt_poly":
        }
    """

    label_dir = label_dir if "" != label_dir else root_dir

    assert isinstance(cname2cid_map, dict), \
        "cname2cid map is not isinstance dict."
    assert os.path.exists(list_path), \
        f"{list_path} not exists."

    image_id = 0

    records_list = []
    with open(list_path, "r", encoding="utf-8") as fr:
        for im_id, line in enumerate(fr):
            # 1. Check.
            if not len(line):
                continue
            line = line.rstrip().split(split_character)
            if len(line) != 2:
                logging.warning(
                    f"{line} length not equal 2. format not parse.")
                continue
            image_relative_path, anno_relative_path = line

            image_abspath = root_dir + os.sep + image_relative_path
            anno_abspath = label_dir + os.sep + anno_relative_path

            not_exists_files_list = [file for file in [image_abspath, anno_abspath]
                                     if not os.path.exists(file)]

            if len(not_exists_files_list):
                logging.warning(f"{not_exists_files_list} not exists!!!")

            # 2. Parse.
            root_tree = ET.parse(anno_abspath)
            _id = root_tree.find("id")
            im_id = _id if _id is not None else image_id
            im_file = image_abspath

            size_ele = root_tree.find("size")
            im_w = int(size_ele.find("width").text)
            im_h = int(size_ele.find("height").text)
            im_d = int(size_ele.find("depth").text)

            objs = root_tree.findall('object')
            gt_class = np.zeros((len(objs), ), dtype=np.int32)
            gt_bbox_xywh = np.zeros((len(objs), 4), dtype=np.float32)
            gt_difficult = np.zeros((len(objs), ), dtype=np.int32)
            gt_is_crowd = np.zeros((len(objs), ), dtype=np.int32)

            for i, obj in enumerate(objs):
                _difficult = obj.find("difficult")
                if _difficult is None:
                    _difficult = 0
                else:
                    _difficult = int(_difficult.text)

                _is_crowd = obj.find("is_crowd")
                if _is_crowd is None:
                    _is_crowd = 0
                else:
                    _is_crowd = int(_is_crowd.text)

                # get cls, box.
                _cname = obj.find('name').text
                # todo: -> cname2id_map.get(_cname, -1) ???
                _cid = cname2cid_map[_cname]

                bndbox_ele = obj.find("bndbox")
                assert bndbox_ele

                _xmin = max(float(bndbox_ele.find("xmin").text), 0.)
                _ymin = max(float(bndbox_ele.find("ymin").text), 0.)
                _xmax = min(float(bndbox_ele.find("xmax").text), im_w - 1.)
                _ymax = min(float(bndbox_ele.find("ymax").text), im_h - 1.)

                gt_bbox_xywh[i] = [(_xmin + _xmax) / 2., (_ymin + _ymax) / 2.,
                                   (_xmax - _xmin) + 1., (_ymax - _ymin) + 1.]
                gt_is_crowd[i] = _is_crowd
                gt_difficult[i] = _difficult
                gt_class[i] = _cid

            record_desc = {
                "im_path": im_file,
                "im_id": im_id,
                "im_h": im_h,
                "im_w": im_w,
                "im_d": im_d,
                "is_crowd": gt_is_crowd,
                "difficult": gt_difficult,
                "gt_class": gt_class,
                "gt_xywh": gt_bbox_xywh,
                "gt_poly": [],
            }

            if len(objs):
                records_list.append(record_desc)

    return records_list


if __name__ == "__main__":

    fpath = r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt"
    cname2cid_map = get_cname2cid_dict_from_txt()
    record_list = voc_parse(cname2cid_map, fpath)
    print(len(record_list))
    print(record_list[0])

    records = voc_parse(cname2cid_map, fpath)
    print(type(records))
    print(records[0])
