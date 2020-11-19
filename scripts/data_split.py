import os


def generator_image_label_list(root_dir: str,
                               image_dir: str,
                               label_dir: str,
                               image_suffix: str = "jpeg",
                               label_suffix: str = "xml"):

    image_path = os.path.join(root_dir, image_dir)
    label_path = os.path.join(root_dir, label_dir)

    assert os.path.exists(image_path) and os.path.isdir(image_path)
    assert os.path.exists(label_path) and os.path.isdir(label_path)

    image2label_list = []
    for image_name in os.listdir(image_path):
        if not image_name.endswith(image_suffix):
            continue

        label_name = image_name.split(".")[0] + "." + \
            label_suffix.replace(".", "")

        label_abspath = label_path + os.sep + label_name
        assert os.path.exists(label_abspath), f"{label_abspath} not exists."

        image_relative_path = os.path.join(image_dir, image_name)
        label_relative_path = os.path.join(label_dir, label_name)

        image2label_list.append((image_relative_path, label_relative_path))

    return image2label_list


def dump_txt(image2label_list, txt_path):

    with open(txt_path, "w", encoding="utf-8") as fw:
        [fw.write(f"{image} {anno}\n") for (image, anno) in image2label_list]


if __name__ == "__main__":
    root_dir = r"D:\workspace\DataSets\det\Insect"

    image_dir = "JPEGImages"
    anno_dir = "Annotations"
    dump_txt_dir = root_dir + os.sep + "ImageSets"

    train_image2label_list = generator_image_label_list(
        root_dir,
        os.path.join(image_dir, "train"),
        os.path.join(anno_dir, "train"),
        "jpeg", "xml")

    valid_image2label_list = generator_image_label_list(
        root_dir,
        os.path.join(image_dir, "val"),
        os.path.join(anno_dir, "val"),
        "jpeg", "xml")

    test_image2label_list = generator_image_label_list(
        root_dir,
        os.path.join(image_dir, "test"),
        os.path.join(anno_dir, "test"),
        "jpeg", "xml")

    dump_txt(train_image2label_list,
             dump_txt_dir + os.sep + "train_list.txt")
    dump_txt(valid_image2label_list,
             dump_txt_dir + os.sep + "val_list.txt")
    dump_txt(test_image2label_list,
             dump_txt_dir + os.sep + "test_list.txt")
