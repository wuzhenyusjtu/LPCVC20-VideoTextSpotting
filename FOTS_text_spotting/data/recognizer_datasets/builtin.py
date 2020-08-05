import os

from detectron2.data.datasets.register_coco import register_coco_instances

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "sample_dataset_train": ("../../../sample_dataset/train_images1", "../../../sample_dataset/train_16pt.json"),
    "sample_dataset_val": ("../../../sample_dataset/test_images1", "../../../sample_dataset/test_16pt.json"),
    "case_sensitive_sample_dataset_train": ("../../../new_sample_dataset/train_images", "../../../new_sample_dataset/train_16pt.json"),
    "case_sensitive_sample_dataset_val": ("../../../new_sample_dataset/test_images", "../../../new_sample_dataset/test_16pt.json"),
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
    "synthtext": ("SynthText/train_images","SynthText/annotations/train.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="path_to_dataset"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()