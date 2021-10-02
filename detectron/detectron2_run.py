import torch
import detectron2
from detectron2.utils.logger import setup_logger
import random
import cv2
from albumentations import VerticalFlip
from typing import List
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import yaml
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import matplotlib.patches as patches
from collections import defaultdict
from detectron.dataset_new_data import mask_to_class
import detectron.detectron2_evaluation as eval
from configs.config_generator import create_config_from_params
from argparse import ArgumentParser
from detectron.phases import OpPhase
from detectron.utils.nrrd_mapping import (
    mapping_layers,
    nrrd_mapping,
    pedickle_package_mapping,
    mapping_layers_pedickle_package,
    mapping_layers_vascular_dissection,
    mapping_vascular_dissection,
    mapping_layer_mesocolon_gerota,
    mapping_mesocolon_gerota,
    mapping_tme,
    mapping_layers_tme,
)

# saving all logs
setup_logger("./output/saved_logs.log")


def strong_aug(p: int = 1):
    """
    Function to perform  an mirroring on the x axis based on the albumentation func
    Args:
        p:the probability that the image will be flipped

    Returns:
        vertically flipped image based on a probability
    """
    return VerticalFlip(p=p)


def rotate_and_flip(image: np.ndarray):
    """
    Performs firstly a flip along the x axis and than a 90 degree rotation
    Args:
        image: image that should be processed

    Returns:
        a flipped and rotated image
    """
    augmentation = strong_aug(p=1)
    data = {"image": image}
    augmented = augmentation(**data)
    return np.rot90(augmented["image"], axes=(0, 1), k=-1)


class StartTraining:
    """
    Class that contains methods for training/evaluation a model
    """

    def __init__(self, config_name: str):
        self.yaml_config = config_name

    @staticmethod
    def get_balloon_dicts(
        img_dir: str,
        json_with_description_name: str = "dataset_registration_detectron2_multilabeled_all_layers.json",
    ) -> List[dict]:
        """
        Creating a description for each image in the image dir according to the json description
        While extracting the description of from the json, Bounding Boxes will be also calculated

        Args:
            img_dir: dir where the images are located
            json_with_description_name: description json of every single image in the img_dir

        Returns:
            List with description of every image
            F.e
            [{'height': 540, 'width': 960, 'file_name':
            '/instruments/train/frame_00000.png',
             'image_id': 0, 'annotations': [{'bbox': [0.0, 218.12345679012344,
              630.0987654320987, 539.0],
             'bbox_mode': <BoxMode.XYXY_ABS: 0>,
             'segmentation': [[0.5, 521.5, 435.537037037037, 297.6358024691358,
                , 355.2901234567901, 415.537037037037, 125.5, 539.5,
                1.5864197530864197, 538.9938271604938]],
                'category_id': 2}, ..... ]


        """
        json_file = os.path.join(img_dir, json_with_description_name)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            try:
                record = {}

                filename = os.path.join(img_dir, v["filename"])
                record["height"] = v["height"]
                record["width"] = v["width"]
                record["file_name"] = filename
                record["image_id"] = idx
                annos = v["regions"]
                objs = []
                for _, anno in annos.items():
                    assert not anno["region_attributes"]
                    anno = anno["shape_attributes"]
                    py = anno["all_points_x"]
                    px = anno["all_points_y"]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": anno["category_id"],
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
            except Exception as ex:
                filename = os.path.join(img_dir, v["filename"])
                print(f"could not process {filename} {ex}")
        return dataset_dicts

    def register_dataset_and_metadata(self, path_to_data: str, classes_list: List[str]):
        """
        Registrs the dataset according to the https://detectron2.readthedocs.io/tutorials/datasets.html

        Args:
            path_to_data: path to the folder, where the train and validation forlder is located
                          folder train has images for training and a json that describes the
                          data (bounding boxes, labels etc)
            classes_list: is a list of all possible labels that might occur

        Returns:
            a registration Metadata object that can be further used for training/testing/validation
            it is similar to a Dataloader

        """
        for d in ["train", "val"]:
            DatasetCatalog.register(
                "resection_" + d,
                lambda mode=d: self.get_balloon_dicts(path_to_data + mode),
            )
            MetadataCatalog.get("resection_" + d).set(thing_classes=classes_list)
        instruments_metadata = MetadataCatalog.get("resection_train")

        return instruments_metadata

    def check_registration(
        self,
        instruments_metadata: Metadata,
        path_to_training_data: str,
        json_with_desription_name: str = "dataset_registration_detectron2.json",
        to_write: str = "./output/crossval/20190917/img/",
    ) -> None:

        """
        testing the registred dataset and its metadata by visualising the results of the annotation on the image


        Args:
            instruments_metadata: the registred data
            path_to_training_data:
            to_write:
            json_with_desription_name:

        """
        dataset_dicts = self.get_balloon_dicts(
            path_to_training_data, json_with_description_name=json_with_desription_name
        )

        for d in tqdm(random.sample(dataset_dicts, 10)):
            img = cv2.imread(d["file_name"])
            print(f'Took: {d["file_name"]}')
            visualizer = Visualizer(
                img[:, :, ::-1], metadata=instruments_metadata, scale=0.5
            )
            vis = visualizer.draw_dataset_dict(d)

            if not cv2.imwrite(
                f'{to_write}{os.path.basename(d["file_name"])}',
                vis.get_image()[:, :, ::-1],
            ):
                raise Exception("Could not write image")
            print(f'wrote to {to_write}{os.path.basename(d["file_name"])}')

    def _start_training(
        self, train_name: str = "resection_train", classes_list=None
    ) -> None:
        """
        Starting training
        Args:
            train_name: dataset which was registred
            classes_list: unique classes list that contains the dataset

        Returns:

        """
        if classes_list is None:
            classes_list = [
                "first incision",
                "plane first incision",
                "incised area",
                "instrumenfrenestrated bipolar forceps",
                "instrumenpermanent cautery hook",
                "instrument cadiere forceps",
                "not needed",
                "triangle upperr corner",
                "triangle lower corners",
                "instrument monopolar curved scissors",
                "compress",
                "instrument assistant",
            ]
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        # set up parameters from the config file
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.yaml_config["TRAIN"]["NUM_WORKERS"]
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.SOLVER.IMS_PER_BATCH = self.yaml_config["TRAIN"]["BATCH"]
        cfg.SOLVER.BASE_LR = self.yaml_config["TRAIN"]["BASE_LR"]
        cfg.SOLVER.MAX_ITER = self.yaml_config["TRAIN"]["MAXITER"]
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.yaml_config["TRAIN"][
            "ROI_HEADS_BATCH_SIZE_PER_IMAGE"
        ]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_list)

        # where the model is stored or where we want to save it
        cfg.OUTPUT_DIR = self.yaml_config["TRAIN"]["OUTPUT_DIR"]
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


def start_training(yaml_config: dict, train: bool, validation: bool) -> None:
    """
     Starting training/validation based on the train param
    :param yaml_config: contains the config for the network
    :param train: set ups whether to train a network or to validate it
    :param validation: set ups whether to train a network or to validate it
    :return:
        nothing to return
    """

    new_data = yaml_config["NEW_DATA"]
    if new_data:
        classes_list = yaml_config["NEW"]["CLASSES_LIST"]
        path_to_data = yaml_config["NEW"]["PATH_TO_DATA"]

    else:
        classes_list = yaml_config["OLD"]["CLASSES_LIST"]
        path_to_data = yaml_config["OLD"]["PATH_TO_DATA"]

    starter = StartTraining(yaml_config)
    evaluation_obj = eval.Evaluation(yaml_config)
    instruments_metadata = starter.register_dataset_and_metadata(
        path_to_data, classes_list=classes_list
    )

    # choosing processing mode either train or validation
    if train:
        starter._start_training(train_name="resection_train", classes_list=classes_list)

    if validation:
        evaluation_obj.inference_on_trained_mode(
            instruments_metadata,
            path_to_data,
            json_description=yaml_config["NEW"]["JSON_DESCRIPTION"],
            model_location=yaml_config["VAL"]["NAME_MODEL_FILE"],
            visualise=True,
        )

        evaluation_obj.dump_metrics(location_to_save=yaml_config["VAL"]["SAVE_JSON"])


def get_classes_list(op_phase: str) -> List[str]:
    """
    get the list of labels corresponding to a given op_phase
    :param op_phase: like "TME"
    :return:
        list of labels
    """
    classes_list = None
    if op_phase == OpPhase.FIRST_INCISION.name:
        classes_list = list(mapping_layers.keys())
    elif op_phase == OpPhase.PEDICLE_PACKAGE.name:
        classes_list = list(mapping_layers_pedickle_package.keys())
    elif op_phase == OpPhase.VASCULAR_DISSECTION.name:
        classes_list = list(mapping_layers_vascular_dissection.keys())
    elif op_phase == OpPhase.MESOCOLON_GEROTA.name:
        classes_list = list(mapping_layer_mesocolon_gerota.keys())
    elif op_phase == OpPhase.TME.name:
        classes_list = list(mapping_layers_tme.keys())
    return classes_list


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--op_name", "-op", type=str, default="20190917")
    arg_parser.add_argument("--path_location", "-p", type=str, default="crossval_full")
    arg_parser.add_argument("--op_phase", "-ph", type=str, default="FIRST_INCISION")
    args = arg_parser.parse_args()
    classes_list = get_classes_list(args.op_phase)

    config = create_config_from_params(
        op_name=args.op_name,
        path_location=args.path_location,
        op_phase=args.op_phase,
        classes_list=classes_list,
    )
    start_training(yaml_config=config, train=False, validation=True)


if __name__ == "__main__":
    main()
