import torch
from argparse import ArgumentParser
from detectron2 import model_zoo
from detectron.detectron2_run import create_config_from_params, get_classes_list, StartTraining
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import io
import streamlit as st
from PIL import Image






class NctModel:

    def __init__(self):
        arg_parser = ArgumentParser()
        arg_parser.add_argument("--op_name", "-op", type=str, default="20190917")
        arg_parser.add_argument("--path_location", "-p", type=str, default="crossval_full")
        arg_parser.add_argument("--op_phase", "-ph", type=str, default="FIRST_INCISION")
        args = arg_parser.parse_args()
        self.classes_list = get_classes_list(args.op_phase)

        self.config = create_config_from_params(
            op_name=args.op_name,
            path_location=args.path_location,
            op_phase=args.op_phase,
            classes_list=self.classes_list,
        )
        self.cfg = self.get_trained_cfg()
        self.trainer = StartTraining(self.config)
        self.instruments_metadata = self.trainer.register_dataset_and_metadata(
            path_to_data=self.config["NEW"]["PATH_TO_DATA"], classes_list=self.classes_list
        )

    def get_trained_cfg(self):


        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        train_name = "resection_train"
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.config['TRAIN']['NUM_WORKERS']
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = self.config['TRAIN']['BATCH']
        cfg.SOLVER.BASE_LR = self.config['TRAIN']['BASE_LR']
        cfg.SOLVER.MAX_ITER = self.config['TRAIN']['MAXITER']
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config['TRAIN'][
            'ROI_HEADS_BATCH_SIZE_PER_IMAGE']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes_list)
        cfg.OUTPUT_DIR = self.config['TRAIN']['OUTPUT_DIR']
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.config["TRAIN"]["NUM_WORKERS"]

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config["TRAIN"][
            "ROI_HEADS_BATCH_SIZE_PER_IMAGE"
        ]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config["NEW"]["CLASSES_LIST"])
        cfg.OUTPUT_DIR = self.config["TRAIN"]["OUTPUT_DIR"]
        cfg.MODEL.WEIGHTS = os.path.join(f'../{cfg.OUTPUT_DIR}', self.config["VAL"]["NAME_MODEL_FILE"])
        # cfg.MODEL.WEIGHTS = 'https://drive.google.com/file/d/147xNvdXebqHW8MePFCZzjiUxr_E8s1LT/view?usp=sharing'
        # cfg.MODEL.WEIGHTS = "https://drive.google.com/u/0/uc?export=download&confirm=bhxW&id=147xNvdXebqHW8MePFCZzjiUxr_E8s1LT"
        # af = requests.get("https://drive.google.com/u/0/uc?export=download&confirm=bhxW&id=147xNvdXebqHW8MePFCZzjiUxr_E8s1LT")

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold
        cfg.DATASETS.TEST = ("resection_val",)

        print(f"taken {cfg.MODEL.WEIGHTS}")
        return cfg

    def model_predict(self, im):
        # classes_list = get_classes_list(args.op_phase)
        #
        # config = create_config_from_params(
        #     op_name=args.op_name,
        #     path_location=args.path_location,
        #     op_phase=args.op_phase,
        #     classes_list=classes_list,
        # )
        # cfg = get_trained_cfg(config, classes_list)
        # trainer = StartTraining(config)
        # instruments_metadata = trainer.register_dataset_and_metadata(
        #     path_to_data=config["NEW"]["PATH_TO_DATA"], classes_list=classes_list
        # )
        # dataset_dicts = trainer.get_balloon_dicts(
        #     f'{config["NEW"]["PATH_TO_DATA"]}val', json_with_description_name=config["NEW"]["JSON_DESCRIPTION"]
        # )

        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=self.instruments_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
        )
        # outputs["instances"]._fields["pred_masks"].data.numpy()
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(1024, 1280)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        # cv2_imshow(out.get_image()[:, :, ::-1])

        buf = io.BytesIO()
        plt.savefig(
            buf,
            bbox_inches="tight",
            pad_inches=0,
            dpi=600,
        )
        buf.seek(0)
        im = Image.open(buf)
        # im.show()
        # plt.show(block=False)
        # plt.close("all")
        # buf.close()
        return im


# im = cv2.imread("../data/output/crossval_full/20190917/val/20190917_FIRST_INCISION_00000.png")
# model = NctModel()
# image = model.model_predict(im)
# image.show()


