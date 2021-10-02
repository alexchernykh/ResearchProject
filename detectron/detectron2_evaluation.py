import numpy as np
from collections import defaultdict
import torch
import os
import pandas as pd
from tqdm import tqdm
import json
import random
import cv2
from PIL import Image
from typing import Dict
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2
from detectron2.config.config import CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.utils.visualizer import ColorMode
from detectron.dataset_new_data import get_yaml_config
import detectron.detectron2_run as training
from detectron.dataset_new_data import mask_to_class
import matplotlib.patches as patches
from detectron.phases import OpPhase
from detectron.utils.nrrd_mapping import (
    mapping_layers,
    mapping_layers_pedickle_package,
    mapping_layers_vascular_dissection,
    mapping_layer_mesocolon_gerota,
    mapping_mesocolon_gerota,
    mapping_tme,
    mapping_layers_tme,
)



class Metrics:
    """
    Metrics for evaluation of the trained model
    For the evaluation two metrics where used:
    1. jaccard index
    2. dice
    """

    @staticmethod
    def jaccard_ternaus(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        is the implementation of the jaccard index by Ternaus

        Returns:
            a floating point value as a score
        """
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        return (intersection + 1e-15) / (union + 1e-15)

    @staticmethod
    def dice_ternaus(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        dice coefficient by Ternaus

        Returns:
            floating point value representing the dice score
        """
        return (2 * (y_true * y_pred).sum() + 1e-15) / (
            y_true.sum() + y_pred.sum() + 1e-15
        )


class Evaluation:
    def __init__(self, config_name: str):
        self.jaccard_default_dict = defaultdict(list)
        self.config_name = config_name
        self.metric = Metrics()
        self.yaml_config = config_name
        self.class_list = self.yaml_config["NEW"]["CLASSES_LIST"]
        self.cfg = self.make_config()
        self.dir_to_safe = self.yaml_config["VAL"]["DIR_TO_SAVE"]

    def map_value_to_key(
        self, dict_layers_encoding: Dict[str, int], number_found_layer: int
    ) -> str:
        """
        For example get from
                'first incision': 0,
        based on the 0, the name 'first incision'
        Args:
            dict_layers_encoding: dict wherer will be performed the iteration
            number_found_layer: the label number.

        Returns:
            the original name of the label
                f.e 'first incision'
        """
        for layer_name, layer_num in dict_layers_encoding.items():
            if layer_num == number_found_layer:
                return layer_name

    def get_original_label(self, number_found_layer: int) -> str:
        """
        Get the original name based on the layer name
        Args:
            number_found_layer: the layers number

        Returns:
            the origianl layer name
        """
        if self.config_name["OP_PHASE"] == OpPhase.FIRST_INCISION.name:
            string_label_name = self.map_value_to_key(
                mapping_layers, number_found_layer
            )
        if self.config_name["OP_PHASE"] == OpPhase.PEDICLE_PACKAGE.name:
            string_label_name = self.map_value_to_key(
                mapping_layers_pedickle_package, number_found_layer
            )
        if self.config_name["OP_PHASE"] == OpPhase.VASCULAR_DISSECTION.name:
            string_label_name = self.map_value_to_key(
                mapping_layers_vascular_dissection, number_found_layer
            )
        if self.config_name["OP_PHASE"] == OpPhase.MESOCOLON_GEROTA.name:
            string_label_name = self.map_value_to_key(
                mapping_layer_mesocolon_gerota, number_found_layer
            )
        if self.config_name["OP_PHASE"] == OpPhase.TME.name:
            string_label_name = self.map_value_to_key(
                mapping_layers_tme, number_found_layer
            )

        return string_label_name

    def make_config(self) -> CfgNode:
        """
        generating a detectron config for the network
        :return:
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        train_name = "resection_train"
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.yaml_config["TRAIN"]["NUM_WORKERS"]
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = self.yaml_config["TRAIN"]["BATCH"]
        cfg.SOLVER.BASE_LR = self.yaml_config["TRAIN"]["BASE_LR"]
        cfg.SOLVER.MAX_ITER = self.yaml_config["TRAIN"]["MAXITER"]
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.yaml_config["TRAIN"][
            "ROI_HEADS_BATCH_SIZE_PER_IMAGE"
        ]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_list)
        cfg.OUTPUT_DIR = self.yaml_config["TRAIN"]["OUTPUT_DIR"]
        return cfg

    def plot_inly_predicted_mask(
        self, list_class_mask: list, filename: str, class_name: int, saving
    ) -> None:
        """
        Masks of the predicted class (class_name) will be cobined together (semantic segmentation) and
        saved to a specified locations

        :return:
            None, only the masks of a predicted class will be saved to a specified location
        """
        if len(list_class_mask) > 0:
            mask = self.get_right_mask(filename, class_name)
            predicted_mask = np.logical_or.reduce(list_class_mask)
            ternaus_jaccard = (
                self.metric.jaccard_ternaus(
                    torch.from_numpy(mask.astype(bool)).float(),
                    torch.from_numpy(predicted_mask).float(),
                )
                .numpy()
                .tolist()
            )
            print(f"{filename}, Layer {class_name} {round(ternaus_jaccard, 4)}")

            original_label_name = self.get_original_label(class_name)
            self.jaccard_default_dict[os.path.basename(filename)].append(
                {str(original_label_name): round(ternaus_jaccard, 4)}
            )

        predicted_mask = np.logical_or.reduce(list_class_mask)

        image_stacked_number = os.path.basename(filename)[-6:-4]
        if image_stacked_number.startswith("_"):
            image_stacked_number = f"_{str(image_stacked_number[1:]).zfill(5)}"

        if saving:
            Image.fromarray(
                mask_to_class(predicted_mask.astype("uint8")), mode="L"
            ).save(
                f"../detectron/output/{self.dir_to_safe}/images_predicted_multilabeled/layerwise_masks/{class_name}/"
                f"{os.path.basename(filename)[:-4]}_{str(class_name).zfill(2)}.png"
            )
            print(
                f"Saved ../detectron/output/{self.dir_to_safe}/images_predicted_multilabeled/layerwise_masks/{class_name}/"
                f"{os.path.basename(filename)[:-4]}_{str(class_name).zfill(2)}.png"
            )

    def plotting_each_layer(
        self,
        list_class_mask,
        list_class_box,
        filename,
        output,
        class_name,
        saving,
        mask,
    ) -> None:
        """
        Performs plotting of a certain layerresults

        image -> mask_gt -> predicted layer

        Args:
            list_class_mask:  list of mask that were predicted by detectron
            list_class_box: list of mask that were predicted by detectron
            filename: filename to access than the original gt masks
            output: original image
            class_name: name of the layer
            saving: save results or nots

        Returns:
            nothing to return only visualising

        """
        if len(list_class_mask) > 0:
            mask = self.get_right_mask(filename, class_name)
            predicted_mask = np.logical_or.reduce(list_class_mask)
            ternaus_jaccard = (
                self.metric.jaccard_ternaus(
                    torch.from_numpy(mask.astype(bool)).float(),
                    torch.from_numpy(predicted_mask).float(),
                )
                .numpy()
                .tolist()
            )
            print(f"{filename}, Layer {class_name} {round(ternaus_jaccard, 4)}")

            original_label_name = self.get_original_label(class_name)
            self.jaccard_default_dict[os.path.basename(filename)].append(
                {str(original_label_name): round(ternaus_jaccard, 4)}
            )
            # print(ternaus_jaccard)

            if saving:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                ax1.set_title("Original Image")
                ax2.set_title(f"Original Mask for Layer{class_name}")

                # mask = self.get_right_mask(filename, class_name)
                ax2.imshow(mask)
                # predicted_mask = np.logical_or.reduce(list_class_mask)
                ax3.set_title(f"Predicted mask for Layer{class_name}")

                ax3.imshow(predicted_mask)
                for box in list_class_box:
                    # Create a Rectangle patch to show the bounding box
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )

                    ax3.add_patch(rect)

                fig.suptitle(
                    f"{os.path.basename(filename)}    Jaccard:{ternaus_jaccard:.4f}"
                )

                plt.savefig(
                    f"../detectron/output/{self.dir_to_safe}/images_predicted_multilabeled/layerwise/"
                    f"{class_name}/{os.path.basename(filename)}",
                    dpi=300,
                )
                print(
                    f"Saved ../detectron/output/{self.dir_to_safe}/images_predicted_multilabeled/layerwise/"
                    f"{class_name}/{os.path.basename(filename)}"
                )
                plt.close("all")
        else:
            print("###########Not found#################")

    def get_right_mask(self, filename: str, class_name: int):
        """
        Open the corresponding ground truth mask
        Args:
            filename:
            class_name:

        Returns:

        """
        try:
            opened_image = np.array(
                Image.open(
                    f"../data/output/{self.dir_to_safe}/masks_val/"
                    f"{os.path.basename(filename)[:-4]}_layer_{str(class_name).zfill(2)}.png"
                ),
                dtype=np.uint8,
            )
            return opened_image
        except:
            return None

    def get_metrics_for_detection(
        self,
        out: detectron2.utils.visualizer.Visualizer,
        im: np.ndarray,
        outputs: dict,
        filename: str,
        visualise: bool = True,
    ) -> None:
        """
        Calculating metrics based on  predictions
        """
        output = im.copy()
        visualise = True
        predicted_label = list(
            outputs["instances"]._fields["pred_classes"].data.numpy()
        )
        predicted_masks = list(outputs["instances"]._fields["pred_masks"].data.numpy())
        predicted_boxes = list(
            outputs["instances"]._fields["pred_boxes"].tensor.data.numpy()
        )
        print(f"{predicted_label}")

        list_zero_class_mask, list_first_class_mask, list_second_class_mask = [], [], []
        list_zero_class_box, list_first_class_box, list_second_class_box = [], [], []
        list_third_class_mask, list_third_class_box = [], []
        list_forth_class_mask, list_forth_class_box = [], []
        list_fifth_class_mask, list_fifth_class_box = [], []
        list_sixth_class_mask, list_sixth_class_box = [], []
        list_seventh_class_mask, list_seventh_class_box = [], []
        list_eigth_class_mask, list_eigth_class_box = [], []
        list_nineth_class_mask, list_nineth_class_box = [], []
        list_tenth_class_mask, list_tenth_class_box = [], []

        for index, class_value in enumerate(predicted_label):
            if class_value == 0:
                list_zero_class_mask.append(predicted_masks[index])
                list_zero_class_box.append(predicted_boxes[index])
            if class_value == 1:
                list_first_class_mask.append(predicted_masks[index])
                list_first_class_box.append(predicted_boxes[index])
            if class_value == 2:
                list_second_class_mask.append(predicted_masks[index])
                list_second_class_box.append(predicted_boxes[index])
            if class_value == 3:
                list_third_class_mask.append(predicted_masks[index])
                list_third_class_box.append(predicted_boxes[index])
            if class_value == 4:
                list_forth_class_mask.append(predicted_masks[index])
                list_forth_class_box.append(predicted_boxes[index])
            if class_value == 5:
                list_fifth_class_mask.append(predicted_masks[index])
                list_fifth_class_box.append(predicted_boxes[index])
            if class_value == 6:
                list_sixth_class_mask.append(predicted_masks[index])
                list_sixth_class_box.append(predicted_boxes[index])
            if class_value == 7:
                list_seventh_class_mask.append(predicted_masks[index])
                list_seventh_class_box.append(predicted_boxes[index])
            if class_value == 8:
                list_eigth_class_mask.append(predicted_masks[index])
                list_eigth_class_box.append(predicted_boxes[index])
            if class_value == 9:
                list_nineth_class_mask.append(predicted_masks[index])
                list_nineth_class_box.append(predicted_boxes[index])
            if class_value == 10:
                list_tenth_class_mask.append(predicted_masks[index])
                list_tenth_class_box.append(predicted_boxes[index])

        predicted_label_masks = [
            list_zero_class_mask,
            list_first_class_mask,
            list_second_class_mask,
            list_third_class_mask,
            list_forth_class_mask,
            list_fifth_class_mask,
            list_sixth_class_mask,
            list_seventh_class_mask,
            list_eigth_class_mask,
            list_nineth_class_mask,
            list_tenth_class_mask,
        ]

        predicted_label_box = [
            list_zero_class_box,
            list_first_class_box,
            list_second_class_box,
            list_third_class_box,
            list_forth_class_box,
            list_fifth_class_box,
            list_sixth_class_box,
            list_seventh_class_box,
            list_eigth_class_box,
            list_nineth_class_box,
            list_tenth_class_box,
        ]

        for iter_index_class in range(0, len(self.class_list)):
            # going through all classes that we have trainied in a certain phase
            try:
                mask = self.get_right_mask(filename, class_name=iter_index_class)
                if mask is not None:
                    # evaluate if mask that was opened was ok
                    if np.any(np.nonzero(mask)):
                        # if (np.any(mask != [0, 0, 0], axis=-1)):
                        # if mot all pixels are black -> there were some annotations in it

                        if iter_index_class in predicted_label:
                            list_class_mask = predicted_label_masks[iter_index_class]
                            list_class_box = predicted_label_box[iter_index_class]
                            self.plot_inly_predicted_mask(
                                list_class_mask,
                                filename,
                                class_name=iter_index_class,
                                saving=visualise,
                            )

                            # turn of for performance optimisation
                            # self.plotting_each_layer(list_class_mask, list_class_box, filename, output,
                            #                          class_name=iter_index_class,
                            #                          saving=visualise, mask=mask)
                        else:
                            original_label_name = self.get_original_label(
                                iter_index_class
                            )
                            self.jaccard_default_dict[
                                os.path.basename(filename)
                            ].append({original_label_name: 0.0000})
                    else:
                        original_label_name = self.get_original_label(iter_index_class)
                        self.jaccard_default_dict[os.path.basename(filename)].append(
                            {original_label_name: None}
                        )
                else:
                    original_label_name = self.get_original_label(iter_index_class)
                    self.jaccard_default_dict[os.path.basename(filename)].append(
                        {original_label_name: None}
                    )
            except:
                original_label_name = self.get_original_label(iter_index_class)
                self.jaccard_default_dict[os.path.basename(filename)].append(
                    {original_label_name: 0.0000}
                )
                print(f"Issue in  for {filename}")

        if visualise:
            fig = plt.figure(frameon=False)
            # fig.set_size_inches(1024, 1280)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.savefig(
                f"../detectron/output/{self.dir_to_safe}/images_predicted_multilabeled/full_prediction"
                f"/predicted_{os.path.basename(filename)}.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            plt.show(block=False)
            plt.close("all")

    def inference_on_trained_mode(
        self,
        instruments_metadata: Metadata,
        path_to_data: str,
        json_description: str,
        model_location="model_final_renderpoint.pth",
        visualise=False,
    ) -> None:

        """
        Perfomring a prediction based on a tranied model.
        Here will be evaluated on a Jaccard index metric.
        The produced predictions can be either visualised and saved or not based on the visualise parameter

        Args:
            instruments_metadata: metadata information for datectron network
            path_to_data: path to the evaluation data
            json_description: description json file, which contains the ground truth information about
                the validation images
            model_location: location where the weights are stored which wil be loaded by the model
            visualise: boolean expression whether to visualise the information or not

        Returns:
        - Nothing to return
        """
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_location)
        print(f"taken {self.cfg.MODEL.WEIGHTS}")
        # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.DATASETS.TEST = ("resection_val",)
        predictor = DefaultPredictor(self.cfg)
        training_obj = training.StartTraining(self.config_name)
        dataset_dicts = training_obj.get_balloon_dicts(
            f"{path_to_data}val", json_with_description_name=json_description
        )

        for d in tqdm(dataset_dicts):
            base_name = os.path.basename(d["file_name"])
            im = cv2.imread(d["file_name"])
            print(f'Filename = {d["file_name"]}')

            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=instruments_metadata,
                scale=1,
                instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
            )
            outputs["instances"]._fields["pred_masks"].data.numpy()

            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            self.get_metrics_for_detection(
                out, im, outputs, filename=d["file_name"], visualise=visualise
            )

    def dump_metrics(self, location_to_save: str) -> None:
        """
        Dump the collected metrics to json
        Args:
            location_to_save: path where to save the json data with the results of the predictions
        Returns:
            - nothing to return, the collected metrics will be dumped to a json file
        """
        with open(location_to_save, "w") as f:
            json.dump(self.jaccard_default_dict, f)
            print(
                f"Successfully dumped self.jaccard_default_dict to {location_to_save}"
            )

        # TODO: bug with different ammount of classes
        full_dict = pd.DataFrame.from_dict(
            self.jaccard_default_dict, orient="index", columns=self.class_list
        )
        correct_dataframe = remove_dict_struct_from_pd(full_dict, self.class_list)

        correct_dataframe.to_csv(f"{location_to_save[:-5]}.csv")
