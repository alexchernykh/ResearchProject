import glob
import os
import nrrd
from albumentations import VerticalFlip
import numpy as np
from skimage.measure import label, regionprops
import json
import PIL.Image as Image
from skimage import measure
from skimage.measure import label, regionprops
from typing import List, Tuple, Dict
import yaml
from enum import Enum
from collections import Counter
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
import albumentations
from tqdm import tqdm
import math
from configs.config_generator import create_config_from_params
from argparse import ArgumentParser
from detectron.phases import OpPhase

padding = 2
# parameter for the images -> how many places should be filled with 0 (f.e
# index = 1 -> 00001)
index_nulls = 5
# parameter for the layers -> how many places should be filled with 0 (f.e
# layer = 1 -> 01)
layer_nulls = 2

# where the black part of the image ends to crop it
crop_lower_height, crop_upper_height = 315, 1595
crop_lower_width, crop_upper_width = 28, 1052
# crop_lower_width, crop_upper_width = 0, 1024
w, h = 1920, 1080  # image size that should be used

mapping = {0: 0, 1: 255}


def get_yaml_config(name: str = "../configs/configs.yaml") -> dict:
    with open(name, "r") as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_config


class ModeRun(Enum):
    """
    This is a choice to run in different modes based on the mask layers we need to use for the annotation
    """

    ALL_LABELS = 1
    PLANE_ONLY = 2


class NrrdToTrainingTestSplitter:
    """
    This is a class for splitting the data into training testing set.
    It gets an location
    """

    @staticmethod
    def get_mask_images(folder_data: List[str]) -> Tuple[List[str], List[str]]:
        """
        Getting all nrrd images in a folder
        Args:
            folder_data: where to look for nrrd data

        Returns:
            returning the full image and mask path
            images ['../data/nrrds_to_process/20200504_first incision.nrrd']
            masks ['../data/nrrds_to_process/20200504_first incision.seg.nrrd']
        """
        folder_data = sorted(folder_data)

        images, mask_local = [], []

        for file in folder_data:
            # print(f'No match found{os.path.basename(file)}')
            if file.endswith("seg.nrrd"):
                mask_local.append(file)
            elif file.endswith(".nrrd"):
                images.append(file)
            elif not (file.endswith("seg.nrrd")) and not (file.endswith(".nrrd")):
                print(f"No match found{os.path.basename(file)}")

        print(f"{len(images)}")
        print(f"{len(mask_local)}")
        return images, mask_local

    @staticmethod
    def split_data(
        folder_image_data: List[str],
        folder_mask_data: List[str],
        train_size: float = 0.7,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:

        folder_image_data = sorted(folder_image_data)
        folder_mask_data = sorted(folder_mask_data)

        len_data = len(folder_image_data)
        print(len_data)

        train_image_paths = folder_image_data[: int(len_data * train_size)]
        test_image_paths = folder_image_data[int(len_data * train_size):]

        train_mask_paths = folder_mask_data[: int(len_data * train_size)]
        test_mask_paths = folder_mask_data[int(len_data * train_size):]

        return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths

    def image_mask_with_full_paths(self, folder_data, base_name=None):
        if base_name is None:
            base_name = "/Users/chernykh_alexander/Github/chirurgie_research/data/nrrds_to_process/"

        images, masks = self.get_mask_images(folder_data)

        images = [base_name + imag for imag in images]
        masks = [base_name + single_mask for single_mask in masks]
        return images, masks


def get_basename_for_image(folder_data):
    """
    Getting the base name of the nrrd
    Args:
        folder_data: location of the nrrd
            '../data/nrrds_to_process/20200504_first incision.nrrd'
    Returns:
        the name of the nrrd without the path to it
        '20200504_first incision.seg.nrrd'
    """
    data = os.path.basename(folder_data)
    if data.endswith(".nrrd"):
        data = data[:-5]
        if data.endswith(".seg"):
            return data[:-4]
        return data


class AugmentationImg:
    @classmethod
    def strong_aug(self, p=1):
        """
        Function to perform  an mirroring on the x axis based on the albumentation func
        Args:
            p:the probability that the image will be flipped

        Returns:
            Vertical flipped image
        """
        return VerticalFlip(p=p)

    def rotate_and_flip(self, img: np.ndarray) -> np.ndarray:
        """
        Performs firstly a flip along the x axis and than a 90 degree rotation
        Args:
            img: image that should be processed

        Returns:
            a flipped and rotated image
        """
        augmentation = self.strong_aug(p=1)
        data = {"image": img}
        augmented = augmentation(**data)
        return np.rot90(augmented["image"], axes=(0, 1), k=-1)


def check_op_phase(yaml_config: dict, layer_name: str) -> str:
    """
    Setting the mapping layer according to the Phase name
    Args:
        yaml_config: config with the set up parameters
        layer_name: the layername from a certain mapping
    Returns:
    """
    if yaml_config["OP_PHASE"] == OpPhase.FIRST_INCISION.name:
        print(OpPhase.FIRST_INCISION.name)
        layer_output = mapping_layers[layer_name]
    elif yaml_config["OP_PHASE"] == OpPhase.PEDICLE_PACKAGE.name:
        print(OpPhase.PEDICLE_PACKAGE.name)
        layer_output = mapping_layers_pedickle_package[layer_name]
    elif yaml_config["OP_PHASE"] == OpPhase.VASCULAR_DISSECTION.name:
        print(OpPhase.VASCULAR_DISSECTION.name)
        layer_output = mapping_layers_vascular_dissection[layer_name]
    elif yaml_config["OP_PHASE"] == OpPhase.MESOCOLON_GEROTA.name:
        print(OpPhase.MESOCOLON_GEROTA.name)
        layer_output = mapping_layer_mesocolon_gerota[layer_name]
    elif yaml_config["OP_PHASE"] == OpPhase.TME.name:
        print(OpPhase.TME.name)
        layer_output = mapping_layers_tme[layer_name]
    return layer_output


def check_nrrd(yaml_config: dict, op_name: str) -> dict:
    """
    Getting thee whole layers encoding
    Args:
        yaml_config: config with the set up parameters
        op_name: the name of the Opphase

    Returns:

    """
    if yaml_config["OP_PHASE"] == OpPhase.FIRST_INCISION.name:
        print(OpPhase.FIRST_INCISION.name)
        layers_dict = nrrd_mapping[op_name]
    elif yaml_config["OP_PHASE"] == OpPhase.PEDICLE_PACKAGE.name:
        print(OpPhase.PEDICLE_PACKAGE.name)
        layers_dict = pedickle_package_mapping[op_name]
    elif yaml_config["OP_PHASE"] == OpPhase.VASCULAR_DISSECTION.name:
        print(OpPhase.VASCULAR_DISSECTION.name)
        layers_dict = mapping_vascular_dissection[op_name]
    elif yaml_config["OP_PHASE"] == OpPhase.MESOCOLON_GEROTA.name:
        print(OpPhase.MESOCOLON_GEROTA.name)
        layers_dict = mapping_mesocolon_gerota[op_name]
    elif yaml_config["OP_PHASE"] == OpPhase.TME.name:
        print(OpPhase.TME.name)
        layers_dict = mapping_tme[op_name]
    return layers_dict


def save_img(images: str, yaml_config: dict, train: bool = True) -> None:
    """
    Saving the image from an nrrd to a .png file in a location specified in yaml
    Image are saved with a rotation and mirroring because of the transformation
    by the slicer while saving after annotation.
    Args:
        images: '../data/nrrds_to_process/20200504_first incision.nrrd'
        yaml_config: config files content
        train:

    Returns:
        Nothing to return the images will be only saved
    """
    train = yaml_config["CREATE_DATASET"]["TRAIN"]
    base_name = get_basename_for_image(images)  # getting the name to save
    base_name = f"{base_name[:8]}_{yaml_config['OP_PHASE']}"
    # load the nrrd to further save img
    nrrd_image_data, header = nrrd.read(images)

    for index in tqdm(range(nrrd_image_data.shape[3])):
        image_reshaped = np.moveaxis(
            nrrd_image_data[
                :3,
                crop_lower_height:crop_upper_height,
                crop_lower_width:crop_upper_width,
                index,
            ],
            0,
            -1,
        )

        img_rotated = AugmentationImg.rotate_and_flip(image_reshaped)
        img = Image.fromarray(img_rotated, "RGB")
        if train:
            img.save(
                f'{yaml_config["CREATE_DATASET"]["SAVE_IMAGES_LOCATION_TRAIN"]}{base_name}_{str(index).zfill(index_nulls)}.png'
            )
            if yaml_config["CREATE_DATASET"]["VERBOSE"]:
                print(
                    f"Saved image {base_name}_{str(index).zfill(index_nulls)}.png to {yaml_config['CREATE_DATASET']['SAVE_IMAGES_LOCATION_TRAIN']}"
                )
        else:
            img.save(
                f'{yaml_config["CREATE_DATASET"]["SAVE_IMAGES_LOCATION_VAL"]}{base_name}_{str(index).zfill(index_nulls)}.png'
            )
            if yaml_config["CREATE_DATASET"]["VERBOSE"]:
                print(
                    f"Saved image {base_name}_{str(index).zfill(index_nulls)}.png to {yaml_config['CREATE_DATASET']['SAVE_IMAGES_LOCATION_VAL']}"
                )


def mask_to_class(imgage_mask_np: np.ndarray) -> np.ndarray:
    """
    Performs a mapping of a value in a numpy ndarray to a certain value according
    to the mapping element
    Args:
        imgage_mask_np: is an array that has to bee mapped

    Returns:
        a numpy array with mapped values
    """
    for k in mapping:
        imgage_mask_np[imgage_mask_np == k] = mapping[k]
    return imgage_mask_np


def save_mask_multilabel(
    nrrd_masks: str, yaml_config: dict, train: bool = True, verbose: bool = True
) -> None:
    """
    Performs saving of a mask for each image
    The masks are representing a binary mask for  a certain class
    Based on the offset values will be adapted the mask overlay
    Rotatting an flippping the mask, like it was done for the image

    Args:
        nrrd_masks: '../data/nrrds_to_process/20200504_first incision.seg.nrrd'
        yaml_config: yaml config content
        train: which mode is now used to choose the saving location
        verbose:

    Returns:
        nothing to return -> image saved in a location
    """
    base_name = get_basename_for_image(nrrd_masks)  # getting the name to save
    base_name = f"{base_name[:8]}_{yaml_config['OP_PHASE']}"
    nrrd_masks_data, header_mask = nrrd.read(nrrd_masks)

    # get the layers reprresentation for the nrrd
    layers_dict = check_nrrd(yaml_config=yaml_config, op_name=base_name[:8])
    # layers_dict = nrrd_mapping[base_name[:8]]
    np_ndarray_to_fill = np.zeros(shape=(w, h))
    for layer in range(nrrd_masks_data.shape[0]):

        # get the layers real name from layers_dict
        layer_name = layers_dict[str(layer)]
        if (
            not layer_name
        ):  # if a layer is False = not represented in mapping skipp it -> it is empty
            break
        for index in tqdm(range(nrrd_masks_data.shape[3])):
            # iterating over the samples of certain layer
            imgage_mask_np = nrrd_masks_data[layer, :, :, index]
            imgage_mask_np_converted = mask_to_class(imgage_mask_np)
            imgage_mask_np_converted = mask_to_class(
                imgage_mask_np
            )  # convert to bina ry

            # get the image offsets from corners from the nrrd header
            x_offset, y_offset, z_offset = header_mask["space origin"]
            x_offset_abs, y_offset_abs, z_offset_abs = (
                int(abs(x_offset)),
                int(abs(y_offset)),
                int(abs(z_offset)),
            )

            # ValueError: could not broadcast input array from shape (1794,1065) into shape (1794,1035)s

            # filling the empty numpy array with values from the nrrd mask
            np_ndarray_to_fill[
                x_offset_abs:imgage_mask_np_converted.shape[0] + x_offset_abs,
                y_offset_abs:imgage_mask_np_converted.shape[1] + y_offset_abs,
            ] = imgage_mask_np_converted
            # flipping and mirroring the mask
            img_rotated = AugmentationImg.rotate_and_flip(
                np_ndarray_to_fill[
                    crop_lower_height:crop_upper_height,
                    crop_lower_width:crop_upper_width,
                ].astype("uint8")
            )
            mask_layer = Image.fromarray(img_rotated, mode="L")
            layer_output = check_op_phase(yaml_config, layer_name)
            if train:
                image_name_to_save = f'{yaml_config["CREATE_DATASET"]["SAVE_MASKS_LOCATION"]}{base_name}_{str(index).zfill(index_nulls)}_layer_{str(layer_output).zfill(layer_nulls)}.png'
                mask_layer.save(image_name_to_save)
                if yaml_config["CREATE_DATASET"]["VERBOSE"]:
                    print(
                        f"{image_name_to_save} to {yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION']}"
                    )
            else:
                image_name_to_save = f'{yaml_config["CREATE_DATASET"]["SAVE_MASKS_LOCATION_VAL"]}{base_name}_{str(index).zfill(index_nulls)}_layer_{str(layer_output).zfill(layer_nulls)}.png'
                mask_layer.save(image_name_to_save)
                if yaml_config["CREATE_DATASET"]["VERBOSE"]:
                    print(
                        f"{image_name_to_save} to {yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION_VAL']}"
                    )


def create_img_to_mask_layers_index(
    counter_of_images_in_single_nrrd: Dict[str, int], yaml_config
) -> Dict[str, List[str]]:
    """
        Create an association of the img and the according masknames for it
        Args:
            counter_of_images_in_single_nrrd:
            yaml_config: config file

        Returns:
    {'../data/output/images_new/train/20190917_first incision_0.png':
                ['../data/output/images_new/masks/20190917_first incision_0_layer_0.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_1.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_2.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_3.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_4.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_5.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_6.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_7.png',
                '../data/output/images_new/masks/20190917_first incision_0_layer_8.png']....}

    """
    image_to_masks_dict = {}
    # find_unique 20190917_first incision
    keys = list(counter_of_images_in_single_nrrd.keys())
    for key in keys:
        for index in range(counter_of_images_in_single_nrrd[key]):
            index = str(index).zfill(index_nulls)
            if yaml_config["CREATE_DATASET"]["TRAIN"]:

                image_name_for_dict = f"{yaml_config['CREATE_DATASET']['SAVE_IMAGES_LOCATION_TRAIN']}{key}_{yaml_config['OP_PHASE']}_{index}.png"
                mask_layer_list = sorted(
                    glob.glob(
                        f"{yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION']}{key}_{yaml_config['OP_PHASE']}_{index}_*"
                    )
                )
            else:
                # image_name_for_dict = f"{yaml_config['CREATE_DATASET']['SAVE_IMAGES_LOCATION_VAL']}{key}_{yaml_config['OP_PHASE']}_{index}.png"
                # mask_layer_list = sorted(glob.glob(
                #     f"{yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION_VAL']}{key}_{yaml_config['OP_PHASE']}_{index}_*"))
                image_name_for_dict = f"{yaml_config['CREATE_DATASET']['SAVE_IMAGES_LOCATION_VAL']}{key}.png"
                mask_layer_list = sorted(
                    glob.glob(
                        f"{yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION_VAL']}{key}*"
                    )
                )

            length_mask: int = len(mask_layer_list)
            # if length_mask == 9:
            print(f" {length_mask} Layers for {key}_first incision_{index}")
            image_to_masks_dict[image_name_for_dict] = mask_layer_list

    return image_to_masks_dict


def get_keys(image_to_masks_dict: dict) -> List[str]:
    # get keys from dict to access further the
    return list(image_to_masks_dict.keys())


def create_layer_annotation(masks_list: List[str], regions: dict, padding: int) -> dict:
    """
    Creting a region annotation based on a list of masks
    Args:
        masks_list:
            f.e ['../data/output/images_new/masks/20190917_first incision_0_layer_0.png',
            '../data/output/images_new/masks/20190917_first incision_0_layer_1.png']
        regions: an empty dict that has to be filled by COCO annotations for polygons
        padding: padding from the corner of an images

    Returns:
        a filled with all attributes dict

    """
    region_index = 0
    for single_mask in masks_list:

        mask = np.array(Image.open(single_mask))
        lbl_0 = label(mask)
        props = regionprops(lbl_0)
        # fig, ax = plt.subplots()
        mask_like = np.zeros_like(mask)
        props_amount = len(props)
        if props_amount > 1:
            print(f"More props than 1 :  {single_mask}")

        # get the layer name
        layer = os.path.splitext(os.path.basename(single_mask))[0][-2:]

        for index_prop, prop in enumerate(props):

            rectangle1 = mask[
                prop.bbox[0] - padding:prop.bbox[2] + padding,
                prop.bbox[1] - padding:prop.bbox[3] + padding,
            ]
            # trying to find contours if not possible proceed to next layer
            try:
                contours = measure.find_contours(rectangle1, 0.8)
            except BaseException:
                print("could not calculate contour")
                break

            def addition_y(y):
                return y + prop.bbox[1] - padding

            def addition_x(x):
                return x + prop.bbox[0] - padding

            max_length, index_contour = max(
                [(i.shape[0], index) for index, i in enumerate(contours)]
            )
            # for n, contour in enumerate(contours):
            y = addition_y(contours[index_contour][:, 1].astype(int))
            x = addition_x(contours[index_contour][:, 0].astype(int))

            shape_attr = dict()
            shape_attr["name"] = "polygon"
            shape_attr["all_points_x"] = x.tolist()
            shape_attr["all_points_y"] = y.tolist()
            shape_attribute = dict()
            shape_attr["category_id"] = int(layer)
            shape_attribute["shape_attributes"] = shape_attr
            shape_attribute["region_attributes"] = {}
            mask_like[x, y] = 255
            regions[str(region_index)] = shape_attribute
            region_index += 1
    return regions


def train_validation_split(im_desk, split_size=0.8):
    train_size = int(len(im_desk) * split_size)
    sampled_train = random.sample(im_desk, train_size)

    print(sampled_train)
    print(f"Size: {len(sampled_train)}")

    sampled_validation = [sample for sample in im_desk if sample not in sampled_train]
    print("############################")
    print("############################")
    should = len(im_desk) - len(sampled_train)
    if len(sampled_validation) is not should:
        return 0, 0
    return sampled_train, sampled_validation


def find_polygons(
    for_json: dict,
    img_name: str,
    image_to_masks_dict: Dict[str, List[str]],
    padding: int,
    mode: ModeRun = ModeRun.ALL_LABELS,
) -> dict:
    """
    Process a single image by creating for it an annotation  in a COCO format.
    This annotation is than appended into the for_json, which will be further dumped to JSON
    and used by the Detectron Dataloader as a description for an image.

    Args:
        for_json: accumulate image description to further dump it
        img_name:
        image_to_masks_dict:
        padding:
        mode:

    Returns:
        for_json with an appended infromation about img_name
    """
    masks_list = image_to_masks_dict[img_name]
    if mode == ModeRun.PLANE_ONLY:
        # todo fix also to train only on the resection plane if resection line
        # is not annotated
        indexes_with_plane = [
            index
            for index, single_mask_name in enumerate(masks_list)
            if single_mask_name.endswith("01.png")
            or single_mask_name.endswith("00.png")
        ]
        if len(indexes_with_plane) < 2:
            return for_json
    img = np.array(Image.open(img_name))

    filename = os.path.basename(img_name)
    single_image = {"filename": filename}
    for_json[str(filename)] = single_image
    single_image["fileref"] = ""
    single_image["size"] = img.size
    print(filename)
    height, width, channels = img.shape
    single_image["height"] = height
    single_image["width"] = width
    single_image["base64_img_data"] = ""
    single_image["file_attributes"] = {}
    regions = {}
    # single_image['regions'] = regions

    if mode == ModeRun.ALL_LABELS:

        regions = create_layer_annotation(masks_list, regions, padding)

    elif mode == ModeRun.PLANE_ONLY:
        regions = create_layer_annotation(masks_list[:2], regions, padding)

    else:
        print("FALSE MODE CHOSEN")
    single_image["regions"] = regions
    return for_json


def create_dict_train(sampled_train: List[str], masks_dir: List[str]):
    """

    Args:
        sampled_train: ['../data/output/final_training/val/20190917_first incision_00001.png', ....]
        masks_dir:

    Returns:
        {'../data/output/final_training/val/20190917_first incision_00001.png': [....],
    """
    image_to_masks_dict = {}

    for train_img in sampled_train:
        img_name = os.path.basename(train_img)[:-4]
        corresponding_mask = [mask for mask in masks_dir if img_name in mask]
        image_to_masks_dict[str(train_img)] = corresponding_mask
    return image_to_masks_dict


def create_dataset_description(
    img_dir: List[str],
    padding: int,
    yaml_config: dict,
    mode: ModeRun = ModeRun.ALL_LABELS,
) -> None:
    """
    Here will be stored created the dataset_description file for the detectron
    Args:
        img_dir:
        mask_dir:

    Returns:
        Nothing to return
        Json with the image description will be dumped
    """
    for_json = dict()
    counter_of_images_in_single_nrrd = dict(
        Counter([os.path.basename(img)[:8] for img in img_dir])
    )

    # counter_of_images_in_single_nrrd
    # Counter
    image_to_masks_dict = create_img_to_mask_layers_index(
        counter_of_images_in_single_nrrd, yaml_config=yaml_config
    )
    mask_dir = glob.glob(f"{yaml_config['CREATE_DATASET']['SAVE_MASKS_LOCATION']}*.png")
    # image_to_masks_dict = create_dict_train(img_dir, mask_dir)
    list_of_keys = get_keys(image_to_masks_dict)
    for index, image in tqdm(enumerate(list_of_keys)):
        try:
            for_json = find_polygons(
                for_json, image, image_to_masks_dict, padding=padding, mode=mode
            )
        except:
            print("create_dataset_description issue")
            continue

    if yaml_config["CREATE_DATASET"]["TRAIN"]:
        with open(
            yaml_config["NEW"]["PATH_TO_TRAINING_DATA"]
            + yaml_config["CREATE_DATASET"]["OUTPUT_FILE"],
            "w",
        ) as f:
            json.dump(for_json, f)
        print(f"Saved to {yaml_config['CREATE_DATASET']['OUTPUT_FILE']}")

    else:
        with open(
            yaml_config["NEW"]["PATH_TO_VALIDATION_DATA"]
            + yaml_config["CREATE_DATASET"]["OUTPUT_FILE"],
            "w",
        ) as f:
            json.dump(for_json, f)
        print(f"Saved to {yaml_config['CREATE_DATASET']['OUTPUT_FILE_VAL']}")


def process_images_and_masks(
    nrrd_images: List[str], nrrd_masks: List[str], yaml_config: dict, train=True
) -> None:
    """
    Starting the saving image and extracting the according masks for each image
    Args:
        nrrd_images: names of the nrrds (in glob.glob format)
        nrrd_masks:names of the nrrds (in glob.glob format)
        yaml_config:
        train:

    Returns:

    """
    for index, nrrd_file in enumerate(nrrd_images):
        save_img(nrrd_images[index], yaml_config, train)
        save_mask_multilabel(nrrd_masks[index], yaml_config, train=train)
        pass


def create_dataset(config: dict, images: str, mask: str) -> None:

    mode = ModeRun.ALL_LABELS

    train = config["CREATE_DATASET"]["TRAIN"]

    if config["CREATE_DATASET"]["SAVE_IMG_AND_MASKS"]:
        # to print everything out
        process_images_and_masks(
            sorted(images), sorted(mask), train=train, yaml_config=config
        )

    if config["CREATE_DATASET"]["CREATE_DESCRIPTION_FILE"]:
        padding = config["CREATE_DATASET"]["PADDING"]

        if train:
            img_dir = sorted(glob.glob(config["CREATE_DATASET"]["IMG_DIR_TRAIN"]))
            # mask_dir = sorted(glob.glob("../data/output/images/masks/*.png"))
        else:
            img_dir = sorted(glob.glob(config["CREATE_DATASET"]["IMG_DIR_VAL"]))
        # '../data/output/final_op_mode/train/20190917_first incision_00055.png'
        create_dataset_description(
            img_dir, mode=mode, padding=padding, yaml_config=config
        )


def create_op_to_nrrd_mapping(list_vascular_dissection: List[str]):
    nrrd_list = os.listdir("./data/nrrds_to_process/")
    op_to_nrrd_mapper = {}
    for op_name in list_vascular_dissection:
        single_op_nrrds = []
        for single_nrrd in nrrd_list:
            if single_nrrd.startswith(op_name):
                single_op_nrrds.append(single_nrrd)
        op_to_nrrd_mapper[op_name] = single_op_nrrds

    return op_to_nrrd_mapper


def start_from_cmd():
    """
    Starting point to create a dataset from nrrd's to png.
    Returns:

    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--op_name", "-op", type=str, default="20190917")
    arg_parser.add_argument("--path_location", "-p", type=str, default="crossval_full")
    arg_parser.add_argument(
        "--model_pth_name",
        "-m",
        type=str,
        default="model_final_FIRST_INCISION_20190917.pth",
    )
    arg_parser.add_argument("--op_phase", "-ph", type=str, default="FIRST_INCISION")
    args = arg_parser.parse_args()

    config = create_config_from_params(
        op_name=args.op_name,
        path_location=args.path_location,
        op_phase=args.op_phase,
        train=False,
        save_img_and_mask=False,
    )
    folder_data = glob.glob(config["CREATE_DATASET"]["FOLDER_DATA_NRRD"])
    splitter_dataset_object = NrrdToTrainingTestSplitter()
    images, mask = splitter_dataset_object.get_mask_images(folder_data)
    create_dataset(config=config, images=images, mask=mask)
    return config


if __name__ == "__main__":

    op_to_nrrd_mapper = create_op_to_nrrd_mapping(["20190917"])
    for key in tqdm(op_to_nrrd_mapper.keys()):
        if op_to_nrrd_mapper[key]:
            print(f"Key = {key}")
            splitter_dataset_object = NrrdToTrainingTestSplitter()
            image, mask = splitter_dataset_object.image_mask_with_full_paths(
                op_to_nrrd_mapper[key]
            )
            config = create_config_from_params(
                op_name=key,
                path_location="crossval_full",
                op_phase="FIRST_INCISION",
                train=True,
                save_img_and_mask=False,
            )
            print(image, mask)
            config["CREATE_DATASET"]["CREATE_DESCRIPTION_FILE"] = False
            create_dataset(config=config, images=image, mask=mask)

    # start_from_cmd()
