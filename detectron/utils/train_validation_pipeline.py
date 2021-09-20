from shutil import copy
import glob
import itertools
import glob
from pprint import pprint
import os
from typing import List
from tqdm import tqdm


def create_cross_validation_combos(basename_dir: List[str]):
    training_val_dict = {}
    combinat_list = list(itertools.combinations(basename_dir, 9))

    for c in combinat_list:
        not_in = [base_dir for base_dir in basename_dir if base_dir not in c]
        current_cobo = {}
        current_cobo["val"] = not_in
        current_cobo["train"] = sorted(list(c))
        training_val_dict[str(not_in[0])] = current_cobo
    return training_val_dict


def copy_files(first_img, distanation):
    """
    copying files from one destination to another
    Args:
        first_img:
        distanation:

    Returns:

    """
    for img in tqdm(first_img):
        copy(img, distanation)
        print(f"copied {img}")


def copy_image_files_train(
    training_val_dict,
    op_name,
    base_name_to_save: str,
    dir_name_to_save: str = "/masks/",
    validation: bool = True,
):
    distanation = (
        base_name_to_save
        + str(os.path.basename(training_val_dict[op_name]["val"][0]))
        + dir_name_to_save
    )
    if validation:
        first_img = glob.glob(training_val_dict[op_name]["val"][0] + "/*.png")
        copy_files(first_img, distanation)
    else:
        anount_of_train_ops = len(training_val_dict[op_name]["train"])
        for training_op in range(anount_of_train_ops):
            first_img = glob.glob(
                training_val_dict[op_name]["train"][training_op] + "/*.png"
            )
            copy_files(first_img, distanation)


def copy_image_files_val(
    training_val_dict,
    op_name,
    base_name_to_save: str,
    dir_name_to_save: str = "/masks_val/",
):
    distanation = (
        base_name_to_save
        + str(os.path.basename(training_val_dict[op_name]["val"][0]))
        + dir_name_to_save
    )
    # anount_of_train_ops = len(training_val_dict[op_name]['val'])
    # for training_op in range(anount_of_train_ops):
    first_img = glob.glob(training_val_dict[op_name]["val"][0] + "/*.png")
    copy_files(first_img, distanation)


def create_validation_dirs():
    dir_list = glob.glob("/Users/chernykh_alexander/Desktop/maks/*")
    base_name_to_save = "/Users/chernykh_alexander/Desktop/to_save/"
    training_val_dict = create_cross_validation_combos(basename_dir=dir_list)

    op_names = list(training_val_dict.keys())
    for op_name in op_names:
        # copy_image_files_val(training_val_dict, op_name, base_name_to_save)
        copy_image_files_train(
            training_val_dict,
            op_name,
            base_name_to_save,
            dir_name_to_save="/masks_val/",
            validation=True,
        )
        copy_image_files_train(
            training_val_dict,
            op_name,
            base_name_to_save,
            dir_name_to_save="/masks/",
            validation=False,
        )


def create_training_dirs():
    dir_list = glob.glob("/Users/chernykh_alexander/Desktop/images/*")
    base_name_to_save = "/Users/chernykh_alexander/Desktop/to_save/"
    training_val_dict = create_cross_validation_combos(basename_dir=dir_list)

    op_names = list(training_val_dict.keys())
    for op_name in op_names:
        # copy_image_files_val(training_val_dict, op_name, base_name_to_save)
        copy_image_files_train(
            training_val_dict,
            op_name,
            base_name_to_save,
            dir_name_to_save="/val/",
            validation=True,
        )
        copy_image_files_train(
            training_val_dict,
            op_name,
            base_name_to_save,
            dir_name_to_save="/train/",
            validation=False,
        )
