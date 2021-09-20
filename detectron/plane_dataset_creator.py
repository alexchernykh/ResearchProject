from detectron.dataset_new_data import (
    process_images_and_masks,
    create_dataset_description,
)
import matplotlib.pyplot as plt
import glob
import collections
import os
import numpy as np
from PIL import Image
import cv2
import json
from tqdm import tqdm
from detectron.dataset_new_data import (
    create_img_to_mask_layers_index,
    get_yaml_config,
    get_keys,
    create_layer_annotation,
)


class ResectionOnPlane:
    """
    This class aims to produce only data based on plane.
    Therefore all other labels besides the ground truth plane and annotated resection line
    """

    def __init__(
        self, planes_img_dir="../data/output/final_op_mode/masks_val/foo/*.png"
    ):

        self.yaml_config = get_yaml_config("../configs/plane_config.yaml")
        self.planes_img_dir = sorted(glob.glob(planes_img_dir))
        self.counter_of_images_in_single_nrrd = dict(
            collections.Counter(
                [os.path.basename(img)[:8] for img in self.planes_img_dir]
            )
        )
        self.image_to_masks_dict = create_img_to_mask_layers_index(
            self.counter_of_images_in_single_nrrd, yaml_config=self.yaml_config
        )
        self.list_of_keys = get_keys(self.image_to_masks_dict)

    def create_plane_only_img(self, location_to_save):
        """
        Generating images for training detection of a resection line onl based on plane information
        Args:
            location_to_save:

        Returns:

        """

        for index, image_key in tqdm(enumerate(self.list_of_keys)):
            try:
                plane_element = [
                    mask
                    for mask in self.image_to_masks_dict[self.list_of_keys[index]]
                    if mask.endswith("01.png")
                ]
                name_to_save = os.path.basename(image_key)
                op_name = name_to_save[:23]
                frame_counter = name_to_save[23:-4]
                mask_np = np.array(Image.open(plane_element[0]))
                image_np = np.array(Image.open(self.list_of_keys[index]))

                dst0 = cv2.bitwise_and(mask_np, image_np[:, :, 0])
                dst1 = cv2.bitwise_and(mask_np, image_np[:, :, 1])
                dst2 = cv2.bitwise_and(mask_np, image_np[:, :, 2])

                plane_image = Image.fromarray(
                    (np.dstack((dst0, dst1, dst2))).astype(np.uint8)
                )
                plane_image.save(f"{location_to_save}{op_name}{frame_counter}.png")
                print(f"Saved image to {location_to_save}{op_name}{frame_counter}.png")
            except:
                print(f"could not process {index}, {image_key}")
                continue

    def generate_json_description(self):
        """
        Generate ground truth annotations to process further it by the network
        Returns:
            - Nothing, only json with a description of the annotations will be generated
        """
        for_json = {}
        padding = self.yaml_config["CREATE_DATASET"]["PADDING"]
        for index, image_key in enumerate(tqdm(self.list_of_keys)):

            if (
                os.path.basename(self.planes_img_dir[index])[:24]
                == os.path.basename(self.list_of_keys[index])[:24]
            ):
                plane_element = [
                    mask
                    for mask in self.image_to_masks_dict[self.list_of_keys[index]]
                    if mask.endswith("00.png")
                ]
                # print(plane_element)
                try:
                    cur_img_name = self.planes_img_dir[index][
                        :68
                    ]  # + os.path.basename(self.planes_img_dir[index])[30:]
                    img = np.array(Image.open(cur_img_name))

                    filename = os.path.basename(cur_img_name)
                    single_image = {
                        "filename": filename,
                        "fileref": "",
                        "size": img.size,
                    }
                    height, width, channels = img.shape
                    single_image["height"] = height
                    single_image["width"] = width
                    single_image["base64_img_data"] = ""
                    single_image["file_attributes"] = {}
                    regions = {}
                    single_image["regions"] = regions

                    regions = create_layer_annotation(
                        plane_element, regions, padding=padding
                    )
                    single_image["regions"] = regions
                    for_json[str(filename)] = single_image
                except:
                    print(f"No elements where detected for ")
            else:
                print("no plane")
        # for_json['20190917_first incision_00000.png']['regions']['0']['shape_attributes']['category_id']
        if self.yaml_config["CREATE_DATASET"]["TRAIN"]:
            with open(self.yaml_config["CREATE_DATASET"]["OUTPUT_FILE"], "w") as f:
                json.dump(for_json, f)
                print(f"Saved to {self.yaml_config['CREATE_DATASET']['OUTPUT_FILE']}")
        else:
            with open(self.yaml_config["CREATE_DATASET"]["OUTPUT_FILE_VAL"], "w") as f:
                json.dump(for_json, f)
                print(
                    f"Saved to {self.yaml_config['CREATE_DATASET']['OUTPUT_FILE_VAL']}"
                )


def main():
    resection_obj = ResectionOnPlane(
        planes_img_dir="../data/output/final_op_mode/train/*.png"
    )
    # resection_obj = ResectionOnPlane(planes_img_dir="../data/output/final_op_mode/val/*.png")
    # resection_obj.create_plane_only_img(location_to_save="../data/output/final_op_mode/plane_train/")
    resection_obj.generate_json_description()


if __name__ == "__main__":
    main()
