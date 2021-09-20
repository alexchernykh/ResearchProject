from detectron.plane_postprocessing.resec_line_length_estimation import AverageResectionLineDistance
from detectron.plane_postprocessing.triangle_points import calculate_left_corner_angle
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
import json


def dump_json(to_dump_object: dict, location_to_save: str) -> None:
    with open(location_to_save, 'w') as f:
        json.dump(to_dump_object, f)
        print(f"Successfully dumped to {location_to_save}")


def collect_resec_line_metrics(file_description: dict,
                               resec_line_obj: AverageResectionLineDistance,
                               filename: str):
    """
    Collecting the metrics for the filename that describe the resection line:
        - starting_point
        - end_point
        - length
        - slope_angle
    Args:
        file_description: where the
        resec_line_obj:
        filename:

    Returns:

    """

    mask_draft = np.array(Image.open(filename))
    starting_point, end_point, max_length = resec_line_obj.get_edge_point(mask_draft)
    third_coordinate = (starting_point[0], end_point[1])
    """
            This function calculates the angle C of the bottom left point, to further use it
            for the angle of the resection line.
        #
        #           A
        #          |\
        # (b_line)|  \ (c_line)
        #        |    \
        #        C---- B
        #       (a_line)
        Args:
            x_row_lower_right:
            y_column_lower_right:
            x_row_left:
            y_column_left:
            x_row_top:
            y_column_top:
    """
    gamma_angle_degrees = math.degrees(calculate_left_corner_angle(third_coordinate[0], third_coordinate[1],
                                                                   starting_point[0], starting_point[1], end_point[0],
                                                                   end_point[1]))

    # file_description[os.path.basename(filename)[:29]] = resec_line_descriptor_dict
    file_description["starting_point"] = starting_point
    file_description["end_point"] = end_point
    file_description["length"] = int(round(max_length))
    file_description["slope_angle"] = round(gamma_angle_degrees, 1)
    return file_description


def main():
    resec_line_obj = AverageResectionLineDistance(
        img_dir=sorted(
            glob.glob("/Users/chernykh_alexander/Github/chirurgie_research/data/output/crossval/*/masks_val/*00.png")),
        location_to_save='./../../csv/detectron/plane_evaluation/gt_evaluation/resec_line_gt_description_UPDATED_final.json')

    json_to_dump = {}
    for filename in tqdm(resec_line_obj.img_dir):
        # filename = resec_line_obj.img_dir[0]

        resec_line_descriptor_dict: dict = {}
        json_to_dump[os.path.basename(filename)[:29] + '.png'] = resec_line_descriptor_dict
        try:
            resec_line_descriptor_dict = collect_resec_line_metrics(resec_line_descriptor_dict, resec_line_obj,
                                                                    filename)
        except:
            resec_line_descriptor_dict["starting_point"] = 0
            resec_line_descriptor_dict["end_point"] = 0
            resec_line_descriptor_dict["length"] = 0
            resec_line_descriptor_dict["slope_angle"] = 0

    dump_json(to_dump_object=json_to_dump,
              location_to_save=resec_line_obj.location_to_save)
    print('successful')


if __name__ == '__main__':
    main()
