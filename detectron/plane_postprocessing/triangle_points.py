from scipy import ndimage
from math import acos, degrees, cos, sin
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from detectron.plane_postprocessing.plane_evaluation import get_centroid
import scipy.misc
from skimage.draw import line_aa
import os
from typing import List, Tuple, Dict
import cv2
import json

# mapping for a certain numpy array
mapping = {
    255: 100
}


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


def get_lower_left_point(img_np2: np.ndarray) -> Tuple[int, int]:
    """
    Calculating the maximal left point on the plane
    Args:
        img_np2: is the plane on which will be searched

    Returns:
        the indexes of the element found point
    """
    for i in range(img_np2.shape[1]):
        for j in range(img_np2.shape[0] - 1, 0, -1):
            if img_np2[j, i] > 0:
                return i, j


def get_lower_right(img_np2: np.ndarray) -> Tuple[int, int]:
    """
    Calculating the maximal right point on the plane
    Args:
        img_np2: is the plane on which will be searched

    Returns:
        the indexes of the element found point
    """
    for j in range(img_np2.shape[0] - 1, 0, -20):
        for i in range(img_np2.shape[1] - 1, 0, -1):
            if img_np2[j, i] > 0:
                return i, j


def get_top_trianlge(img_np2: np.ndarray) -> Tuple[int, int]:
    """
    Calculating the first point on top of the plane
    Args:
        img_np2: is the plane on which will be searched

    Returns:
        the indexes of the element found point
    """
    for j in range(img_np2.shape[0]):
        for i in range(img_np2.shape[1] - 1, 0, -1):
            if img_np2[j, i] > 0:
                return i, j


def create_triangle_points(
        img_np2: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Creating a trianlge on a plane mask
    Args:
        img_np2: is the plane mask on which will be created a traingle.

    Returns:
        returns the (x,y) coordinate of the three points locations on the ndarray
    """
    x_row_left, y_column_left = get_lower_left_point(img_np2)
    x_row_lower_right, y_column_lower_right = get_lower_right(img_np2)
    x_row_top, y_column_top = get_top_trianlge(img_np2)
    return x_row_lower_right, y_column_lower_right, x_row_left, y_column_left, x_row_top, y_column_top


def calculate_left_corner_angle(
        x_row_lower_right: int,
        y_column_lower_right: int,
        x_row_left: int,
        y_column_left: int,
        x_row_top: int,
        y_column_top: int) -> float:
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

    Returns:
        float value in radians for the Angle of C = gamma_angle
    """

    B = np.array([round(x_row_lower_right), round(y_column_lower_right)])
    C = np.array([round(x_row_left), round(y_column_left)])
    A = np.array([round(x_row_top), round(y_column_top)])
    a_line = round(np.linalg.norm(C - B))
    b_line = round(np.linalg.norm(A - C))
    # c_line = round(np.linalg.norm(A-B))
    c_line = round(np.linalg.norm(B - A))

    gamma_angle = acos((a_line * a_line + b_line * b_line -
                        c_line * c_line) / (2.0 * a_line * b_line))
    return gamma_angle


def compute_resection_line_endpoint(
        centroid: Tuple[float, float], gamma_angle: float, line_length: int = 218) -> Tuple[float, float]:
    """
    Computing the the end point of the resection line based on the angle and duration
    Args:
        line_length:
        centroid:
        gamma_angle:

    Returns:
        the  (x, y )coordinates of the endpoint

    """
    # TODO: aproxomate the line length from the trainings data
    xx = round(centroid[1]) + (line_length * (cos(gamma_angle / 2)))
    # need to change the direction of the sin because of the direction from
    # top to bottom
    yy = round(centroid[0]) + (line_length * (-sin(gamma_angle / 2)))
    return xx, yy


def scale(x: np.ndarray) -> np.ndarray:
    return x + 1


def expand_line_thickness(rr: np.ndarray) -> np.ndarray:
    """
        expanding the line to make it thicker
    Args:
        rr: array of coordinates

    Returns:
        thicker line
    """
    scaled_rr = scale(rr)
    rr_extended = np.append(rr, scaled_rr)
    return rr_extended


def compute_resection_line_mask(img_np2: np.ndarray,
                                centroid: Tuple[float, float],
                                xx: float,
                                yy: float) -> np.ndarray:
    """
        computing the mask of the resection line to further overlay the original image
    Args:
        img_np2: plane
        centroid: (x,y)  coordinates of the initial point
        xx: x coordinate of the end point
        yy: y coordinate of the end point

    Returns:
        mask with the resection line
    """
    where_to_draw_line = np.zeros_like(img_np2, dtype=np.uint8)
    # rr, cc, val = line_aa(int(round(centroid[0])), int(
    #     round(xx)), int(round(centroid[1])), int(round(yy)))

    # rr, cc, val = line_aa(int(round(centroid[0])),int(round(centroid[1])),
    #                       int(round(xx)), int(round(yy)))

    rr, cc, val = line_aa(int(round(xx)), int(round(yy)), int(round(centroid[0])), int(round(centroid[1])), )

    rr_extended = expand_line_thickness(expand_line_thickness(rr))
    cc_extended = expand_line_thickness(expand_line_thickness(cc))

    where_to_draw_line[rr_extended, cc_extended] = 255
    return where_to_draw_line


def convert_resec_to_alpha(
        filename: np.ndarray,
        alpha_data: np.ndarray) -> np.ndarray:
    """
        creating a mask for the image to overlay it over the original image
    Args:
        filename: image that should be opened
        alpha_data: alpha mask that shows which pixels should be visible

    Returns:
        a rgba representation of a resection line
    """

    rgba = cv2.cvtColor(filename, cv2.COLOR_RGB2RGBA)

    # Then assign the mask to the last channel of the image
    rgba[:, :, 3] = alpha_data
    return rgba


def convert_lumina_to_alpha(filename: str) -> np.ndarray:
    """
        Convert an plane of an image to an rgba image to further overlay over original image
    Args:
        filename: Plane name which should be converted to an rgba

    Returns:
        converted rgba image of a plane
    """
    alpha_data = np.array(Image.open(filename))

    rgba = cv2.cvtColor(
        np.array(
            Image.open(filename).convert('RGB')),
        cv2.COLOR_RGB2RGBA)

    # Then assign the mask to the last channel of the image
    rgba[:, :, 3] = alpha_data
    return rgba


#     Image.fromarray(rgba).save('./test_png.png')

def rgb_to_rgba(filename: str) -> np.ndarray:
    """
    Converting an RGB image to a RGBA
    This need to be done to overlay over this masks. Because for overlaying the
    image need to be (width, height 4) for every image (mask and original rgb)
    Args:
        filename: name of the file that should be converted to rgba

    Returns:
        converted rgba image of a the original plane
    """
    # alpha_data = np.array(Image.open(filename))
    rgba = cv2.cvtColor(
        np.array(
            Image.open(filename).convert('RGB')),
        cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = 255
    return rgba


def full_process_resection_line(mapping_dict: Dict[str, str],
                                dir_crossval: str = None,
                                location_to_save: str = None,
                                saving_fig: bool = True) -> None:
    """
    Processing the image pairs original img : plane predicted
    1. calculating the trianlge on the predicted plane
    2. calculate the angle of the lower left angle
    3. calculate centroid of the predicted plane
    4. compute resection line endpooints (begin ----- endpoint)
    5. draw a line inside the numpy array between the two points with an angle
    6. images  and masks to RGBA for further visualisation
    Args:
        saving_fig:
        location_to_save:
        dir_crossval:
        mapping_dict:

    Returns:
        nothing to return
        the result will be saved as an image
    """

    json_to_dump = {}
    # iterating over pair of original image - mask predicted
    for original_img, plane_image in tqdm(mapping_dict.items()):
        img_np2 = np.array(Image.open(mapping_dict[original_img]))
        x_row_lower_right, y_column_lower_right, x_row_left, y_column_left, x_row_top, y_column_top = \
            create_triangle_points(img_np2)
        gamma_angle = calculate_left_corner_angle(
            x_row_lower_right,
            y_column_lower_right,
            x_row_left,
            y_column_left,
            x_row_top,
            y_column_top)

        centroid = get_centroid(img_np2)
        print(f"Cerntroid = {centroid}")
        xx, yy = compute_resection_line_endpoint(centroid, gamma_angle, line_length=218)
        where_to_draw_line = compute_resection_line_mask(
            img_np2, centroid, xx, yy)

        description_object = {'slope_angle_art': round(degrees(gamma_angle), 1),
                              'centroid_art': [int(round(centroid[0])), int(round(centroid[1]))],
                              'end_point_art': [int(round(xx)), int(round(yy))], 'length_art': 218}
        json_to_dump[str(os.path.basename(original_img))] = description_object

        if saving_fig:
            size_X, size_Y = 1024, 1280  # put images resolution, else output may look wierd
            where_to_draw_line_modified = np.resize(np.asarray(
                Image.fromarray(where_to_draw_line).convert('RGB')), (size_X, size_Y, 3))

            # combining the original image with the plane
            image_plane = convert_lumina_to_alpha(plane_image)
            image_whole = rgb_to_rgba(original_img)
            added_image = cv2.addWeighted(image_whole, 1, image_plane, 0.2, 0)

            # alternative https://stackoverflow.com/questions/52520480/python-save-binary-line-masks-with-assigned-line-width-without-using-cv2
            fig = plt.figure(frameon=False)
            # fig.set_size_inches(1024, 1280)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(added_image, aspect='auto')
            plt.plot(centroid[1], centroid[0], 'ro')
            plt.plot(round(xx), round(yy), 'ro')
            plt.plot([centroid[1], round(xx)], [centroid[0], round(yy)], 'k-', lw=2)
            plt.savefig(
                f"./../output/{dir_crossval}/artificial_resec_line1/resection_line_orig_and_artificial/{os.path.basename(original_img)}",
                bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()

    with open(location_to_save, 'w') as f:
        json.dump(json_to_dump, f)
        print(f"Dumped json to {location_to_save}")


def create_plane_img_mapping(
        predicted_plane_list: List[str],
        original_img_list: List[str]):
    """
    Create mapping
        {original_image : its predicted plane}
    Args:
        predicted_plane_list: list of glob.glob typed planes
        original_img_list: list of glob.glob typed original images

    Returns:
        a dict with a mapping
{'./../../data/output/final_training/val/20200416_first incision_00020.png': '../output/final_training/images_predicted_multilabeled/predicted_plane/20200416_first incision_00020_01.png',
 './../../data/output/final_training/val/20200427_first incision_00003.png': '../output/final_training/images_predicted_multilabeled/predicted_plane/20200427_first incision_00003_01.png',
    """
    mapping_dict = {}
    for original_img in original_img_list:
        predicted_sub_list = [predicted_plane for predicted_plane in predicted_plane_list if os.path.basename(
            predicted_plane).startswith(os.path.basename(original_img)[:-4])]  # except the .png part

        if predicted_sub_list:
            mapping_dict[str(original_img)] = predicted_sub_list[0]
    return mapping_dict


def main():
    dir_to_process = "crossval_full/20190917"
    predicted_plane_list = sorted(glob.glob(
        f"./../output/{dir_to_process}/images_predicted_multilabeled/layerwise_masks/1/*.png"))
    original_img_list = glob.glob(f"./../../data/output/{dir_to_process}/val/*.png")
    mapping_dict = create_plane_img_mapping(
        predicted_plane_list, original_img_list)
    full_process_resection_line(mapping_dict=mapping_dict,
                                dir_crossval=dir_to_process,
                                location_to_save=f'./../../csv/detectron/plane_evaluation/artificial_resec_line/{dir_to_process[9:]}.json',
                                saving_fig=True)


if __name__ == '__main__':
    main()
