from collections import Counter
import os
import glob
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import KDTree
from typing import Tuple, List
from detectron.dataset_new_data import create_img_to_mask_layers_index, get_yaml_config, get_keys
import pandas as pd
from tqdm import tqdm

mapping = {
    0: 0,
    255: 123
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



def get_centroid(img_np:np.ndarray)-> Tuple[float,float]:
    """
    Calculating the centroid of the plane
    Args:
        img_np: image on which the centroid should be found

    Returns:
        two points of the centroid (x, y )
    """
    try:
        return ndimage.measurements.center_of_mass(img_np[:,:,1])
    except:
        return ndimage.measurements.center_of_mass(img_np[:, :])



def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)



def map_to_list(sequence_of_nums:np.ndarray) -> List[int]:
    """
    Convertint ndarray to list
    Args:
        sequence_of_nums:

    Returns:

    """
    return [num for num in sequence_of_nums]

def ndarray_to_points_list(resec_line):
    x_coordinates_resec_line, y_coordinates_resec_line =np.where(resec_line>0)
    x_coordinates_resec_line_list = x_coordinates_resec_line.tolist()
    y_coordinates_resec_line_list = y_coordinates_resec_line.tolist()
    tuples_list  = zip (x_coordinates_resec_line_list, y_coordinates_resec_line_list)
    return  list(tuples_list)


def generatekd_tree_for_line(resec_line)->Tuple [KDTree, np.ndarray]:

    x_coordinates_resec_line, y_coordinates_resec_line =np.where(resec_line>0)
    x_coordinates_resec_line_list = map_to_list(x_coordinates_resec_line)
    y_coordinates_resec_line_list = map_to_list(y_coordinates_resec_line)
    resec_np_line = np.array([x_coordinates_resec_line_list, y_coordinates_resec_line_list]).transpose()
    # resec_np_line_tuples = np.ndarray(ndarray_to_points_list(resec_line))
    tree = KDTree(resec_np_line)
    return tree, resec_np_line



def euclid_min (resection_line_tuples, centroid):
    min_distance = np.inf
    min_pair_index = 0
    for index_euc, values_pair in enumerate(resection_line_tuples):
        if (smaller_distance := np.linalg.norm(np.array(centroid)-np.array(values_pair)))<min_distance:
            # print(f'{smaller_distance=}  , {index_euc= }')
            min_distance = smaller_distance
            min_pair_index = index_euc
    return min_pair_index

def get_closest_point_to_centroid(coordinate:Tuple[float, float],
                                  resec_line:List[Tuple[int,int]])-> Tuple[int,int]:
    """
    Getting the coordinate og the closest point to the centroid
    Creating a point in a (1,2) shape
    Based om a the binary mask of the resection line creating a KD- Tree for fas NlogN search.
    Search fot a closest point (top k=1) to the centroud

    Args:
        resec_line:
        coordinate:

    Returns:

    """
    # centroid point in the shape as the K-D tree
    # centroid = np.array([int(coordinate[0]), int(coordinate[1])]).reshape(2,1).transpose()
    # tree, resec_np_line= generatekd_tree_for_line(resec_line)
    # dist, ind = tree.query(centroid, k=1)
    centroid = round(coordinate[0]), round(coordinate[1])
    min_pair_index = euclid_min(resec_line, centroid)
    x_closest_point_list, y_closest_point_list = resec_line[min_pair_index]
    # closest_point_list


    # closest_point_list = resec_np_line[ind,:].tolist()
    # x_closest_point_list, y_closest_point_list = [it for sublist in closest_point_list for item in sublist for it in item]
    return x_closest_point_list, y_closest_point_list


def get_distance_between_centroid_and_nearest_resection_point(coordinate,
                                                              x_closest_point_list,
                                                              y_closest_point_list)-> float:
    a = np.array([y_closest_point_list,x_closest_point_list])

    b = np.array([round(coordinate[0]), round(coordinate[1])])
    return np.linalg.norm(a-b)


def main():

    generate_images = False
    img_dir = sorted(glob.glob("./../../data/output/images_new/plane_val/foo/*.png"))
    # img_dir = sorted(glob.glob("../output/10_august_run/images_predicted_multilabeled/predicted_plane/*.png"))
    yaml_config = get_yaml_config('./../../configs/plane_config.yaml')
    # yaml_config = get_yaml_config('../configs/configs.yaml')
    #TODO fix the issue with not founding the masks for the image
    counter_of_images_in_single_nrrd = dict(Counter([os.path.basename(img)[:8] for img in img_dir]))
    image_to_masks_dict = create_img_to_mask_layers_index(counter_of_images_in_single_nrrd ,yaml_config=yaml_config)
    list_of_keys = sorted(get_keys(image_to_masks_dict))

    df = pd.DataFrame(columns=["Filename", 'centroid', "euclid_distance","plane_size"])


    for index_img, img_name in tqdm(enumerate(img_dir)):

        # index_img = 55
        # print(f"Image: {os.path.basename(img_name)}")
        try:
            img_np = np.array(Image.open(img_dir[index_img]))

            upper_triangle_point = [mask for mask in image_to_masks_dict[list_of_keys[index_img]] if mask.endswith("07.png")]
            lower_triangle_points = [mask for mask in image_to_masks_dict[list_of_keys[index_img]] if mask.endswith("08.png")]

            resection_line = [mask for mask in image_to_masks_dict[list_of_keys[index_img]] if mask.endswith("00.png")]



            centroid = get_centroid(img_np)

            upper_tr = np.array(Image.open(upper_triangle_point[0]))

            resec_line = np.array(Image.open(resection_line[0]))

            # lower_tr = np.array(Image.open(lower_triangle_points[0]))
            # triangle = np.logical_or(upper_tr, lower_tr)
            # gt_and_triangle = np.logical_or(resec_line, triangle)
            gt_and_triangle = mask_to_class(resec_line)
        except:
            print(f'Errror for {img_dir[index_img]}')
            continue
        if resec_line.max() != 0:
            # if upper_triangle_point is not None and lower_triangle_points is not None and resec_line_tuples is not None:

            # plt.imshow(gt_and_triangle)



            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            # ax1.imshow(img_np)
            # plt.plot(centroid[0], centroid[1], 'ro')
            # ax2.imshow(gt_and_triangle)
            # plt.show()

            resec_line_tuples = ndarray_to_points_list(resec_line)
            # closest_point = closest_node((centroid[0], centroid[1]), resec_line_tuples)
            x_closest_point_list, y_closest_point_list = get_closest_point_to_centroid(centroid, resec_line_tuples)

            euclidean_distance = get_distance_between_centroid_and_nearest_resection_point(centroid,
                                                                                            x_closest_point_list, y_closest_point_list)

            area_size_of_plane = np.count_nonzero(img_np)
            # print(f'{euclidean_distance=}')

            df.loc[index_img] = [os.path.basename(img_name)] + [centroid] + [euclidean_distance] + [area_size_of_plane]


            if generate_images:
                #todo нарисовать чтобы были разные цвета линии и предсказанного
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                ax1.set_title(f'Original Image {os.path.basename(img_name)}')
                ax1.imshow(img_np)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
                plt.plot(centroid[0], centroid[1], 'ro', label = 'Calculated centroid')
                plt.plot(y_closest_point_list,x_closest_point_list, 'co', label = 'Calculated closest point\n to resection line')
                ax2.set_title(f'GT triangle and resection line with centroid')
                ax2.imshow(gt_and_triangle)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                # fig.suptitle(f'{os.path.basename(img_name)}')
                # plt.show()
                # plt.savefig(f'./output/img/centroid_toresection_line/5_september/{os.path.basename(img_name)}', dpi=300)
                plt.savefig(f"../output/10_august_run/images_predicted_multilabeled/predicted_plane/"
                            f"centroids/{os.path.basename(img_name)}", dpi=300)
                print(f"Saving to ../output/10_august_run/images_predicted_multilabeled/predicted_plane/"
                            f"centroids/{os.path.basename(img_name)}")

        else:
            print(f'No annotated resection line for {os.path.basename(img_name)}')
    # df.to_csv('plane_evaluation_train.csv')
    df.to_csv('plane_evaluation_val_predicted_correct_1.csv')


if __name__ == '__main__':
    main()