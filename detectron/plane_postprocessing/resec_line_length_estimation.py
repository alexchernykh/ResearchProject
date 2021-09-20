import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt
from detectron.plane_postprocessing.plane_evaluation import map_to_list, ndarray_to_points_list
import json
import os
import glob
import tqdm


class AverageResectionLineDistance:
    """
    This class performs calculation of the resection line length and estimates the mean
    resection line length from the trainings data.
    """

    def __init__(self,
                 img_dir,
                 location_to_save: str = "./length.json"):
        """"
        Args:
            img_dir: directory with masks
            location_to_save: where to save the json file
        """
        self.img_dir = img_dir
        self.resection_images = [img for img in self.img_dir if img.endswith("00.png")]
        self.location_to_save = location_to_save

    @staticmethod
    def calculate_distance_matrix(resec_line_coordinatees_list: List[Tuple[int, int]]) -> np.ndarray:
        """
        Creating a distance matrix as a DataFrame for further search on the matrix values.
        Args:
            resec_line_coordinatees_list:

        Returns:

        """
        data = pd.DataFrame(resec_line_coordinatees_list, columns=['x', 'y'])
        samples_amount = data.shape[0]
        distance_matrix = np.zeros((samples_amount, samples_amount))
        for sample_i in range(samples_amount):
            distance_matrix[:, sample_i] = np.linalg.norm(
                data - data.iloc[sample_i], axis=1)
            # set diagonal values to inf
            distance_matrix[sample_i, sample_i] = 0
            # set the matrix values of the  upper part of the symmetrical matrix to
            # inf
        for sample_i in range(samples_amount):
            distance_matrix[sample_i, :sample_i] = 0

        return distance_matrix

    def get_edge_point(self,
                       mask_draft: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """
        Getting the maximal distance between two points
        Args:
            mask_draft: resection line mask on which should be done search for longest distance

        Returns:
            - the coordinates of the two points where distance is maximal
            - maximal distance
        """
        resec_line_coordinatees_list = ndarray_to_points_list(mask_draft)

        distance_matrix = self.calculate_distance_matrix(resec_line_coordinatees_list)

        # get the elements with the maximal distance from each other
        max_resec_line_length = np.amax(distance_matrix)
        first_elem, second_elem = np.where(
            distance_matrix == np.amax(distance_matrix))

        first_elem = first_elem.tolist()[0]
        second_elem = second_elem.tolist()[0]
        return resec_line_coordinatees_list[second_elem], resec_line_coordinatees_list[
            first_elem], max_resec_line_length

    @staticmethod
    def visualise_points(mask_draft: np.ndarray,
                         starting_point: Tuple[int, int],
                         endpoint: Tuple[int, int]
                         ) -> None:
        """
        Visualisastion of the resection line mask and the two farest points
        Args:
            mask_draft: resection line mask
            starting_point: first farest point
            endpoint:second farest point

        Returns:
            - only visualisation of the mask and the points
        """
        plt.imshow(mask_draft)
        plt.plot(starting_point[1],
                 starting_point[0],
                 'bo',
                 label='Calculated centroid')
        plt.plot(endpoint[1],
                 endpoint[0],
                 'ro',
                 label='Calculated centroid')
        plt.show()

    @staticmethod
    def single_resec_line_calculation(filename,
                                      visualise=False
                                      ) -> float:
        """
        To calculate the resection line length simply collect the two farrest points.
        Distance of a point will be judged based on the euclidean distance.

        Args:
            filename: filename of the resection line mask
            visualise:
        Returns:
            the distance of the resection line
        """
        mask_draft = np.array(Image.open(filename))
        starting_point, endpoint, max_length = get_edge_point(mask_draft)
        if visualise:
            visualise_points(mask_draft=mask_draft,
                             starting_point=starting_point,
                             endpoint=endpoint)
        return max_length

    def calculate_average_resec_line_length(self) -> None:
        """
        Calculating the resection line length of a GT annotation mask for all masks in a dir.

        Returns:
            only saving the the json with the {image_name: line_length} structure and a mean value
        """

        to_dump_length = {}

        for resec_img in tqdm.tqdm(self.resection_images):
            try:
                max_length = single_resec_line_calculation(resec_img)
                to_dump_length[str(os.path.basename(resec_img))] = round(max_length, 2)
            except BaseException:
                print(
                    f"could not calculate length for {os.path.basename(resec_img)}")

        mean_length = np.mean(list(to_dump_length.values()))
        to_dump_length["Mean"] = round(mean_length)

        with open(self.location_to_save, 'w') as f:
            json.dump(to_dump_length, f)


def main():
    img_dir = glob.glob("./../../data/output/final_training/masks/*.png")
    # use only the images with resection line
    resec_distance = AverageResectionLineDistance(img_dir=img_dir,
                                                  location_to_save='./../../csv/detectron/plane_evaluation/length.json')
    resec_distance.calculate_average_resec_line_length()


if __name__ == '__main__':
    main()
