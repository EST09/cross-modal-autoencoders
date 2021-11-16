import pandas as pd
from PIL import Image

image_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GBM-image.csv"
sequence_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GSM4567140_PJ069.matrix.txt"

class Matched:

    def __init__(self, image_path, sequence_path):
        self.image_path = image_path
        self.sequence_path = sequence_path
        self.drop_empty_image_rows()
        self.get_matched_cells()
        self.get_centroids_of_matched()


    def drop_empty_image_rows(self):
        image_data = pd.read_csv(self.image_path, index_col=0)
        nan_value = float("NaN")
        image_data.replace("", nan_value, inplace=True)
        image_data.dropna(subset = ["lane"], inplace=True)
        return image_data

    def get_matched_cells(self):
        image_subset = self.drop_empty_image_rows()
        sequence_data = pd.read_csv(self.sequence_path, delimiter = "\t")
        # subset so that each row in image_subset has a column in sequence_data
        image_subset = image_subset[image_subset.index.isin(sequence_data.columns)]
        return image_subset

    def get_centroids_of_matched(self):
        image_data = self.get_matched_cells()
        centroids = image_data[["c2_XM", "c2_YM"]]
        return centroids

# samples = Matched(image_path, sequence_path)
# centroids = samples.get_centroids_of_matched()


def load_tif():
    pass


def crop_image_to_single_cell():
    # https://forum.image.sc/t/saving-each-roi-as-individual-images/3227/13
    pass