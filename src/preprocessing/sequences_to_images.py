import pandas as pd
from PIL import Image
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import napari

Image.MAX_IMAGE_PIXELS = 600000000

image_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GBM-image.csv"
sequence_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GSM4567140_PJ069.matrix.txt"
tif_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GBM-BF-c1.tif"

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


# apply above
samples = Matched(image_path, sequence_path)
centroids = samples.get_centroids_of_matched()
# number of cells you would like
centroids.iloc[:100,:].to_csv("roi_with_index.csv") #this path is hardcoded in imagej scripts will need to change

## Open imagej ##

#cropping is done in imagej
#requires index to be present
#roi_csv_with_index
# plugins -> new -> macro
# run points_to_roi_with_names.ijm
# run crop_image_with_names.ijm
# 50x50 seems to work well 

## Will need to do extra formatting to just get each single cell ##

def get_single_cell_at_roi():
    #find cell index falls within
    #keep only largest cell 
    pass
