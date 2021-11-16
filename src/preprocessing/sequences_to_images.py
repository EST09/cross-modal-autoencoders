import pandas as pd

image_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GBM-image.csv"
sequence_path = "/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/data_folder/my_data/GSM4567140_PJ069.matrix.txt"

def drop_empty_image_rows(image_path):
    image_data = pd.read_csv(image_path, index_col=0)
    nan_value = float("NaN")
    image_data.replace("", nan_value, inplace=True)
    image_data.dropna(subset = ["lane"], inplace=True)
    return image_data


def get_matched_cells(sequence_path, image_path):
    matched_coords = []
    image_subset = drop_empty_image_rows(image_path)
    print(image_subset.shape)
    sequence_data = pd.read_csv(sequence_path, delimiter = "\t")
    
    image_subset = image_subset[image_subset.index.isin(sequence_data.columns)]
    print(image_subset.shape)
    
get_matched_cells(sequence_path, image_path)


def get_centroids_of_matched():
    pass

def crop_image_to_single_cell():
    pass