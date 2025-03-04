# Process valid folder
import os

from postprocessing import zip_folder, process_images_in_directory, remove_original_images
from segmentation import process_folder, predictor, mask_generator

folders = ['valid']
base_path = 'working-dataset'
annotation_files = {'valid': 'archive/valid/_annotations.csv'}


# Process valid folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    annotation_csv = annotation_files[folder]
    process_folder(folder_path, annotation_csv, predictor, mask_generator, base_path)


# Define the paths to the folders
base_path = 'working-dataset'
folders = ['valid']

# Remove original images
remove_original_images(base_path, folders)

# Specify the root directory containing subdirectories with images
root_directory = 'working-dataset/valid'

# Process the images
process_images_in_directory(root_directory)

base_path = 'working-dataset'
folders = ['valid']

# Zip valid folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    zip_folder(folder_path)