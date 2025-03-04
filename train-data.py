# Process valid folder
import os

from postprocessing import zip_folder, process_images_in_directory, remove_original_images
from segmentation import process_folder, predictor, mask_generator

folders = ['train']
base_path = 'working-dataset'
annotation_files = {'train': 'archive/train/_annotations.csv'}


# Process each folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    annotation_csv = annotation_files[folder]
    process_folder(folder_path, annotation_csv, predictor, mask_generator, base_path)

# Define the paths to the folders
base_path = 'working-dataset'
folders = ['train']

# Remove original images
remove_original_images(base_path, folders)

# Specify the root directory containing subdirectories with images
root_directory = 'working-dataset/train'

# Process the images
process_images_in_directory(root_directory)

base_path = 'working-dataset'
folders = ['train']

# Zip the folders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    zip_folder(folder_path)
