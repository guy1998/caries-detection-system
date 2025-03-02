import os
import numpy as np
from PIL import Image
import zipfile


def remove_original_images(base_path, folders):
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and not filename.endswith(('.csv', '.zip')):  # Ensure not to remove other files
                os.remove(file_path)
                # print(f'Removed original image: {file_path}')


def crop_and_resize_image(image_path, output_size=(64, 64)):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert image to numpy array
        img_array = np.array(img)

        # Find non-zero pixels
        non_zero_indices = np.argwhere(img_array)

        # Get the bounding box of non-zero pixels
        top_left = non_zero_indices.min(axis=0)
        bottom_right = non_zero_indices.max(axis=0)

        # Crop the image to the bounding box
        cropped_img = img_array[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

        # Convert cropped image back to PIL Image
        cropped_img_pil = Image.fromarray(cropped_img)

        # Resize the image
        resized_img = cropped_img_pil.resize(output_size)

        # Save the resized image, overwriting the original
        resized_img.save(image_path)


def process_images_in_directory(root_directory):
    # Walk through the root directory
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(subdir, file)
                crop_and_resize_image(image_path)
                # print(f"Processed: {image_path}")


def zip_folder(folder_path):
    zip_file_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    print(f'Zipped folder: {zip_file_path}')