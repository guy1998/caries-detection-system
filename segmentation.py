import os
import pandas as pd
import numpy as np

import cv2

from tqdm import tqdm


def load_image_and_annotations(image_path, annotation_csv):
    # Load the image
    image = cv2.imread(image_path)
    # Read the annotation CSV
    annotations = pd.read_csv(annotation_csv)
    # Filter annotations for the given image
    annotations = annotations[annotations['filename'] == os.path.basename(image_path)]
    return image, annotations


def segment_injured_tooth(image, bbox, predictor):
    # Set the image for the predictor
    predictor.set_image(image)
    # Prepare the input box for the segment-anything tool
    input_box = np.array(bbox)
    # Perform segmentation
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False
    )
    return masks, scores


def save_segmented_part(image, mask, label, output_folder, part_index, filename):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Extract the segmented part using the mask
    segmented_part = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    # Define the output path with the original filename and part index
    output_path = os.path.join(output_folder, f'{filename}_segment_{part_index}.png')
    # Save the segmented part
    cv2.imwrite(output_path, segmented_part)
    # print(f'Segmented part saved to {output_path}')


def segment_normal_teeth(image, annotations, mask_generator, normal_folder, filename):
    # Generate masks for the whole image
    masks = mask_generator.generate(image)
    if not os.path.exists(normal_folder):
        os.makedirs(normal_folder)
    # Save each mask that doesn't overlap with any bbox and isn't all black
    h, w, _ = image.shape
    for i, mask in enumerate(masks):
        if np.count_nonzero(mask['segmentation']) < 0.1 * h * w:  # Skip if the mask covers more than 10% of the entire image
            segmented_part = cv2.bitwise_and(image, image, mask=mask['segmentation'].astype(np.uint8))
            if not np.all(segmented_part == 0):  # Check if the segmented part is not all black
                output_path = os.path.join(normal_folder, f'{filename}_segment_{i}.png')
                cv2.imwrite(output_path, segmented_part)
                # print(f'Normal part saved to {output_path}')


def black_out_injured_areas(image, annotations):
    # Create a mask with the same dimensions as the image, initialized to zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Fill the regions corresponding to the injured areas with 255
    for index, row in annotations.iterrows():
        cv2.rectangle(mask, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), 255, -1)
    # Black out the injured areas in the image
    blacked_out_image = image.copy()
    blacked_out_image[mask == 255] = 0
    return blacked_out_image


def remove_images_below_threshold(folder_path, threshold):
    """
    Remove images that have a percentage of non-zero pixels below the given threshold.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read image: {file_path}")
            continue
        total_pixels = image.size
        non_zero_pixels = np.count_nonzero(image)
        percent_non_zero = (non_zero_pixels / total_pixels) * 100
        if percent_non_zero < threshold:
            os.remove(file_path)
            # print(f"Removed image below threshold: {file_path} ({percent_non_zero:.2f}% non-zero pixels)")


def process_folder(folder_path, annotation_csv, predictor, mask_generator, base_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    for image_file in tqdm(image_files, desc=f'Processing {folder_path}'):
        image_path = os.path.join(folder_path, image_file)
        image, annotations = load_image_and_annotations(image_path, annotation_csv)
        # Process each annotation
        for index, row in annotations.iterrows():
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            label = row['class']
            # Segment the injured tooth
            masks, scores = segment_injured_tooth(image, bbox, predictor)
            # Save the segmented part of the first mask with a unique name
            save_segmented_part(image, masks[0], label, os.path.join(base_path, folder_path, label), index, os.path.splitext(image_file)[0])
        # Black out the injured areas in the image
        blacked_out_image = black_out_injured_areas(image, annotations)
        # Segment and save normal teeth
        segment_normal_teeth(blacked_out_image, annotations, mask_generator, os.path.join(base_path, folder_path, 'Normal'), os.path.splitext(image_file)[0])
        # Remove images below the threshold
        remove_images_below_threshold(os.path.join(base_path, folder_path, 'Normal'), threshold=0.21)
