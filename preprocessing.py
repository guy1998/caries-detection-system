import os
import numpy as np

import cv2

def preprocess_images(main_path, target_path):
    folders = ['train', 'valid', 'test']

    for folder in folders:
        input_folder_path = os.path.join(main_path, folder)
        output_folder_path = os.path.join(target_path, folder)

        # Create the target folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for filename in os.listdir(input_folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                img_path = os.path.join(input_folder_path, filename)
                img = cv2.imread(img_path)[:, :, 1]  # Read the image and select the second channel
                img2 = img - 255
                kernel = np.ones((2, 2), np.uint8)
                kernel2 = np.ones((3, 3), np.uint8)

                dilated_mask = cv2.dilate(img2, kernel, iterations=3)
                ret, thresh = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY)
                dilated_mask2 = cv2.dilate(thresh, kernel2, iterations=3)
                img = img / 255.0
                res_img = dilated_mask2 * img

                res_img = np.uint8(res_img)
                clahe_op = cv2.createCLAHE(clipLimit=20)
                final_img = clahe_op.apply(res_img)

                output_img_path = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_img_path, final_img)