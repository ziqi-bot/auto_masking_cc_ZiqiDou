"""
check result
in terminal: tensorboard --host 0.0.0.0 --logdir ./logs/ --port 9090 (--samples_per_plugin=images=40)
and then open http://127.0.0.1:9090 (!NOT http://0.0.0.0:9090/ ) in browser, press ctrl+c to return

automatically masking
model weights
"""


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.visualize import display_instances, display_instancesNew

class CartridgeCaseConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "CartridgeCase"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background +  4 my classes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def process_images_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the model
    config = CartridgeCaseConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
    # model.load_weights('logs/cartridgecase20240117T0028/mask_rcnn_cartridgecase_0300.h5', by_name=True)  #new random cases
    model.load_weights('logs/cartridgecase20240113T1340/mask_rcnn_cartridgecase_0100.h5', by_name=True)

    # Iterate through images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Perform detection
            result = model.detect([image])
            class_names = ['BG', 'breech face impression', 'firing pin impression', 'firing pin drag', 'direction']

            # Create a matplotlib figure
            fig, ax = plt.subplots(1, figsize=(16, 16))

            # Draw and save the annotated image using OpenCV
            display_instances(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'], class_names,
                                scores=None, title="", ax=ax,
                                show_mask=True, show_bbox=True, colors=['red', 'purple', 'cyan'], captions=None)
             # display_instancesNew(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'], class_names,
            #                     scores=None, title="", ax=ax,
            #                     show_mask=True, show_bbox=True, colors=['red', 'purple', 'cyan'], captions=None)

            # Save the annotated image using OpenCV with the original resolution and color format
            output_path = os.path.join(output_folder, filename)
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=fig.dpi)

            plt.close()

# Specify the input and output folders
# input_folder_path = 'imagesNew/test'
# output_folder_path = 'imagesNew/testresult'
input_folder_path = 'images/test'
output_folder_path = 'images/testresult'

# Process images and save the results
process_images_folder(input_folder_path, output_folder_path)
