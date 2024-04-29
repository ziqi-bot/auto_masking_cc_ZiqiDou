"""
Mask R-CNN

#NOTE:, I use tensorflow-gpu ==1.14 cudav10.0 cudnn7.4 scikit-image==0.16.2

(steps per epoch, epoch)
12T2355: 50,10
13T0436: 3,100
13T1149: 50,20
13T1340: 50.100
17T0028: 50,100

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/cartridgecase20240117T0028/mask_rcnn_cartridgecase_0200.h5")
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/cartridgecase20240113T1340/mask_rcnn_cartridgecase_0100.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CartridgeCaseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "CartridgeCase"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background +  4 my classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CartridgeCaseDataset(utils.Dataset):

    def load_CartridgeCase(self, dataset_dir, subset):
        """Load a subset of the Cartridge case dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only 4 class to add.
        self.add_class("CartridgeCase", 1, "breech face impression")
        self.add_class("CartridgeCase", 2, "firing pin impression")
        self.add_class("CartridgeCase", 3, "firing pin drag")
        self.add_class("CartridgeCase", 4, "direction")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "CartridgeCase",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,

                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a CartridgeCase dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "CartridgeCase":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        info = self.image_info[image_id]
        classID = np.zeros((len(info["polygons"]),))
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            if p['name'] == 'breech face impression':
                classID[i,] = 1
            elif p['name'] == 'firing pin impression':
                classID[i,] = 2
            elif p['name'] == 'firing pin drag':
                classID[i,] = 3
            else:
                classID[i,] = 4

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), classID

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "CartridgeCase":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CartridgeCaseDataset()
    dataset_train.load_CartridgeCase(os.path.join(ROOT_DIR, "imagesNew"), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CartridgeCaseDataset()
    dataset_val.load_CartridgeCase(os.path.join(ROOT_DIR, "imagesNew"), "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads')



############################################################
#  Training
############################################################

if __name__ == '__main__':

    # Validate arguments

    # Configurations

    config = CartridgeCaseConfig()
    config.display()

    # Create model

    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    weight ='coco'

    # Select weights file to load
    if weight == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weight == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weight == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weight

    # Load weights
    print("Loading weights ", weights_path)
    if weight == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate

    train(model)

