"""
Usage: import the module, or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 dress.py train --dataset=/path/to/dress/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 dress.py train --dataset=/path/to/dress/dataset --weights=last

    # Inference. Create model with last or coco or custom specified weights. Take images from inputdir. Process it. Put result images in outputdir
    python3 dress.py inference --inputdir=/path/to/dir/with/image/to/find/mask --outputdir=/path/to/dir/where/to/put/                                                                                           processed/images 
                               --weights=(last | coco| path/to/weights)
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd
from rle_helper import rle_decode

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

MRNN_DIR = os.path.join(ROOT_DIR, 'externals/mask_rcnn')
sys.path.append(MRNN_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(MRNN_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_INPUT_PATH = os.path.join(ROOT_DIR, 'input')
DEFAULT_OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')


############################################################
#  Configurations
############################################################

# Resnet101
class DressConfig(Config): 
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "dress"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + dress

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class DressResnet50Config(DressConfig):
    BACKBONE = "resnet50"

class InferenceResnet50Config(DressResnet50Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class InferenceResnet101Config(DressConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    
############################################################
#  Dataset
############################################################

class DressDataset(utils.Dataset):
    def load_dress(self, dataset_dir, subset):
        """Load a subset of the Dress dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class('dress', 1, 'dress')
		
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)
        df = pd.read_csv(os.path.join(dataset_dir, f'{subset}_m.csv'))

        for index, row in df.iterrows():
            #print(row['ImageId'])
            image_path = os.path.join(dataset_dir, row['ImageId'])
            height = row['Height']
            width = row['Width']

            self.add_image(
                "dress",
                image_id=row['ImageId'],
                path=image_path,
                width=width, 
                height=height,
                encoded_pixels=row['EncodedPixels'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info['source'] != 'dress':
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        rle_mask = info['encoded_pixels']
        mask = rle_decode(rle_mask, (info['height'], info['width']))
        mask = mask.reshape((1, info['width'], info['height']))
        mask = mask.T

        return mask, np.ones(1, dtype=np.int32)

    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'dress':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DressDataset()
    dataset_train.load_dress(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DressDataset()
    dataset_val.load_dress(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')

def inference(model, image_dir, output_path):
    image_names = os.listdir(image_dir)
    image_names = [name for name in image_names if name.endswith('.png') or name.endswith('.jpg')]
    for image_path in image_names:
        print(os.path.join(image_dir,image_path))
        image = skimage.io.imread(os.path.join(image_dir,image_path))
        r = model.detect([image], verbose=1)[0]
        image_id = image_path.split('.')[0]
        visualize.save_processed_images(image, r['rois'], r['masks'], r['class_ids'], 
            ['BG', 'dress'], os.path.join(output_path, f'{image_id}.png'), r['scores'])
      

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect dress.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dress/dataset/",
                        help='Directory of the Dress dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--inputdir', required=False,
                        default=DEFAULT_INPUT_PATH,
                        metavar="/path/to/image dir/")
    parser.add_argument('--outputdir', required=False,
                        default=DEFAULT_OUTPUT_PATH,
                        metavar="/path/to/output/",
                        help='output dir for inference')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
   
    args = parser.parse_args()

    # Validate arguments
    #assert args.dataset, "Argument --dataset is required for training"
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    #else :
        #assert args.dataset, "Argument --dataset is required for inference"

    print("Weights: ", args.weights)
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Inputdir: ", args.inputdir)
    print("Outputdir: ", args.outputdir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DressConfig()
    else:
        config = InferenceResnet101Config()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        if not os.path.exists(COCO_WEIGHTS_PATH):
            utils.download_trained_weights(COCO_WEIGHTS_PATH)
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "inference":
        inference(model, args.inputdir, args.outputdir)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))
