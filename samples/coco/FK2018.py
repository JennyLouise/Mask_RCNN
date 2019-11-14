"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug.augmenters as iaa  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from skimage.measure import find_contours
import zipfile
import urllib.request
import shutil
import json
import math
import shapely.geometry

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2018"

############################################################
#  Configurations
############################################################


class FKConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fk2018"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 33#New tunasand set has 33, secondset has 15  # COCO has 80 classes
    STEPS_PER_EPOCH=100
    BATCH_SIZE=16

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1,
        "rpn_bbox_loss": 1,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1,
        "mrcnn_mask_loss": 1
    }
    
    RPN_NMS_THRESHOLD = 0.9
    TRAIN_ROIS_PER_IMAGE = 100 

############################################################
#  Dataset
############################################################

class FKDataset(utils.Dataset):

    def load_fk(self, dataset_dir, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        
        coco = COCO("{}/annotations.json".format(dataset_dir))
        image_dir = dataset_dir

        

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(FKDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FKDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        return info["path"]
        # if info["source"] == "coco":
        #     return "{}".format(info["id"])
        # else:
        #     super(FK2018Dataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]


            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }

            results.append(result)
    return results


def generate_labelme_file(model, dataset, output_dir, label_file):
    image_ids = dataset.image_ids
    t_prediction = 0
    t_start = time.time()

    labels={}
    with open(label_file, 'r') as f:
        i=0
        for class_name in f.readlines():
            labels[i] = class_name.strip()
            i+=1

    print(labels)
    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)



        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        imagename = dataset.image_info[image_id]['path'].split('/')[-1].split('.')[0]
        print("Predicting objects in {}".format(imagename))
        labelme_dict= {
                "imagePath": imagename+'.jpg',
                "imageData": None,
                "shapes": [],
                "version": "3.16.2", 
                "flags": {}, 
                "fillColor": [85, 170, 0, 128], 
                "lineColor": [0, 255, 0, 128], 
                "imageWidth": dataset.image_info[image_id]['width'], 
                "imageHeight": dataset.image_info[image_id]['height']
        }
        for i in range(r['rois'].shape[0]):
            class_id = r['class_ids'][i]
            score = r['scores'][i]
            bbox = np.around(r['rois'][i], 1)
            mask = r['masks'][:, :, i]

            polygon = find_contours(mask, 0.5)[0].tolist()
            n = math.ceil(len(polygon)/20)
            print(len(polygon))
            # polygon = polygon[0::n]
            polygon = shapely.geometry.Polygon(polygon)
            polygon = polygon.simplify(1)
            polygon = list(polygon.exterior.coords)
            print(len(polygon))

            for i in range(len(polygon)):
                polygon[i]=[polygon[i][1], polygon[i][0]]

            labelme_dict['shapes'].append({
                "line_color":None,
                "shape_type": "polygon", 
                "points": polygon,
                "flags": {}, 
                "fill_color": [ 255, 0, 0, 128 ], 
                "label": labels[class_id]
                })
        
        
        out_ann_file = output_dir +"/"+ imagename+'.json'
        with open(out_ann_file, 'w') as f:
            json.dump(labelme_dict, f)


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    print(results)

    # Load results. This modifies results with additional attributes.
    # coco_results = coco.loadRes(results)

    # # Evaluate
    # cocoEval = COCOeval(coco, coco_results, eval_type)
    # cocoEval.params.imgIds = coco_image_ids
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    # print("Prediction time: {}. Average {}/image".format(
    #     t_prediction, t_prediction / len(image_ids)))
    # print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

def train_nnet(section1_epochs=10, section2_epochs=20, section3_epochs=100, learning_rate=0.01, learning_momentum=0.9, 
                optimiser='Adam', add_freq=0.1, add_value=(-10,10), add_pc_freq=0.5, multiply_freq=0.1, 
                multiply_value=(0.75,1.25), multiply_pc_freq=0.5, snp_freq=0.1, snp_p=0.05, jpeg_freq=0.1, 
                jpeg_compression=(1,5), gaussian_freq=0.1, gaussian_sigma=(0.01,0.7), motion_freq=0.1, motion_k=(3,10), 
                contrast_freq=0.1, contrast_alpha=(0.5,1.5), fliplr=0.5, flipud=0.5, affine_freq=0.1, 
                affine_scale=(0,0.02), transform_freq=0.1, transform_scale=(0,0.05), elastic_freq=0.1, elastic_sigma=(4, 6), 
                elastic_alpha=(0,7), rotate=1, dataset="/scratch/jw22g14/FK2018/second_set/", log_file=""):
    config = FKConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR+log_file)
    model_path = COCO_MODEL_PATH
    model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    dataset_train = FKDataset()
    dataset_train.load_fk(dataset+"train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FKDataset()
    dataset_val.load_fk(dataset+"val")
    dataset_val.prepare()


    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = iaa.Sequential([
        iaa.Sometimes(add_freq, iaa.Add(value=add_value, per_channel=add_pc_freq)),
        iaa.Sometimes(multiply_freq, iaa.Multiply(mul=multiply_value, per_channel=multiply_pc_freq)),
        iaa.Sometimes(snp_freq, iaa.SaltAndPepper(snp_p)),
        iaa.Sometimes(jpeg_freq, iaa.JpegCompression(compression=jpeg_compression)),
        iaa.Sometimes(gaussian_freq, iaa.GaussianBlur(sigma=gaussian_sigma)),
        iaa.Sometimes(motion_freq, iaa.MotionBlur(k=motion_k)),
        iaa.Sometimes(contrast_freq, iaa.LinearContrast(alpha=contrast_alpha)),
        iaa.Fliplr(fliplr),
        iaa.Flipud(flipud),
        iaa.Sometimes(affine_freq, iaa.PiecewiseAffine(scale=affine_scale, nb_rows=8, nb_cols=8,polygon_recoverer='auto')),
        iaa.Sometimes(transform_freq, iaa.PerspectiveTransform(scale=transform_scale, keep_size=True)),
        iaa.Sometimes(elastic_freq, iaa.ElasticTransformation(sigma=elastic_sigma, alpha=elastic_alpha)),
        iaa.Sometimes(rotate, iaa.Rot90([0,1,2,3]))

        # iaa.Fliplr(0.5), # horizontal flips
        # iaa.Crop(percent=(0, 0.1)), # random crops
        # # Small gaussian blur with random sigma between 0 and 0.25.
        # # But we only blur about 50% of all images.
        # iaa.Sometimes(0.5,
        #     iaa.GaussianBlur(sigma=(0, 0.25))
        # ),
        # # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.75, 1.5)),
        # # Add gaussian noise.
        # # For 50% of all images, we sample the noise once per pixel.
        # # For the other 50% of all images, we sample the noise per pixel AND
        # # channel. This can change the color (not only brightness) of the
        # # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
        # # Make some images brighter and some darker.
        # # In 20% of all cases, we sample the multiplier once per channel,
        # # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2)),
        # # Apply affine transformations to each image.
        # # Scale/zoom them, translate/move them, rotate them and shear them.
        # iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-180, 180),
        #     #shear=(-8, 8)
        # )
    ], random_order=True)

    #setting augmentation to original "old main" version
#   augmentation = iaa.Sometimes(.667, iaa.Sequential([
#            iaa.Fliplr(0.5), # horizontal flips
#            iaa.Crop(percent=(0, 0.1)), # random crops
#            # Small gaussian blur with random sigma between 0 and 0.25.
#            # But we only blur about 50% of all images.
#            iaa.Sometimes(0.5,
#                iaa.GaussianBlur(sigma=(0, 0.25))
#            ),
#            # Strengthen or weaken the contrast in each image.
#            iaa.ContrastNormalization((0.75, 1.5)),
#            # Add gaussian noise.
#            # For 50% of all images, we sample the noise once per pixel.
#            # For the other 50% of all images, we sample the noise per pixel AND
#            # channel. This can change the color (not only brightness) of the
#            # pixels.
#            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
#            # Make some images brighter and some darker.
#            # In 20% of all cases, we sample the multiplier once per channel,
#            # which can end up changing the color of the images.
#            iaa.Multiply((0.8, 1.2)),
#            # Apply affine transformations to each image.
#            # Scale/zoom them, translate/move them, rotate them and shear them.
#            iaa.Affine(
#                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#                rotate=(-180, 180),
#                #shear=(-8, 8)
#            )
#        ], random_order=True))

    print("Training RPN")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=5, #40
                layers='rpn',
                augmentation=augmentation)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=section1_epochs, #40
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=section2_epochs, #120
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate/10,
                epochs=section3_epochs, #160
                layers='all',
                augmentation=augmentation)











def old_main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on FK2018.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on FK2018")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/fk2018/",
                        help='Directory of the FK2018 dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')



    parser.add_argument('--section1_epochs',
                        default=10,
                        required=False,
                        type=int)
    parser.add_argument('--section2_epochs',
                        default=20,
                        required=False,
                        type=int)
    parser.add_argument('--section3_epochs',
                        default=80,
                        required=False,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.001,
                        required=False,
                        type=float)
    parser.add_argument('--learning_momentum',
                        default=0.9,
                        required=False,
                        type=float)
    parser.add_argument('--optimiser',
                        default='Adam',
                        required=False,
                        type=str)
    





    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FKConfig()
    else:
        class InferenceConfig(FKConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    model.keras_model.summary()
    # Load weights
    print("Loading weights ", model_path)

    # Train or evaluate
    if args.command == "train":
        model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = FKDataset()
        dataset_train.load_fk(args.dataset+"train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = FKDataset()
        dataset_val.load_fk(args.dataset+"val")
        dataset_val.prepare()


        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = iaa.Sometimes(.667, iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.25.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.25))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180),
                #shear=(-8, 8)
            )
        ], random_order=True)) # apply augmenters in random order

        # *** This training schedule is an example. Update to your needs ***


        print("Training RPN")
        

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        model.load_weights(model_path, by_name=True)

        # Validation dataset
        dataset_val = FKDataset()
        coco = dataset_val.load_fk(args.dataset+"val", return_coco=True)
        dataset_val.prepare()
        print("Running FK2018 evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
    #old_main()

    train_nnet(dataset="/scratch/jw22g14/FK2018/first_set/")

