import os
os.environ["KMP_AFFINITY"]="disabled"
import csv
import math
import tensorflow as tf
import FK2018

import mrcnn.model as modellib
import pandas as pd
from mrcnn import utils
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
conf = K.tf.ConfigProto(intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)
K.set_session(K.tf.Session(config=conf))
print("imported keras")
import logging
from multiprocessing import Pool

global ts_dataset
global ae_dive1_dataset
global ae_dive2_dataset
global ae_dive3_dataset



def get_dataset_filepath(experiment):
    #filepath = "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/"
    filepath = "/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/images/processed/image/i20180805_215810/"
    if experiment["colour_correction_type"] == "histogram_normalised":
        filepath += "histogram_normalised/"
    elif experiment["colour_correction_type"] == "greyworld_corrected":
        filepath += "greyworld_correction/"
    elif experiment["colour_correction_type"] == "attenuation_correction":
        filepath += "attenuation_correction/"
    else:
        print(experiment["colour_correction_type"])
        exit()

    if experiment["distortion_correction"]:
        filepath += "distortion_correction/"
    else:
        filepath += "no_distortion_correction/"

    filepath += experiment['rescaled'] + "/"

    filepath += "val"
    return filepath


def get_ae2000_dataset_filepaths(experiment):
    #filepath = "/Volumes/jw22g14_phd/fk2018/ae2000/ae2000_overlap/coco/"
    filepath = "/scratch/jw22g14/FK2018/ae2000/ae2000_overlap/coco/"
    if experiment["colour_correction_type"] == "histogram_normalised":
        filepath += "histogram"
    elif experiment["colour_correction_type"] == "greyworld_corrected":
        filepath += "greyworld"
    elif experiment["colour_correction_type"] == "attenuation_correction":
        filepath += "attn"
    else:
        print(experiment["colour_correction_type"])
        exit()

    if experiment["distortion_correction"]:
        filepath += "_dist"
    else:
        filepath += "_nodist"

    if experiment["rescaled"]:
        filepath += "_rescaled/"
    else:
        filepath += "/"

    return [filepath + "dive1", filepath + "dive2", filepath + "dive3"]


def load_images(image_ids, dataset, config):
    images=[]
    for image_id in image_ids:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset, config, image_id, use_mini_mask=False
            )
            if image is None:
                print("Image not loaded - " + image_id)
                exit()
            images.append({'image':image, 'image_meta':image_meta, 'gt_class_id':gt_class_id, 'gt_bbox':gt_bbox, 'gt_mask':gt_mask})
    return images

def compute_classless_batch_ap(image_ids, dataset, model, config):
    APs = []
    size_APs = {}
    size_overlaps = {}
    overlaps_list = []
    for iteration in range(1):
        images=load_images(image_ids, dataset, config)
        logging.debug("running detection for iteration "+str(iteration))
        for n, image in enumerate(images):
            if len(image['gt_class_id']) > 0:
                logging.debug("getting stats for image " +str(image['image_meta'][0]))
                # Compute AP
                results = model.detect([image['image']], verbose=0)
                r = results[0]
                for id_num in range(len(r["class_ids"])):
                    r["class_ids"][id_num] = 0
                for id_num in range(len(image["gt_class_id"])):
                    image["gt_class_id"][id_num] = 0
                for i, roi in enumerate(r["rois"]):
                    mask_size = np.sum(r["masks"][:, :, i].flatten())
                    singleton_rois = np.array([roi])
                    singleton_class_ids = np.array([r["class_ids"][i]])
                    singleton_scores = np.array([r["scores"][i]])
                    singleton_masks = np.zeros([1024, 1024, 1])
                    singleton_masks[:, :, 0] = np.array(r["masks"][:, :, i])
                    AP, precisions, recalls, overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_class_id'],
                        image['gt_mask'],
                        np.array([roi]),
                        np.array([r["class_ids"][i]]),
                        np.array([r["scores"][i]]),
                        np.array([singleton_masks]),
                    )

                    if mask_size in size_APs.keys():
                        size_APs[mask_size].append(AP)
                        size_overlaps[mask_size].append(overlaps)
                    else:
                        size_APs[mask_size] = [AP]
                        size_overlaps[mask_size] = [overlaps]
                    APs.append(AP)
                    overlaps_list.append(overlaps)
    logging.debug("finished all 5 iterations for these images")
    return size_APs, APs, size_overlaps, overlaps_list

def compute_both_batch_aps(image_ids, dataset, model, config):
    APs = []
    class_list = []
    size_list = []
    overlaps_list = []
    classless_APs = []
    classless_overlaps_list = []
    total_predicted_pixels = 0
    total_groundtruth_pixels = 0
    total_overlapping_pixels = 0



    for iteration in range(1):
        images=load_images(image_ids, dataset, config)
        logging.debug("running detection for iteration "+str(iteration))
        for n, image in enumerate(images):
            if len(image['gt_class_id']) > 0:
                logging.debug("getting stats for image " +str(image['image_meta'][0]))
                # Compute AP
                results = model.detect([image['image']], verbose=0)
                r = results[0]
                r["blank_class_ids"]=[]
                image["gt_blank_class_ids"]=[]
                for id_num in range(len(r["class_ids"])):
                    r["blank_class_ids"].append(0)
                for id_num in range(len(image["gt_class_id"])):
                    image["gt_blank_class_ids"].append(0)
                for i, roi in enumerate(r["rois"]):
                    mask_size = np.sum(r["masks"][:, :, i].flatten())
                    predicted_pixels = np.any(r["masks"], axis=2)
                    groundtruth_pixels = np.any(image['gt_mask'], axis=2)
                    overlap_pixels = np.logical_and(predicted_pixels, groundtruth_pixels)
                    total_predicted_pixels += np.sum(predicted_pixels)
                    total_groundtruth_pixels += np.sum(groundtruth_pixels)
                    total_overlapping_pixels += np.sum(overlap_pixels)

                    singleton_rois = np.array([roi])
                    singleton_class_ids = np.array([r["class_ids"][i]])
                    singleton_scores = np.array([r["scores"][i]])
                    singleton_masks = np.zeros([1024, 1024, 1])
                    singleton_masks[:, :, 0] = np.array(r["masks"][:, :, i])
                    AP, precisions, recalls, overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_class_id'],
                        image['gt_mask'],
                        np.array([roi]),
                        np.array([r["class_ids"][i]]),
                        np.array([r["scores"][i]]),
                        np.array([singleton_masks]),
                    )
                    classless_AP, classless_precision, classless_recalls, classless_overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_blank_class_ids'],
                        image['gt_mask'],
                        np.array([roi]),
                        np.array([r["blank_class_ids"][i]]),
                        np.array([r["scores"][i]]),
                        np.array([singleton_masks]),
                    )

                    APs.append(AP)
                    class_list.append(r["class_ids"][i])
                    size_list.append(mask_size)
                    overlaps_list.append(np.max(overlaps))
                    classless_APs.append(classless_AP)
                    classless_overlaps_list.append(np.max(classless_overlaps))


    logging.debug("finished all 5 iterations for these images")
    return APs, class_list, size_list, overlaps_list, classless_APs, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels



def compute_batch_ap(image_ids, dataset, model, config):

    APs = []
    class_APs = {}
    size_APs = {}
    class_overlaps = {}
    size_overlaps = {}
    overlaps_list = []

    for iteration in range(15):
        images=load_images(image_ids, dataset, config)
        logging.debug("running detection for iteration "+str(iteration))
        for n, image in enumerate(images):
            if len(image['gt_class_id']) > 0:
                logging.debug("getting stats for image " +str(image['image_meta'][0]))
                # Compute AP
                results = model.detect([image['image']], verbose=0)
                r = results[0]
                for i, roi in enumerate(r["rois"]):
                    mask_size = np.sum(r["masks"][:, :, i].flatten())
                    singleton_rois = np.array([roi])
                    singleton_class_ids = np.array([r["class_ids"][i]])
                    singleton_scores = np.array([r["scores"][i]])
                    singleton_masks = np.zeros([1024, 1024, 1])
                    singleton_masks[:, :, 0] = np.array(r["masks"][:, :, i])
                    AP, precisions, recalls, overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_class_id'],
                        image['gt_mask'],
                        np.array([roi]),
                        np.array([r["class_ids"][i]]),
                        np.array([r["scores"][i]]),
                        np.array([singleton_masks]),
                    )
                    if r["class_ids"][i] in class_APs.keys():
                        class_APs[r["class_ids"][i]].append(AP)
                        class_overlaps[r["class_ids"][i]].append(overlaps)
                    else:
                        class_APs[r["class_ids"][i]] = [AP]
                        class_overlaps[r["class_ids"][i]] = [overlaps]

                    if mask_size in size_APs.keys():
                        size_APs[mask_size].append(AP)
                        size_overlaps[mask_size].append(overlaps)
                    else:
                        size_APs[mask_size] = [AP]
                        size_overlaps[mask_size] = [overlaps]
                    APs.append(AP)
                    overlaps_list.append(overlaps)
    logging.debug("finished all 5 iterations for these images")
    return class_APs, size_APs, APs, class_overlaps, size_overlaps, overlaps_list


def load_dataset(filepath):
    dataset = FK2018.FKDataset()
    dataset.load_fk(filepath)
    dataset.prepare()
    return dataset


def get_stats(weights_filepath, dataset):
    config = FK2018.FKConfig()
    print("got config")
    

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    # Must call before using the dataset

    logging.debug("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir="./log", config=config)

        # Load weights
    logging.debug("Loading weights ", weights_filepath)
    model.load_weights(weights_filepath, by_name=True)
    image_ids = dataset.image_ids
    APs, class_list, size_list, overlaps_list, classless_APs, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels = compute_both_batch_aps(image_ids, dataset, model, config)

    return APs, class_list, size_list, overlaps_list, classless_APs, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels



def plot_class_boxplots():
    experiments = get_experiments()
    print(experiments.columns)
    sns.boxplot(
        data=experiments,
        y=experiments.columns[16],
        x="colour_correction_type",
        hue="separate_channel_ops",
    )
    # experiments.boxplot(column=experiments.columns[16], by=["colour_correction_type", "distortion_correction"], rot=90)
    plt.show()


def directory_to_experiment_info(directory):
    colour_correction_type = directory.split("/")[-1].split("-")[0]
    if directory.split("/")[-1].split("-")[1] == "distortion_correction":
        distortion_correction = True
    else:
        distortion_correction = False

    rescaled = directory.split("/")[-1].split("-")[2] 
    number = int(directory.split("/")[-1].split("-")[-1])
    experiment = {
        "colour_correction_type": colour_correction_type,
        "distortion_correction": distortion_correction,
        "rescaled": rescaled,
        "number": number,
    }

    if (number % 4) / 2 < 1:
        experiment["elastic_distortions"] = True
    else:
        experiment["elastic_distortions"] = False
    if (number % 4) % 2 != 0:
        experiment["separate_channel_ops"] = True
    else:
        experiment["separate_channel_ops"] = False

    return experiment

def add_single_experiment(directory, df_filepath, datasets):
    csv_filenames = [
            f for f in os.listdir(directory) if f[0:5] == "resul" and f[-4:] == ".csv"
    ]
    weights_folder = [f for f in os.walk(directory)][0][1][0]
    for filename in csv_filenames:
        experiment = directory_to_experiment_info(directory)
        with open("./" + directory + "/" + filename, "r") as csvfile:
            logging.debug("./" + directory + "/" + filename)
            plots = csv.reader(csvfile, delimiter=",")
            headers = next(plots, None)
            for header in headers:
                experiment[header] = []
            for row in plots:
                for i, header in enumerate(headers):
                    experiment[header].append(float(row[i]))

        experiment["minimum_val_loss"] = min(experiment["val_loss"])
        experiment["minimum_loss"] = min(experiment["loss"])
        number = int(filename.split("_")[1].split(".")[0])
        experiment["number"] = number
        experiment["repeat"] = math.floor(number / 4)
        if (number % 4) / 2 < 1:
            experiment["elastic_distortions"] = True
        else:
            experiment["elastic_distortions"] = False
        if (number % 4) % 2 != 0:
            experiment["separate_channel_ops"] = True
        else:
            experiment["separate_channel_ops"] = False
        print(experiment.keys())

        if not experiment_in_dataframe(df_filepath, experiment):
            weights_file = (
                directory + "/" + weights_folder + "/" + "mask_rcnn_fk2018_best.h5"
            )
            if os.path.exists(weights_file):
                # try:
                logging.debug("getting stats")
                logging.debug("tunasand stats")
                experiment["APs"], \
                experiment["class_list"], \
                experiment["size_list"], \
                experiment["overlaps"], \
                experiment["classless_APs"], \
                experiment["classless_overlaps_list"], \
                experiment["total_predicted_pixels"], \
                experiment["total_groundtruth_pixels"], \
                experiment["total_overlapping_pixels"] = get_stats(weights_file, datasets[0])


                logging.debug("aestats, dive1")
                experiment['ae_APs_dive1'], \
                experiment['ae_class_list_dive1'], \
                experiment['ae_size_list_dive1'], \
                experiment['ae_overlaps_dive1'], \
                experiment["ae_classless_APs_dive1"], \
                experiment["ae_classless_overlaps_list_dive1"], \
                experiment["ae_total_predicted_pixels_dive1"], \
                experiment["ae_total_groundtruth_pixels_dive1"], \
                experiment["ae_total_overlapping_pixels_dive1"]  = get_stats(weights_file, datasets[1])


                logging.debug("aestats, dive2")
                experiment['ae_APs_dive2'], \
                experiment['ae_class_list_dive2'], \
                experiment['ae_size_list_dive2'], \
                experiment['ae_overlaps_dive2'], \
                experiment["ae_classless_APs_dive2"], \
                experiment["ae_classless_overlaps_list_dive2"], \
                experiment["ae_total_predicted_pixels_dive2"], \
                experiment["ae_total_groundtruth_pixels_dive2"], \
                experiment["ae_total_overlapping_pixels_dive2"]  = get_stats(weights_file, datasets[2])

                logging.debug("aestats, dive3")
                experiment['ae_APs_dive3'], \
                experiment['ae_class_list_dive3'], \
                experiment['ae_size_list_dive3'], \
                experiment['ae_overlaps_dive3'], \
                experiment["ae_classless_APs_dive3"], \
                experiment["ae_classless_overlaps_list_dive3"], \
                experiment["ae_total_predicted_pixels_dive3"], \
                experiment["ae_total_groundtruth_pixels_dive3"], \
                experiment["ae_total_overlapping_pixels_dive3"]   = get_stats(weights_file, datasets[3])

                # except:
                #     print("issue getting loss values")
            else:
                print("weights file doesn't exist")
                print(weights_file)
            update_dataframe(df_filepath, experiment)
        else:
            print("already in dataframe, skipping "+filename)



def populate_experiments_dataframe(number):
    print("LOADED IMPORTS - NOW RUNNING CODE")
    df_filepath = './classless_dataframe_'+str(number)+'.csv'
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',filename='log'+str(number)+'.log',level=logging.DEBUG)
    experiments = []
    dataset=array_num_to_dataset(number)
    files = [x for x in os.walk("../../../paper_experiments/weights/")][0][1]
    files = [x for x in files if x.split("-")[0:3] == dataset]
    directories = ["../../../paper_experiments/weights/" + f for f in files]
    experiment=directory_to_experiment_info(directories[0])
    dataset_filepath = get_dataset_filepath(experiment)
    ae_dataset_filepath_dive1, ae_dataset_filepath_dive2, ae_dataset_filepath_dive3 = get_ae2000_dataset_filepaths(experiment)
    datasets=[load_dataset(dataset_filepath), load_dataset(ae_dataset_filepath_dive1), load_dataset(ae_dataset_filepath_dive2), load_dataset(ae_dataset_filepath_dive3)]
    print(directories)
    for directory in directories:
         add_single_experiment(directory, df_filepath, datasets)
    return

def max_overlaps(experiments):
    for experiment in experiments:
         new_overlaps=[]
         print(experiment['overlaps'])

def experiment_in_dataframe(df_filepath, experiment):
    df = pd.read_csv(df_filepath)
    print(df)
    pre_existing_experiment = df.loc[
            (df["colour_correction_type"] == experiment["colour_correction_type"])
            & (df["distortion_correction"] == experiment["distortion_correction"])
            & (df["rescaled"] == experiment["rescaled"])
            & (df["number"] == experiment["number"])
        ]
    print(pre_existing_experiment)
    return not pre_existing_experiment.empty


def update_dataframe(df_filepath, experiment):
    df = pd.read_csv(df_filepath, index_col=0)
    df = df.append(experiment, ignore_index=True)
    logging.debug("saving experiment to "+df_filepath)
    df.to_csv(df_filepath)


def get_experiments():
    df = pd.read_csv("./experiments_dataframe.csv")
    return df


def create_dataframe(number):
    print("CREATING DATAFRAME")
    df = populate_experiments_dataframe(number)

def array_num_to_dataset(number):
    num_to_dataset_dict = {0:['histogram_normalised','no_distortion_correction','dropped_resolution'],
                           1:['histogram_normalised','no_distortion_correction','dropped_resolution_scaledup'],
                           2:['histogram_normalised','distortion_correction','dropped_resolution'],
                           3:['histogram_normalised','distortion_correction','dropped_resolution_scaledup'],
                           4:['greyworld_corrected','no_distortion_correction','dropped_resolution'],
                           5:['greyworld_corrected','no_distortion_correction','dropped_resolution_scaledup'],
                           6:['greyworld_corrected','distortion_correction','dropped_resolution'],
                           7:['greyworld_corrected','distortion_correction','dropped_resolution_scaledup'],
                           8:['attenuation_correction','no_distortion_correction','dropped_resolution'],
                           9:['attenuation_correction','no_distortion_correction','dropped_resolution_scaledup'],
                           10:['attenuation_correction','distortion_correction','dropped_resolution'],
                           11:['attenuation_correction','distortion_correction','dropped_resolution_scaledup'],
                           12:['histogram_normalised','no_distortion_correction','not_rescaled'],
                           13:['histogram_normalised','no_distortion_correction','rescaled'],
                           14:['histogram_normalised','distortion_correction','not_rescaled'],
                           15:['histogram_normalised','distortion_correction','rescaled'],
                           16:['greyworld_corrected','no_distortion_correction','not_rescaled'],
                           17:['greyworld_corrected','no_distortion_correction','rescaled'],
                           18:['greyworld_corrected','distortion_correction','not_rescaled'],
                           19:['greyworld_corrected','distortion_correction','rescaled'],
                           20:['attenuation_correction','no_distortion_correction','not_rescaled'],
                           21:['attenuation_correction','no_distortion_correction','rescaled'],
                           22:['attenuation_correction','distortion_correction','not_rescaled'],
                           23:['attenuation_correction','distortion_correction','rescaled']
     } 
    return num_to_dataset_dict[number]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pass job array id')
    parser.add_argument('array_id', type=int, help='job array id')
    args = parser.parse_args()
    number = args.array_id
    create_dataframe(number)
