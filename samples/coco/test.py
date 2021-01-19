import os
os.environ["KMP_AFFINITY"]="disabled"
import csv
import math
import tensorflow as tf
import FK2018
import objgraph
from pympler.tracker import SummaryTracker

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
    filepath = "/home/jenny/Documents/FK2018/tunasand/05_dive/"
    # filepath = "/scratch/jw22g14/FK2018/tunasand/20180805_215810_ts_un6k/images/processed/image/i20180805_215810/"
    if experiment["colour_correction_type"] == "histogram_normalised":
        filepath += "histogram_normalised/"
    elif experiment["colour_correction_type"] == "grey":
        filepath += "greyworld/"
    elif experiment["colour_correction_type"] == "alt":
        filepath += "altitude_corrected/"
    else:
        print(experiment["colour_correction_type"])
        exit()

    if experiment["distortion_correction"]:
        filepath += "distortion_correction/"
    else:
        filepath += "no_distortion/"

    rescaling_dict={"res_nn":"rescaled_nn", "drop_res":"dropped_resolution", "rescaled":"rescaled", "not_rescaled":"not_rescaled", "drop_res_up":"dropped_resolution_scaledup", "drop_res_up_nn": "dropped_resolution_scaledup_nn", "drop_res_nn": "dropped_resolution_nn"}

    filepath += rescaling_dict[experiment['rescaled']] + "/"

    filepath += "val"
    return filepath


def get_ae2000_dataset_filepaths(experiment):
    filepath = "/home/jenny/Documents/FK2018/ae/"
    if experiment["colour_correction_type"] == "raw":
        filepath += "raw/"
    elif experiment["colour_correction_type"] == "grey":
        filepath += "greyworld/"
    elif experiment["colour_correction_type"] == "alt":
        filepath += "altitude/"
    else:
        print(experiment["colour_correction_type"])
        exit()

    if experiment["distortion_correction"]:
        filepath += "distortion_corrected/"
    else:
        filepath += "no_distortion_correction/"

    filepath_list = []

    #TODO match
    rescaling_dict={"res_nn":"upscaled_nn", "drop_res":"rescaled", "rescaled":"upscaled", "not_rescaled":"not_rescaled", "drop_res_up":"upscaled", "drop_res_up_nn": "upscaled_nn", "drop_res_nn": "rescaled_nn"}
    rescaling_type =rescaling_dict[experiment['rescaled']]
    for area in ["area1", "area2", "area3"]:
        filepath_list.append(f"{filepath}{rescaling_type}/{area}")
    return filepath_list


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


def compute_both_batch_aps(image_ids, dataset, model, config):
    AP_list = []
    classless_AP_list = []
    precision_list = []
    classless_precision_list = []
    recall_list = []
    classless_recall_list = []
    predicted_class_list = []
    gt_class_list = []
    predicted_size_list = []
    gt_size_list = []
    overlaps_list = []
    classless_overlaps_list = []
    total_predicted_pixels = 0
    total_groundtruth_pixels = 0
    total_overlapping_pixels = 0


    iterations=5
    for iteration in range(iterations):
        images=load_images(image_ids, dataset, config)
        logging.debug("running detection for iteration "+str(iteration))
        for n, image in enumerate(images):
            if len(image['gt_class_id']) > 0:
                logging.debug("getting stats for image " +str(image['image_meta'][0]))

                # Compute AP
                results = model.detect([image['image']], verbose=0)
                r = results[0]
                logging.debug(f"With {len(r['rois'])} regions of interest")
                r["blank_class_ids"]=[]
                image["gt_blank_class_ids"]=[]
                for id_num in range(len(r["class_ids"])):
                    r["blank_class_ids"].append(0)
                for id_num in range(len(image["gt_class_id"])):
                    image["gt_blank_class_ids"].append(0)

                masks_shape = r["masks"].shape
                print(image['image_meta'][0])
                print(masks_shape)
                AP, precisions, recalls, overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_class_id'],
                        image['gt_mask'],
                        r['rois'],
                        r["class_ids"],
                        r["scores"],
                        np.reshape(r["masks"], [masks_shape[0], masks_shape[1], 1, masks_shape[2]]),
                        iou_threshold=0.5,
                    )
                AP_list.append(AP)
                precision_list.append(precisions)
                recall_list.append(recalls)

                classless_AP, classless_precisions, classless_recalls, classless_overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        np.array(image['gt_blank_class_ids']),
                        image['gt_mask'],
                        r['rois'],
                        np.array(r["blank_class_ids"]),
                        r["scores"],
                        np.reshape(r["masks"], [masks_shape[0], masks_shape[1], 1, masks_shape[2]]),
                        iou_threshold=0.5,
                    )
                classless_AP_list.append(classless_AP)
                classless_precision_list.append(classless_precisions)
                classless_recall_list.append(classless_recalls)

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
                        iou_threshold=0.5,
                    )
                    classless_AP, classless_precision, classless_recalls, classless_overlaps = utils.compute_ap(
                        image['gt_bbox'],
                        image['gt_blank_class_ids'],
                        image['gt_mask'],
                        np.array([roi]),
                        np.array([r["blank_class_ids"][i]]),
                        np.array([r["scores"][i]]),
                        np.array([singleton_masks]),
                        iou_threshold=0.5,
                    )

                    match_num = np.argmax(overlaps)
                    predicted_class_list.append(r["class_ids"][i])
                    gt_class_list.append(image['gt_class_id'][match_num])
                    predicted_size_list.append(mask_size)
                    gt_size_list.append(np.sum(image['gt_mask'][match_num].flatten()))
                    overlaps_list.append(np.max(overlaps))
                    classless_overlaps_list.append(np.max(classless_overlaps))


    logging.debug(f"found {len(overlaps_list)} overlap values")
    logging.debug(f"finished all {iterations} iterations for these images")
    return AP_list, classless_AP_list, precision_list, classless_precision_list, recall_list, classless_recall_list, predicted_class_list, gt_class_list, predicted_size_list, gt_size_list, overlaps_list, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels




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
    # DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    # Must call before using the dataset
    dataset.prepare()

    print(dataset)
    logging.debug("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # with tf.device(DEVICE):
    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config).as_default():
        model = modellib.MaskRCNN(mode="inference", model_dir="./log", config=config)

        # Load weights
        logging.debug("Loading weights "+ str(weights_filepath))
        model.load_weights(weights_filepath, by_name=True)
        image_ids = dataset.image_ids
        AP_list, classless_AP_list, precision_list, classless_precision_list, recall_list, classless_recall_list, predicted_class_list, gt_class_list, predicted_size_list, gt_size_list, overlaps_list, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels = compute_both_batch_aps(image_ids, dataset, model, config)

    return AP_list, classless_AP_list, precision_list, classless_precision_list, recall_list, classless_recall_list, predicted_class_list, gt_class_list, predicted_size_list, gt_size_list, overlaps_list, classless_overlaps_list, total_predicted_pixels, total_groundtruth_pixels, total_overlapping_pixels



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
    if directory.split("/")[-1].split("-")[1] == "d":
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

    print("got experiment info from directory")
    print(experiment)
    return experiment

def add_single_experiment(directory, df_filepath, datasets):
    csv_filenames = [
            f for f in os.listdir(directory) if f[0:5] == "resul" and f[-4:] == ".csv"
    ]
    print(list(os.walk(directory)))
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
        number = int(filename.split("-")[-1].split(".")[0])
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
            # DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

            # Inspect the model in training or inference modes
            # values: 'inference' or 'training'
            # TODO: code for 'training' test mode not ready yet
            TEST_MODE = "inference"

            # Must call before using the dataset
            

            # with tf.device(DEVICE):
            tf_config=tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            with tf.Session(config=tf_config).as_default():
                model = modellib.MaskRCNN(mode="inference", model_dir="./log", config=config)

                # Load weights
                logging.debug("Loading weights "+ str(weights_file))
                model.load_weights(weights_file, by_name=True)
                image_ids = datasets[0].image_ids

                tracker = SummaryTracker()
                logging.debug("getting stats")
                logging.debug("tunasand stats")

                experiment["AP_list"], \
                experiment["classless_AP_list"], \
                experiment["precision_list"], \
                experiment["classless_precision_list"], \
                experiment["recall_list"], \
                experiment["classless_recall_list"], \
                experiment["predicted_class_list"], \
                experiment["gt_class_list"], \
                experiment["predicted_size_list"], \
                experiment["gt_size_list"], \
                experiment["overlaps"], \
                experiment["classless_overlaps_list"], \
                experiment["total_predicted_pixels"], \
                experiment["total_groundtruth_pixels"], \
                experiment["total_overlapping_pixels"] = compute_both_batch_aps(image_ids, datasets[0], model, config) #get_stats(weights_file, datasets[0])

                objgraph.show_most_common_types()
                roots = objgraph.get_leaking_objects()
                print(len(roots))
                tracker.print_diff()

                for i, dataset in enumerate(["AE_area1", "AE_area2", "AE_area3"]):
                    image_ids = datasets[i+1].image_ids
                    logging.debug(f"aestats, {dataset}")
                    experiment[f"AP_list_{dataset}"], \
                    experiment[f"classless_AP_list_{dataset}"], \
                    experiment[f"precision_list_{dataset}"], \
                    experiment[f"classless_precision_list_{dataset}"], \
                    experiment[f"recall_list_{dataset}"], \
                    experiment[f"classless_recall_list_{dataset}"], \
                    experiment[f"predicted_class_list_{dataset}"], \
                    experiment[f"gt_class_list_{dataset}"], \
                    experiment[f"predicted_size_list_{dataset}"], \
                    experiment[f"gt_size_list_{dataset}"], \
                    experiment[f"overlaps_{dataset}"], \
                    experiment[f"classless_overlaps_list_{dataset}"], \
                    experiment[f"total_predicted_pixels_{dataset}"], \
                    experiment[f"total_groundtruth_pixels_{dataset}"], \
                    experiment[f"total_overlapping_pixels_{dataset}"]  = compute_both_batch_aps(image_ids, datasets[i+1], model, config) #get_stats(weights_file, datasets[i+1])
                    objgraph.show_growth()
                    roots = objgraph.get_leaking_objects()
                    print(len(roots))  
                    tracker.print_diff()

            
            update_dataframe(df_filepath, experiment)
        else:
            print("already in dataframe, skipping "+filename)



def create_dataframe(number=None, outfile="", folder="logs"):
    print("LOADED IMPORTS - NOW RUNNING CODE")
    df_filepath = f"./experiments_dataframe_{outfile}.csv"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',filename='log'+str(number)+'.log',level=logging.DEBUG)
    experiments = []
    files = [x for x in os.walk(f"./{folder}/")][0][1]
    directories = [f"./{folder}/" + f for f in files]
    
    print(directories)
    for directory in directories:
        experiment=directory_to_experiment_info(directory)
        dataset_filepath = get_dataset_filepath(experiment)
        ae_dataset_filepaths = get_ae2000_dataset_filepaths(experiment)
        datasets=[dataset_filepath]#load_dataset(ae_dataset_filepath_dive1), load_dataset(ae_dataset_filepath_dive2), load_dataset(ae_dataset_filepath_dive3)]
        datasets.extend(ae_dataset_filepaths)
        datasets = [load_dataset(d) for d in datasets]
        add_single_experiment(directory, df_filepath, datasets)
    return

def max_overlaps(experiments):
    for experiment in experiments:
         new_overlaps=[]
         print(experiment['overlaps'])

def experiment_in_dataframe(df_filepath, experiment):
    if not os.path.exists(df_filepath):
        return False
    df = pd.read_csv(df_filepath)
    pre_existing_experiment = df.loc[
            (df["colour_correction_type"] == experiment["colour_correction_type"])
            & (df["distortion_correction"] == experiment["distortion_correction"])
            & (df["rescaled"] == experiment["rescaled"])
            & (df["number"] == experiment["number"])
        ]
    print(pre_existing_experiment)
    return not pre_existing_experiment.empty


def update_dataframe(df_filepath, experiment):
    if not os.path.exists(df_filepath):
        df = pd.DataFrame(columns=experiment.keys())
    else:
        df = pd.read_csv(df_filepath, index_col=0)
    df = df.append(experiment, ignore_index=True)
    logging.debug("saving experiment to "+df_filepath)
    df.to_csv(df_filepath)


def get_experiments():
    df = pd.read_csv("./experiments_dataframe.csv")
    return df


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
    parser.add_argument('outfile', type=str, help='output filename')
    parser.add_argument('logs_folder', type=str, default='logs')
    args = parser.parse_args()
    number = args.array_id
    outfile = args.outfile
    logs_folder = args.logs_folder
    create_dataframe(number=number, outfile=outfile, folder=logs_folder)
