import os
import FK2018
import csv
import math
import tensorflow as tf
import mrcnn.model as modellib
import pandas as pd
from mrcnn import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NUMBER = "0"
DF_FILEPATH = "./dataframe_0.csv"


def get_dataset_filepath(experiment):
    filepath = "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/"
    if experiment["colour_correction_type"] == "histogram_normalised":
        filepath += "histogram_normalised/"
    elif experiment["colour_correction_type"] == "greyworld":
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

    filepath += experiment["rescaled"] + "/"

    filepath += "val"
    return filepath


def get_ae2000_dataset_filepaths(experiment):
    filepath = "/Volumes/jw22g14_phd/fk2018/ae2000/ae2000_overlap/coco/"
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


def compute_batch_ap(image_ids, model, dataset, config):
    import numpy

    APs = []
    class_APs = {}
    size_APs = {}
    class_overlaps = {}
    size_overlaps = {}
    overlaps_list = []
    for iteration in range(15):
        for image_id in image_ids:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset, config, image_id, use_mini_mask=False
            )
            if len(gt_class_id) > 0:

                # Run object detection
                results = model.detect([image], verbose=0)
                # Compute AP
                r = results[0]
                for i, roi in enumerate(r["rois"]):
                    mask_size = numpy.sum(r["masks"][:, :, i].flatten())
                    singleton_rois = np.array([roi])
                    singleton_class_ids = np.array([r["class_ids"][i]])
                    singleton_scores = np.array([r["scores"][i]])
                    singleton_masks = np.zeros([1024, 1024, 1])
                    singleton_masks[:, :, 0] = np.array(r["masks"][:, :, i])
                    AP, precisions, recalls, overlaps = utils.compute_ap(
                        gt_bbox,
                        gt_class_id,
                        gt_mask,
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

    return class_APs, size_APs, APs, class_overlaps, size_overlaps, overlaps_list


def get_stats(weights_filepath, dataset_filepath):
    config = FK2018.FKConfig()

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

    dataset = FK2018.FKDataset()
    dataset.load_fk(dataset_filepath)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir="./log", config=config)

        # Load weights
    print("Loading weights ", weights_filepath)
    model.load_weights(weights_filepath, by_name=True)

    image_ids = dataset.image_ids
    class_APs, size_APs, APs, class_overlaps, size_overlaps, overlaps = compute_batch_ap(
        image_ids, model, dataset, config
    )
    return class_APs, size_APs, APs, class_overlaps, size_overlaps, overlaps


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


def populate_experiments_dataframe():
    experiments = []
    files = [x for x in os.walk("../../../paper_experiments/weights/")][0][1]
    print(files)
    directories = ["../../../paper_experiments/weights/" + f for f in files]
    for directory in directories:
        csv_filenames = [
            f for f in os.listdir(directory) if f[0:5] == "resul" and f[-4:] == ".csv"
        ]
        weights_folder = [f for f in os.walk(directory)][0][1][0]
        for filename in csv_filenames:
            experiment = directory_to_experiment_info(directory)
            with open("./" + directory + "/" + filename, "r") as csvfile:
                print("./" + directory + "/" + filename)
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

            if not experiment_in_dataframe(DF_FILEPATH, experiment):
                weights_file = (
                    directory + "/" + weights_folder + "/" + "mask_rcnn_fk2018_best.h5"
                )
                dataset_filepath = get_dataset_filepath(experiment)
                ae_dataset_filepath_dive1, ae_dataset_filepath_dive2, ae_dataset_filepath_dive3 = get_ae2000_dataset_filepaths(
                    experiment
                )
                if os.path.exists(weights_file):
                    # try:
                    experiment["class_APs"], experiment["size_APs"], experiment[
                        "APs"
                    ], experiment["class_overlaps"], experiment[
                        "size_overlaps"
                    ], experiment[
                        "overlaps"
                    ] = get_stats(
                        weights_file, dataset_filepath
                    )

                    experiment["ae_class_APs_dive1"], experiment[
                        "ae_size_APs_dive1"
                    ], experiment["ae_APs_dive1"], experiment[
                        "ae_class_overlaps_dive1"
                    ], experiment[
                        "ae_size_overlaps_dive1"
                    ], experiment[
                        "ae_overlaps_dive1"
                    ] = get_stats(
                        weights_file, ae_dataset_filepath_dive1
                    )

                    experiment["ae_class_APs_dive2"], experiment[
                        "ae_size_APs_dive2"
                    ], experiment["ae_APs_dive2"], experiment[
                        "ae_class_overlaps_dive2"
                    ], experiment[
                        "ae_size_overlaps_dive2"
                    ], experiment[
                        "ae_overlaps_dive2"
                    ] = get_stats(
                        weights_file, ae_dataset_filepath_dive2
                    )

                    experiment["ae_class_APs_dive3"], experiment[
                        "ae_size_APs_dive3"
                    ], experiment["ae_APs_dive3"], experiment[
                        "ae_class_overlaps_dive3"
                    ], experiment[
                        "ae_size_overlaps_dive3"
                    ], experiment[
                        "ae_overlaps_dive3"
                    ] = get_stats(
                        weights_file, ae_dataset_filepath_dive3
                    )
                    # except:
                    #     print("issue getting loss values")
                else:
                    print("weights file doesn't exist")
                    print(weights_file)

                update_dataframe(DF_FILEPATH, experiment)
            else:
                print("already in dataframe, skipping " + filename)

    return pd.DataFrame(experiments)


def add_aestats_to_dataframe():
    df = pd.read_csv("./experiments_dataframe.csv")
    new_experiments = []
    files = [x for x in os.walk("../../../paper_experiments/weights/")][0][1]
    directories = ["../../../paper_experiments/weights/" + f for f in files]
    for directory in directories:
        experiment = directory_to_experiment_info(directory)
        dataset_filepath_dive1, dataset_filepath_dive2, dataset_filepath_dive3 = get_ae2000_dataset_filepaths(
            experiment
        )
        weights_folder = [f for f in os.walk(directory)][0][1][0]
        weights_filepath = (
            directory + "/" + weights_folder + "/mask_rcnn_fk2018_best.h5"
        )
        experiment = df.loc[
            (df["colour_correction_type"] == experiment["colour_correction_type"])
            & (df["distortion_correction"] == experiment["distortion_correction"])
            & (df["rescaled"] == experiment["rescaled"])
            & (df["number"] == experiment["number"])
        ]
        print(experiment)
        experiment = experiment.to_dict()
        experiment["ae_class_APs_dive1"], experiment["ae_size_APs_dive1"], experiment[
            "ae_APs_dive1"
        ] = get_ae_stats(weights_filepath, dataset_filepath_dive1)
        experiment["ae_class_APs_dive2"], experiment["ae_size_APs_dive2"], experiment[
            "ae_APs_dive2"
        ] = get_ae_stats(weights_filepath, dataset_filepath_dive2)
        experiment["ae_class_APs_dive3"], experiment["ae_size_APs_dive3"], experiment[
            "ae_APs_dive3"
        ] = get_ae_stats(weights_filepath, dataset_filepath_dive3)
        print(experiment)
        new_experiments.append(experiment)
    df = pd.DataFrame(new_experiments)
    df.to_csv("./new_experiments_dataframe.csv")


def experiment_in_dataframe(df_filepath, experiment):
    df = pd.read_csv(df_filepath, index_col=0)
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
    print("saving experiment to ./newest_experiments_dataframe.csv")
    df.to_csv("./newest_experiments_dataframe.csv")


def get_experiments():
    df = pd.read_csv("./experiments_dataframe.csv")
    return df


def create_dataframe():
    df = populate_experiments_dataframe()


if __name__ == "__main__":
    create_dataframe()
