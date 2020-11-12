import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
import os
import math
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from itertools import chain
import logging
from matplotlib.patches import Patch


def get_experiments():
    experiments = []
    log_files = [f for f in os.listdir('./logs/csv_files') if f[-4:] == ".csv"]
    for log_file in log_files:
        colour_correction_type = directory.split("-")[0]

        if directory.split("-")[1] == "d":
            distortion_correction = True
        else:
            distortion_correction = False

        rescaled = directory.split("-")[2]

        filenames = [f for f in os.listdir(directory) if f[-4:] == ".csv"]
        print(filenames)

        experiment = {
            "colour_correction_type": colour_correction_type,
            "distortion_correction": distortion_correction,
            "rescaled": rescaled,
        }
        with open("./" + directory + "/" + filename, "r") as csvfile:
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

        print(experiment)
        experiments.append(experiment)

    return experiments


def size_class_overlaps(size_overlaps_string, class_overlaps_string):
    size_list = []
    overlaps_list = []
    class_list = []
    split_size_overlaps = size_overlaps_string[1:-1].split("dtype=float32)],")
    split_class_overlaps = class_overlaps_string[1:-1].split("dtype=float32)],")
    for item in split_size_overlaps:
        split = item.split(": ")
        size = int(split[0])
        overlaps = []
        split_overlaps = split[1][1:-1].split("array(")[1:]
        for overlaps_item in split_overlaps:
            numbers = [float(x) for x in overlaps_item.split("]]")[0][2:].split(",")]
            overlaps.append(max(numbers))
        size_list.append(size)
        overlaps_list.append(np.mean(overlaps))
    for i, item in enumerate(split_class_overlaps):
        split = item.split(": ")
        class_id = split[0]
        overlaps = []
        split_overlaps = split[1][1:-1].split("array(")[1:]
        for overlaps_item in split_overlaps:
            numbers = [float(x) for x in overlaps_item.split("]]")[0][2:].split(",")]
            overlaps.append(max(numbers))
        class_list.append(class_id)
        print(np.mean(overlaps)==overlaps_list[i])
    return size_list, overlaps_list, class_list


def size_overlaps(size_overlaps_string):
    size_list = []
    overlaps_list = []
    split_size_overlaps = size_overlaps_string[1:-1].split("dtype=float32)],")
    for item in split_size_overlaps:
        split = item.split(": ")
        size = int(split[0])
        overlaps = []
        split_overlaps = split[1][1:-1].split("array(")[1:]
        for overlaps_item in split_overlaps:
            numbers = [float(x) for x in overlaps_item.split("]]")[0][2:].split(",")]
            overlaps.append(max(numbers))
        size_list.append(size)
        overlaps_list.append(np.mean(overlaps))
        # numbers =[float(x) for x in item.split("]]")[0][2:].split(',')]
        # size_overlaps_list.append(max(numbers))
    print(size_list)
    print(overlaps_list)
    return size_list, overlaps_list


def overlaps(overlaps_string):
    overlaps_list = []
    split_overlaps = overlaps_string[1:-1].split("array(")[1:]
    for item in split_overlaps:
        numbers = [float(x) for x in item.split("]]")[0][2:].split(",")]
        overlaps_list.append(max(numbers))
    return np.mean(overlaps_list)


def plot_boxplots(experiments):
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, 4.6])
    fig, ax = plt.subplots()

    new_experiments = []
    print(experiments.keys())
    for index, experiment in experiments.iterrows():
        print(experiment)
        experiment["val_mrcnn_class_loss"] = experiment["val_mrcnn_class_loss"][
            1:-1
        ].split(", ")
        experiment["val_loss"] = experiment["val_loss"][1:-1].split(", ")
        experiment["overlaps"] = overlaps(experiment["overlaps"])
        # experiment["ae_overlaps_dive1"] = overlaps(experiment["ae_overlaps_dive1"])
        # experiment["ae_overlaps_dive2"] = overlaps(experiment["ae_overlaps_dive2"])
        # experiment["ae_overlaps_dive3"] = overlaps(experiment["ae_overlaps_dive3"])
        experiment["size_overlaps"] = size_overlaps(experiment["size_overlaps"])
        print(len(experiment["val_mrcnn_class_loss"]))
        print(len(experiment["val_loss"]))
        new_row = {
            "rescaled": experiment["rescaled"],
            "distortion_correction": experiment["distortion_correction"],
            "minimum_loss": experiment["minimum_loss"],
            "number": experiment["number"],
            "minimum_val_loss": experiment["minimum_val_loss"],
            "elastic_distortions": experiment["elastic_distortions"],
            "colour_correction_type": experiment["colour_correction_type"],
            "separate_channel_ops": experiment["separate_channel_ops"],
            "combo": experiment["colour_correction_type"]
            + str(experiment["distortion_correction"])
            + str(experiment["rescaled"]),
            "overlaps": experiment["overlaps"],
            "dataset": "tunasand",
        }
        new_experiments.append(new_row)
        # new_row = {
        #     "rescaled": experiment["rescaled"],
        #     "distortion_correction": experiment["distortion_correction"],
        #     "minimum_loss": experiment["minimum_loss"],
        #     "number": experiment["number"],
        #     "minimum_val_loss": experiment["minimum_val_loss"],
        #     "elastic_distortions": experiment["elastic_distortions"],
        #     "colour_correction_type": experiment["colour_correction_type"],
        #     "separate_channel_ops": experiment["separate_channel_ops"],
        #     "combo": experiment["colour_correction_type"]
        #     + str(experiment["distortion_correction"])
        #     + str(experiment["rescaled"]),
        #     "overlaps": experiment["ae_overlaps_dive1"],
        #     "dataset": "ae_dive1",
        # }
        # new_experiments.append(new_row)
        # new_row = {
        #     "rescaled": experiment["rescaled"],
        #     "distortion_correction": experiment["distortion_correction"],
        #     "minimum_loss": experiment["minimum_loss"],
        #     "number": experiment["number"],
        #     "minimum_val_loss": experiment["minimum_val_loss"],
        #     "elastic_distortions": experiment["elastic_distortions"],
        #     "colour_correction_type": experiment["colour_correction_type"],
        #     "separate_channel_ops": experiment["separate_channel_ops"],
        #     "combo": experiment["colour_correction_type"]
        #     + str(experiment["distortion_correction"])
        #     + str(experiment["rescaled"]),
        #     "overlaps": experiment["ae_overlaps_dive2"],
        #     "dataset": "ae_dive2",
        # }
        # new_experiments.append(new_row)
        # new_row = {
        #     "rescaled": experiment["rescaled"],
        #     "distortion_correction": experiment["distortion_correction"],
        #     "minimum_loss": experiment["minimum_loss"],
        #     "number": experiment["number"],
        #     "minimum_val_loss": experiment["minimum_val_loss"],
        #     "elastic_distortions": experiment["elastic_distortions"],
        #     "colour_correction_type": experiment["colour_correction_type"],
        #     "separate_channel_ops": experiment["separate_channel_ops"],
        #     "combo": experiment["colour_correction_type"]
        #     + str(experiment["distortion_correction"])
        #     + str(experiment["rescaled"]),
        #     "overlaps": experiment["ae_overlaps_dive3"],
        #     "dataset": "ae_dive3",
        # }
        # new_experiments.append(new_row)

    experiments_dataframe = pd.DataFrame(new_experiments)
    print(experiments_dataframe)

    g = sns.boxplot(
        x=experiments_dataframe["dataset"],
        y=experiments_dataframe["overlaps"],
        hue=experiments_dataframe["rescaled"],
        palette="deep",
    )

    # g.get_legend().remove()
    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.title("Rescaling")
    # plt.xlabel('Epoch')
    plt.ylabel("Mean overlap")

    plt.savefig("rescaling_boxplot")


def group_experiments_by(experiments, category):
    category_values = []
    for i, experiment in experiments.iterrows():
        if not experiment[category] in category_values:
            category_values.append(experiment[category])
            # print(experiments.where(experiments[category]==category_values[0]))
    new_experiments = []
    for category_value in category_values:
        category_experiments_indices = experiments[category] == category_value
        category_experiments = experiments[category_experiments_indices]
        new_experiment = {
            category: category_value,
            "val_mrcnn_class_loss": np.mean(
                np.array(list(category_experiments["val_mrcnn_class_loss"])), axis=0
            ),
            "val_mrcnn_bbox_loss": np.mean(
                np.array(list(category_experiments["val_mrcnn_bbox_loss"])), axis=0
            ),
            "val_mrcnn_mask_loss": np.mean(
                np.array(list(category_experiments["val_mrcnn_mask_loss"])), axis=0
            ),
            "val_rpn_bbox_loss": np.mean(
                np.array(list(category_experiments["val_rpn_bbox_loss"])), axis=0
            ),
            "val_rpn_class_loss": np.mean(
                np.array(list(category_experiments["val_rpn_class_loss"])), axis=0
            ),
            "mrcnn_class_loss": np.mean(
                np.array(list(category_experiments["mrcnn_class_loss"])), axis=0
            ),
            "mrcnn_bbox_loss": np.mean(
                np.array(list(category_experiments["mrcnn_bbox_loss"])), axis=0
            ),
            "mrcnn_mask_loss": np.mean(
                np.array(list(category_experiments["mrcnn_mask_loss"])), axis=0
            ),
            "rpn_bbox_loss": np.mean(
                np.array(list(category_experiments["rpn_bbox_loss"])), axis=0
            ),
            "rpn_class_loss": np.mean(
                np.array(list(category_experiments["rpn_class_loss"])), axis=0
            ),
        }
        if "size_list" in category_experiments.keys():
            new_experiment["size_list"] = []
            new_experiment["overlaps_list"] = []
            for item in category_experiments["size_list"]:
                new_experiment["size_list"].extend(item)
            for item in category_experiments["overlaps_list"]:
                new_experiment["overlaps_list"].extend(item)
        new_experiments.append(new_experiment)

    experiments_dataframe = pd.DataFrame(new_experiments)
    return experiments_dataframe


def plot_colour_correction_lossvsvalloss(experiments):

    new_experiments = []
    print(experiments.keys())
    for index, experiment in experiments.iterrows():
        print(experiment)
        experiment["val_mrcnn_class_loss"] = [
            float(n) for n in experiment["val_mrcnn_class_loss"][1:-1].split(", ")
        ]
        experiment["val_mrcnn_bbox_loss"] = [
            float(n) for n in experiment["val_mrcnn_bbox_loss"][1:-1].split(", ")
        ]
        experiment["val_mrcnn_mask_loss"] = [
            float(n) for n in experiment["val_mrcnn_mask_loss"][1:-1].split(", ")
        ]
        experiment["mrcnn_bbox_loss"] = [
            float(n) for n in experiment["mrcnn_bbox_loss"][1:-1].split(", ")
        ]
        experiment["val_rpn_bbox_loss"] = [
            float(n) for n in experiment["val_rpn_bbox_loss"][1:-1].split(", ")
        ]
        experiment["mrcnn_mask_loss"] = [
            float(n) for n in experiment["mrcnn_mask_loss"][1:-1].split(", ")
        ]
        experiment["rpn_class_loss"] = [
            float(n) for n in experiment["rpn_class_loss"][1:-1].split(", ")
        ]
        experiment["rpn_bbox_loss"] = [
            float(n) for n in experiment["rpn_bbox_loss"][1:-1].split(", ")
        ]
        experiment["val_loss"] = [
            float(n) for n in experiment["val_loss"][1:-1].split(", ")
        ]
        experiment["mrcnn_class_loss"] = [
            float(n) for n in experiment["mrcnn_class_loss"][1:-1].split(", ")
        ]
        experiment["val_rpn_class_loss"] = [
            float(n) for n in experiment["val_rpn_class_loss"][1:-1].split(", ")
        ]
        experiment["loss"] = [float(n) for n in experiment["loss"][1:-1].split(", ")]
        experiment["overlaps"] = overlaps(experiment["overlaps"])
        for epoch in range(len(experiment["val_loss"])):
            new_row = {
                "rescaled": experiment["rescaled"],
                "distortion_correction": experiment["distortion_correction"],
                "minimum_loss": experiment["minimum_loss"],
                "number": experiment["number"],
                "minimum_val_loss": experiment["minimum_val_loss"],
                "elastic_distortions": experiment["elastic_distortions"],
                "colour_correction_type": experiment["colour_correction_type"],
                "separate_channel_ops": experiment["separate_channel_ops"],
                "epoch": epoch,
                "mrcnn_mask_loss": float(experiment["mrcnn_mask_loss"][epoch]),
                "val_mrcnn_bbox_loss": float(experiment["val_mrcnn_bbox_loss"][epoch]),
                "loss": float(experiment["loss"][epoch]),
                "val_mrcnn_class_loss": float(
                    experiment["val_mrcnn_class_loss"][epoch]
                ),
                "val_mrcnn_mask_loss": float(experiment["val_mrcnn_mask_loss"][epoch]),
                "mrcnn_bbox_loss": float(experiment["mrcnn_bbox_loss"][epoch]),
                "val_rpn_bbox_loss": float(experiment["val_rpn_bbox_loss"][epoch]),
                "rpn_class_loss": float(experiment["rpn_class_loss"][epoch]),
                "rpn_bbox_loss": float(experiment["rpn_bbox_loss"][epoch]),
                "mrcnn_class_loss": float(experiment["mrcnn_class_loss"][epoch]),
                "val_rpn_class_loss": float(experiment["val_rpn_class_loss"][epoch]),
                "val_loss": float(experiment["val_loss"][epoch]),
                "combo": experiment["colour_correction_type"]
                + str(experiment["distortion_correction"])
                + str(experiment["rescaled"]),
            }
            new_experiments.append(new_row)

    experiments_dataframe = pd.DataFrame(new_experiments)
    sns.set()
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, 4.6])
    fig, ax = plt.subplots()

    sns.lineplot(x="epoch", y="val_loss", data=experiments_dataframe, hue="combo")
    print(len(ax.lines))
    for i in range(len(ax.lines)):
        ax.lines[i].set_linestyle("--")
    g = sns.lineplot(x="epoch", y="loss", data=experiments_dataframe, hue="combo")
    g.legend_.remove()
    # ax.lines[3].set_linestyle("--")
    print(ax.lines)
    print(ax.lines[3].get_linestyle())
    # plt.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # custom_lines = [Patch(facecolor=ax.lines[0].get_color()),
    #                 Patch(facecolor=ax.lines[1].get_color()),
    #                 Patch(facecolor=ax.lines[2].get_color()),
    #                 # Patch(facecolor=ax.lines[3].get_color()),
    #                 ]
    # handles=custom_lines
    # labels = labels[1:4]
    # handles.extend([ax.lines[4], ax.lines[0]])

    # labels.extend(['training', 'validation'])
    # print(handles)
    # print(labels)
    # ax.legend(handles=[], labels=[])
    plt.title("Loss and validation loss per epoch")
    plt.savefig("combo_type_loss_vs_valloss")


def stringlist_to_list(stringlist):
    real_list = [float(n) for n in stringlist[1:-1].split(", ")]
    return real_list


def plot_size_overlap_scatter(experiments):
    new_experiments = []
    print(experiments.keys())
    for index, experiment in experiments.iterrows():
        print(experiment)
        for value in [
            "val_mrcnn_class_loss",
            "val_mrcnn_bbox_loss",
            "val_mrcnn_mask_loss",
            "mrcnn_bbox_loss",
            "val_rpn_bbox_loss",
            "mrcnn_mask_loss",
            "rpn_class_loss",
            "rpn_bbox_loss",
            "val_loss",
            "mrcnn_class_loss",
            "val_rpn_class_loss",
            "loss",
        ]:
            experiment[value] = stringlist_to_list(experiment[value])
        experiment["overlaps"] = overlaps(experiment["overlaps"])
        # experiment["ae_overlaps_dive1"] = overlaps(experiment["ae_overlaps_dive1"])
        # experiment["ae_overlaps_dive2"] = overlaps(experiment["ae_overlaps_dive2"])
        # experiment["ae_overlaps_dive3"] = overlaps(experiment["ae_overlaps_dive3"])
        experiment["size_list"], experiment["overlaps_list"], experiment["class_list"] = size_class_overlaps(experiment["size_overlaps"], experiment["class_overlaps"])
        # experiment["ae_size_list_dive1"], experiment["ae_overlaps_list_dive1"], experiment["ae_class_list_dive1"] = size_class_overlaps(experiment["ae_size_overlaps_dive1"], experiment["ae_class_overlaps_dive1"])
        # experiment["ae_size_list_dive2"], experiment["ae_overlaps_list_dive2"], experiment["ae_class_list_dive2"] = size_class_overlaps(experiment["ae_size_overlaps_dive2"], experiment["ae_class_overlaps_dive2"])
        # experiment["ae_size_list_dive3"], experiment["ae_overlaps_list_dive3"], experiment["ae_class_list_dive3"] = size_class_overlaps(experiment["ae_size_overlaps_dive3"], experiment["ae_class_overlaps_dive3"])
        # new_experiments.append(experiment)

    experiments_dataframe = pd.DataFrame(new_experiments)
    experiments_dataframe = group_experiments_by(experiments_dataframe, "rescaled")

    sns.set()
    for i, experiment in experiments_dataframe.iterrows():
        data=pd.DataFrame([experiment["size_list"], experiment["overlaps_list"], experiment["class_list"]])
        sns.scatterplot(x="size_list", y="overlaps_list", hue="class_list", data=data)
        # x = experiment["size_list"]
        # y = experiment["overlaps_list"]
        # plt.scatter(x, y, marker=".", s=1, label=experiment["rescaled"])
        axes = plt.gca()
        axes.set_xlim([0, 30000])
        axes.set_ylim([-0.01, 1])
        axes.set_ylabel("IOU scores")
        axes.set_xlabel("Size of groundtruth segment in pixels")
        plt.title("IOU to size of groundtruth scatterplot - " + experiment["rescaled"])
        plt.savefig(experiment["rescaled"] + "_scatter_plot")
        plt.clf()

    for i, experiment in experiments_dataframe.iterrows():
        data=pd.DataFrame([experiment["size_list"], experiment["overlaps_list"], experiment["class_list"]])
        sns.scatterplot(x="size_list", y="overlaps_list", hue="class_list", data=data)
        # x = experiment["size_list"]
        # y = experiment["overlaps_list"]
        # plt.scatter(x, y, marker=".", s=1, label=experiment["rescaled"])
    axes = plt.gca()
    axes.set_xlim([0, 30000])
    axes.set_ylim([-0.01, 1])
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    plt.legend()
    plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    plt.savefig("all_rescaling_scatter_plot")
    plt.clf()


def plot_colour_correction_stackedarea(experiments):
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, 4.6])
    fig, ax = plt.subplots()

    new_experiments = []
    print(experiments.keys())
    for index, experiment in experiments.iterrows():
        print(experiment)
        for value in [
            "val_mrcnn_class_loss",
            "val_mrcnn_bbox_loss",
            "val_mrcnn_mask_loss",
            "mrcnn_bbox_loss",
            "val_rpn_bbox_loss",
            "mrcnn_mask_loss",
            "rpn_class_loss",
            "rpn_bbox_loss",
            "val_loss",
            "mrcnn_class_loss",
            "val_rpn_class_loss",
            "loss",
        ]:
            experiment[value] = stringlist_to_list(experiment[value])
        experiment["overlaps"] = overlaps(experiment["overlaps"])

        new_experiments.append(experiment)

    experiments_dataframe = pd.DataFrame(new_experiments)
    experiments_dataframe = group_experiments_by(experiments_dataframe, "rescaled")

    sns.set(rc={"lines.linewidth": 0.3})
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, 4.6])
    fig, ax = plt.subplots()

    for i, experiment in experiments_dataframe.iterrows():
        y = [
            experiment["rpn_bbox_loss"],
            experiment["rpn_class_loss"],
            experiment["mrcnn_bbox_loss"],
            experiment["mrcnn_mask_loss"],
            experiment["mrcnn_class_loss"],
        ]
        labels = [
            "rpn_bbox_loss",
            "rpn_class_loss",
            "mrcnn_bbox_loss",
            "mrcnn_mask_loss",
            "mrcnn_class_loss",
        ]
        x = range(1, 101)
        axes = plt.gca()
        axes.set_ylim([0, 4.6])
        axes.set_ylabel("Training Epoch")
        axes.set_xlabel("Training Loss")
        plt.stackplot(x, y, labels=labels)
        plt.legend()
        plt.title(experiment["rescaled"])
        plt.savefig(experiment["rescaled"] + "_stacked_loss")
        plt.clf()
        axes = plt.gca()
        axes.set_ylim([0, 4.6])
        y = [
            experiment["val_rpn_bbox_loss"],
            experiment["val_rpn_class_loss"],
            experiment["val_mrcnn_bbox_loss"],
            experiment["val_mrcnn_mask_loss"],
            experiment["val_mrcnn_class_loss"],
        ]
        labels = [
            "val_rpn_bbox_loss",
            "val_rpn_class_loss",
            "val_mrcnn_bbox_loss",
            "val_mrcnn_mask_loss",
            "val_mrcnn_class_loss",
        ]
        plt.stackplot(x, y, labels=labels)
        axes.set_ylabel("Training Epoch")
        axes.set_xlabel("Validation Loss")
        plt.legend()
        plt.title(experiment["rescaled"])
        plt.savefig(experiment["rescaled"] + "_stacked_val_loss")
        plt.clf()


experiments = pd.read_csv("./experiments_dataframe.csv")
print(experiments)
plot_boxplots(experiments)
plot_colour_correction_stackedarea(experiments)
plot_colour_correction_lossvsvalloss(experiments)
plot_size_overlap_scatter(experiments)
