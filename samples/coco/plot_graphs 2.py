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
import glob
from pandas.plotting import table

import matplotlib.ticker as ticker



experiment_level_graph_settings=[
                    ["rescaled", "deep", ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"],["Original Scale", "Average Low\n Altitude Scale", "Average High\n Altitude Scale", "Average Low\n Altitude Scale\n with High\n Altitude Resolution"]],
                    ["distortion_correction", "Blues", [False, True], ["Original Image", "Distortion Corrected"]], 
                    ["colour_correction_type", "Reds", ["grey", "alt"], ["Greyworld Corrected", "Altitude Corrected"]], 
                    # ["separate_channel_ops", "PiYG", [False, True], ["Cross Channel Operations", "Separate Channel Operations"]],
                    # ["elastic_distortions","PRGn",[False, True], ["Without Elastic Distortions", "With Elastic Distortions"]]
                ]

instance_level_graph_settings = []
instance_level_graph_settings.extend(experiment_level_graph_settings)
instance_level_graph_settings.append(["gt_class", "Accent", [8.0, 3.0, 5.0, 2.0, 1.0, 6.0], ["Seastar", "Small\n Rockfish", "Fish 4", "Large\n Rockfish", "Eel", "Crab"]])

def get_experiments():
    experiments = []
    directories = [x for x in os.walk("../../../")][0][1]
    for directory in directories:
        colour_correction_type = directory.split("_")[0]

        if directory.split("_")[1] == "dist":
            distortion_correction = True
        else:
            distortion_correction = False

        if directory.split("_")[2] == "rescaled":
            rescaled = True
        else:
            rescaled = False

        filenames = [f for f in os.listdir(directory) if f[-4:] == ".csv"]
        print(filenames)

        for filename in filenames:
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


def int_to_string(bucket):
    return f"{bucket*500}-{(bucket+1)*500}"

def size_to_bucket(size, bucket_size):
    bucket = math.floor(size / bucket_size)
    return bucket

def size_buckets(size_list, bucket_size):
    size_bucket=[]
    for size in size_list:
        bucket = size_to_bucket(size, bucket_size)
        size_bucket.append(bucket)
    return size_bucket


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
    split_overlaps = overlaps_string[1:-1].split(",")[1:]
    for item in split_overlaps:
        numbers = [float(x) for x in split_overlaps]
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
            "overlaps_AE_area1",
            "overlaps_AE_area2",
            "overlaps_AE_area3",
            "gt_size_list",
            "overlaps",
            "gt_class_list",
            "gt_size_list_AE_area1",
            "gt_class_list_AE_area1",
            "gt_size_list_AE_area2",
            "gt_class_list_AE_area2",
            "gt_size_list_AE_area3",
            "gt_class_list_AE_area3",
            "AP_list",
            "AP_list_AE_area1",
            "AP_list_AE_area2",
            "AP_list_AE_area3",
            "classless_AP_list",
            "classless_AP_list_AE_area1",
            "classless_AP_list_AE_area2",
            "classless_AP_list_AE_area3",
        ]:
            experiment[value] = stringlist_to_list(experiment[value])
        experiment['mean_overlap'] = np.mean(experiment['overlaps'])
        experiment['0_threshold'] = len([i for i in experiment['overlaps'] if i > 0])/len(experiment['overlaps'])
        experiment['5_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.05])/len(experiment['overlaps'])
        experiment['10_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.1])/len(experiment['overlaps'])
        experiment['15_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.15])/len(experiment['overlaps'])
        experiment['20_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.2])/len(experiment['overlaps'])
        experiment['30_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.3])/len(experiment['overlaps'])
        experiment['40_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.4])/len(experiment['overlaps'])
        experiment['50_threshold'] = len([i for i in experiment['overlaps'] if i >= 0.5])/len(experiment['overlaps'])
        experiment['mean_nonzero_overlaps'] = np.mean([item for item in experiment['overlaps'] if item != 0])
        experiment['mean_AP'] = np.mean(experiment['AP_list'])
        experiment['mean_classless_AP'] = np.mean(experiment['classless_AP_list'])
        experiment['mean_classless_AP_AE_area1'] = np.mean(experiment["classless_AP_list_AE_area1"])
        experiment['mean_classless_AP_AE_area2'] = np.mean(experiment["classless_AP_list_AE_area2"])
        experiment['mean_classless_AP_AE_area3'] = np.mean(experiment["classless_AP_list_AE_area3"])
        print(experiment['rescaled'])
        print(experiment['overlaps'])
        print(experiment['30_threshold'])
        

        for ae_dataset in ["AE_area1", "AE_area2", "AE_area3"]:
            experiment[f'mean_overlap_{ae_dataset}'] = np.mean(experiment[f'overlaps_{ae_dataset}'])
            experiment[f'0_threshold_{ae_dataset}'] = len([i for i in experiment[f'overlaps_{ae_dataset}'] if i > 0])/len(experiment[f'overlaps_{ae_dataset}'])
            experiment[f'10_threshold_{ae_dataset}'] = len([i for i in experiment[f'overlaps_{ae_dataset}'] if i >= 0.1])/len(experiment[f'overlaps_{ae_dataset}'])
            experiment[f'30_threshold_{ae_dataset}'] = len([i for i in experiment[f'overlaps_{ae_dataset}'] if i >= 0.3])/len(experiment[f'overlaps_{ae_dataset}'])
            experiment[f'mean_nonzero_overlaps{ae_dataset}'] = np.mean([item for item in experiment[f'overlaps_{ae_dataset}'] if item != 0])
            print(experiment[f"AP_list_{ae_dataset}"])
            experiment[f"mean_AP_{ae_dataset}"] = np.mean(experiment[f"AP_list_{ae_dataset}"])
            experiment[f"mean_classless_AP_{ae_dataset}"] = np.mean(experiment[f"classless_AP_list_{ae_dataset}"])
        experiment['dataset'] = 'tunasand'
        new_experiments.append(experiment)

    

    print("Calculating threshold for 95% and 68% of points")
    nonzero_overlap = [item for sublist in list(experiments['overlaps']) for item in stringlist_to_list(sublist) if item != 0]
    nonzero_overlap.sort()
    threshold_95 = nonzero_overlap[int(len(nonzero_overlap)*0.05)]
    print(threshold_95)
    threshold_68 = nonzero_overlap[int(len(nonzero_overlap)*0.32)]
    print(threshold_68)
    experiments_dataframe = pd.DataFrame(new_experiments)
    sns.set()
    plt.figure(figsize=(8,5))
    mAP_summary_table(experiments_dataframe)
    include_row = experiments_dataframe.rescaled.isin(["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"])
    filtered_experiments = experiments_dataframe[include_row]
    for value in experiment_level_graph_settings:
        plt.clf()
        g = sns.boxplot(
            x=filtered_experiments[value[0]],
            y=filtered_experiments["30_threshold"],
            palette=value[1],
            order=value[2],
        )

        g.set_xticklabels(value[3], rotation=0)
        g.set_ylim([0,1])
        g.set_xlabel("")
        plt.title(value[0])
        plt.ylabel("Object Detection Rate")

        plt.savefig(f"{value[0]}_detection_boxplot", bbox_inches='tight')

        plt.clf()
        g = sns.boxplot(
            x=filtered_experiments[value[0]],
            y=filtered_experiments["mean_AP"],
            palette=value[1],
            # order=value[2],
        )

        print(value)
        g.set_xticklabels(value[3], rotation=0)
        g.set_ylim([0,1])
        g.set_xlabel("")
        plt.title(value[0])
        plt.ylabel("mAP")

        plt.savefig(f"{value[0]}_mAP_boxplot", bbox_inches='tight')

        plt.clf()
        g = sns.boxplot(
            x=filtered_experiments[value[0]],
            y=filtered_experiments["mean_classless_AP"],
            palette=value[1],
            # order=value[2],
        )

        g.set_xticklabels(value[3], rotation=0)
        g.set_ylim([0,1])
        g.set_xlabel("")
        plt.title(value[0])
        plt.ylabel("mAP")

        plt.savefig(f"{value[0]}_classless_mAP_boxplot", bbox_inches='tight')

        for ae_dataset in ["AE_area1", "AE_area2", "AE_area3"]:
            plt.clf()
            g = sns.boxplot(
                x=filtered_experiments[value[0]],
                y=filtered_experiments[f"mean_AP_{ae_dataset}"],
                palette=value[1],
                order=value[2],
            )

            g.set_xticklabels(value[3], rotation=0)
            g.set_ylim([0,1])
            g.set_xlabel("")
            plt.title(value[0])
            plt.ylabel("mAP")

            plt.savefig(f"{value[0]}_mAP_boxplot_{ae_dataset}", bbox_inches='tight')

            plt.clf()
            g = sns.boxplot(
                x=filtered_experiments[value[0]],
                y=filtered_experiments[f"mean_classless_AP_{ae_dataset}"],
                palette=value[1],
                order=value[2],
            )

            g.set_xticklabels(value[3], rotation=0)
            g.set_ylim([0,1])
            g.set_xlabel("")
            plt.title(value[0])
            plt.ylabel("mAP")

            plt.savefig(f"{value[0]}_classless_mAP_boxplot_{ae_dataset}", bbox_inches='tight')


        plt.clf()
        g = sns.boxplot(
            x=experiments_dataframe["dataset"],
            y=experiments_dataframe["mean_nonzero_overlaps"],
            hue=experiments_dataframe[value[0]],
            palette=value[1],
            hue_order=value[2],
        )

        # g.get_legend().remove()
        # g.set_xticklabels(g.get_xticklabels(), rotation=45)
        # plt.title(value[0])
        # plt.xlabel('Epoch')
        plt.ylabel("Mean Segmentation IOU Scores")

        plt.savefig(f"{value[0]}_mean_IOU_boxplot", bbox_inches='tight')
        plt.figure(figsize=(7,5))

    plt.figure(figsize=(6,5))
    for feature in experiment_level_graph_settings: #[['rescaled', ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"], ["Original Scale", "Average Low\n Altitude Scale", "Average High\n Altitude Scale", "Average Low\n Altitude Scale\n with High\n Altitude Resolution"]], ['distortion_correction', [False, True], ["Original Image", "Distortion Corrected"]]]:
        for threshold in ['0','10','30']:
            plt.clf()
            g = sns.boxplot(
                x=filtered_experiments[feature[0]],
                y=filtered_experiments[f"{threshold}_threshold"],
                # hue=experiments_dataframe["rescaled"],
                order=feature[2],
                palette=feature[1],
            )

            # g.get_legend().remove()
            g.set_xticklabels(feature[3], rotation=0)
            g.set_ylim([0,1])
            g.set_xlabel("")
            plt.title(f"Success Rate at {threshold}% IOU Threshold, {feature[0]} Comparison")
            # plt.xlabel('Epoch')
            plt.ylabel("Object Detection Rate")

            plt.savefig(f"{feature[0]}_boxplot_threshold{threshold}", bbox_inches='tight')

            for ae_dataset in ["AE_area1", "AE_area2", "AE_area3"]:
                plt.clf()
                g = sns.boxplot(
                    x=filtered_experiments[feature[0]],
                    y=filtered_experiments[f"{threshold}_threshold_{ae_dataset}"],
                    # hue=experiments_dataframe["rescaled"],
                    order=feature[2],
                    palette=feature[1],
                )

                # g.get_legend().remove()
                g.set_xticklabels(feature[3], rotation=0)
                g.set_ylim([0,1])
                g.set_xlabel("")
                plt.title(f"Object Detection Rate at {threshold}% IOU Threshold, {feature[0]} Comparison")
                # plt.xlabel('Epoch')
                plt.ylabel("Object Detection Rate")

                plt.savefig(f"{feature[0]}_boxplot_threshold{threshold}_{ae_dataset}", bbox_inches='tight')

        plt.figure(figsize=(7,5))


    

def group_experiments_by_predictions(experiments, category):
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
            "size": np.mean(
                np.array(list(category_experiments["size"])), axis=0
            )
        }
        
        new_experiments.append(new_experiment)


    experiments_dataframe = pd.DataFrame(new_experiments)
    return experiments_dataframe

def group_experiments_by_epoch(experiments, category):
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
            category: category_value
            }

        for value in ["val_mrcnn_class_loss", "val_mrcnn_bbox_loss", "val_mrcnn_mask_loss", "val_rpn_bbox_loss", "val_rpn_class_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss", "rpn_bbox_loss", "rpn_class_loss"]:
            try:
                new_experiment[value]=np.mean(
                np.array(list(category_experiments[value])), axis=0
                )
            except:
                print("unable to get the mean across axis 0 for value: " + value)
                print(category_experiments)
                new_experiment[value]=np.zeros(100)

        #     "val_mrcnn_class_loss": np.mean(
        #         np.array(list(category_experiments["val_mrcnn_class_loss"])), axis=0
        #     ),
        #     "val_mrcnn_bbox_loss": np.mean(
        #         np.array(list(category_experiments["val_mrcnn_bbox_loss"])), axis=0
        #     ),
        #     "val_mrcnn_mask_loss": np.mean(
        #         np.array(list(category_experiments["val_mrcnn_mask_loss"])), axis=0
        #     ),
        #     "val_rpn_bbox_loss": np.mean(
        #         np.array(list(category_experiments["val_rpn_bbox_loss"])), axis=0
        #     ),
        #     "val_rpn_class_loss": np.mean(
        #         np.array(list(category_experiments["val_rpn_class_loss"])), axis=0
        #     ),
        #     "mrcnn_class_loss": np.mean(
        #         np.array(list(category_experiments["mrcnn_class_loss"])), axis=0
        #     ),
        #     "mrcnn_bbox_loss": np.mean(
        #         np.array(list(category_experiments["mrcnn_bbox_loss"])), axis=0
        #     ),
        #     "mrcnn_mask_loss": np.mean(
        #         np.array(list(category_experiments["mrcnn_mask_loss"])), axis=0
        #     ),
        #     "rpn_bbox_loss": np.mean(
        #         np.array(list(category_experiments["rpn_bbox_loss"])), axis=0
        #     ),
        #     "rpn_class_loss": np.mean(
        #         np.array(list(category_experiments["rpn_class_loss"])), axis=0
        #     ),
        # }
        
        new_experiments.append(new_experiment)


    experiments_dataframe = pd.DataFrame(new_experiments)
    return experiments_dataframe


def plot_colour_correction_lossvsvalloss(experiments, name):

    new_experiments = []
    print(experiments.keys())
    for index, experiment in experiments.iterrows():
        print(experiment["val_mrcnn_class_loss"])
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
        experiment["overlaps"] = stringlist_to_list(experiment["overlaps"])
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
                "combo": str(experiment["colour_correction_type"])
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
    plt.savefig(name + "combo_type_loss_vs_valloss")


def stringlist_to_list(stringlist):
    if type(stringlist) != str:
        return [0]
    if len(stringlist) < 3:
        return [0]
    real_list = [float(n) for n in stringlist[1:-1].split(", ")]
    return real_list

def get_labels(buckets):
    # buckets = sorted(list(buckets))
    return [int_to_string(i) for i in buckets]

def plot_size_overlap_boxplots(experiments):
    ts_experiments = []
    ae_dive1_experiments = []
    ae_dive2_experiments = []
    ae_dive3_experiments = []
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
            "overlaps_AE_area1",
            "overlaps_AE_area2",
            "overlaps_AE_area3",
            "gt_size_list",
            "overlaps",
            "gt_class_list",
            "gt_size_list_AE_area1",
            "gt_class_list_AE_area1",
            "gt_size_list_AE_area2",
            "gt_class_list_AE_area2",
            "gt_size_list_AE_area3",
            "gt_class_list_AE_area3",
            "predicted_size_list",
            "predicted_class_list",
            "predicted_size_list_AE_area1",
            "predicted_class_list_AE_area1",
            "predicted_size_list_AE_area2",
            "predicted_class_list_AE_area2",
            "predicted_size_list_AE_area3",
            "predicted_class_list_AE_area3",
        ]:
            print(value)
            experiment[value] = stringlist_to_list(experiment[value])

        experiment["size_bucket"] = size_buckets(experiment["predicted_size_list"], 500)
        experiment["ae_size_bucket_dive1"] = size_buckets(experiment["predicted_size_list_AE_area1"], 20000)
        experiment["ae_size_bucket_dive2"] = size_buckets(experiment["predicted_size_list_AE_area2"], 20000)
        experiment["ae_size_bucket_dive3"] = size_buckets(experiment["predicted_size_list_AE_area3"], 20000)
        for prediction in range(len(experiment["gt_class_list"])):
            if experiment["overlaps"][prediction] >= 0.3:
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
                    "size_bucket":experiment["size_bucket"][prediction],
                    "gt_size":experiment["gt_size_list"][prediction],
                    "gt_class":experiment["gt_class_list"][prediction],
                    "overlap":experiment["overlaps"][prediction],
                    "dataset":"tunasand"
                }
                if new_row['rescaled'] in ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"]:
                    ts_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area1"])):
            if experiment["overlaps_AE_area1"][prediction] >= 0.3:
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
                    "size_bucket":experiment["ae_size_bucket_dive1"][prediction],
                    "gt_class":experiment["gt_class_list_AE_area1"][prediction],
                    "overlap":experiment["overlaps_AE_area1"][prediction],
                    "dataset":"ae_dive1"
                }
                if new_row['rescaled'] in ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"]:
                    ae_dive1_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area2"])):
            if experiment["overlaps_AE_area2"][prediction] >= 0.3:
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
                    "size_bucket":experiment["ae_size_bucket_dive2"][prediction],
                    "gt_class":experiment["gt_class_list_AE_area2"][prediction],
                    "overlap":experiment["overlaps_AE_area2"][prediction],
                    "dataset":"ae_dive2"
                }
                if new_row['rescaled'] in ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"]:
                    ae_dive2_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area3"])):
            if experiment["overlaps_AE_area3"][prediction] >= 0.3:
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
                    "size_bucket":experiment["ae_size_bucket_dive3"][prediction],
                    "gt_class":experiment["gt_class_list_AE_area3"][prediction],
                    "overlap":experiment["overlaps_AE_area3"][prediction],
                    "dataset":"ae_dive3"
                }
                if new_row['rescaled'] in ["not_rescaled", "res_nn", "drop_res_nn", "drop_res_up_nn"]:
                    ae_dive3_experiments.append(new_row)




    # Filter dictionary by keeping elements whose keys are divisible by 2
    ts_experiments = pd.DataFrame(ts_experiments)
    ae_dive1_experiments = pd.DataFrame(ae_dive1_experiments)
    ae_dive2_experiments = pd.DataFrame(ae_dive2_experiments)
    ae_dive3_experiments = pd.DataFrame(ae_dive3_experiments)

    for value in set(ts_experiments['rescaled']):
        subset = ts_experiments[ts_experiments['rescaled']==value]

        sns.set()
        labels=get_labels(set(subset['size_bucket']))
        plt.clf()
        # palette = get_palette(subset['size_bucket'])
        counts = subset.groupby("size_bucket").count()
        print(value)
        print(counts)
        print(counts['rescaled'].sum())

        ax = sns.boxplot(x=subset["size_bucket"], y=subset["overlap"])

        axes = plt.gca()
        axes.set_ylabel("IOU scores")
        axes.set_xlabel("Size of groundtruth segment in pixels")
        ax.tick_params(which='major', width=1.00)
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', width = 0.75)
        ax.tick_params(which='minor', length=2.5)
        # ax.get_legend().remove()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        new_labels=['blah']
        new_labels.extend(labels[0::5])
        ax.xaxis.set_ticklabels(new_labels,rotation=50)

        plt.title("IOU to size of groundtruth scatterplot")
        plt.savefig(f"scatter_{value}_box_plot", bbox_inches='tight')

    plt.clf()

    ax = sns.boxplot(x=ts_experiments["gt_class"], y=ts_experiments["overlap"], hue=ts_experiments['size_bucket'])


    axes = plt.gca()
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width = 0.75)
    ax.tick_params(which='minor', length=2.5)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # new_labels=['blah']
    # new_labels.extend(labels[0::5])
    # ax.xaxis.set_ticklabels(new_labels,rotation=50)

    plt.title("IOU to size of groundtruth scatterplot")
    plt.savefig("scatter_class_box_plot", bbox_inches='tight')


    plt.clf()
    sns.boxplot(x=ae_dive1_experiments["size_bucket"], y=ae_dive1_experiments["overlap"])

    axes = plt.gca()
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    # axes.set_xticklabels(labels,rotation=50)
    plt.title("IOU to size of groundtruth scatterplot - " + experiment["rescaled"])
    plt.savefig("ae_dive1_scatter_box_plot", bbox_inches='tight')
    plt.clf()

    sns.boxplot(x=ae_dive2_experiments["size_bucket"], y=ae_dive2_experiments["overlap"])

    axes = plt.gca()
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    # axes.set_xticklabels(labels,rotation=50)
    plt.title("IOU to size of groundtruth scatterplot - " + experiment["rescaled"])
    plt.savefig("ae_dive2_scatter_box_plot", bbox_inches='tight')
    plt.clf()

    sns.boxplot(x=ae_dive3_experiments["size_bucket"], y=ae_dive3_experiments["overlap"])

    axes = plt.gca()
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    # axes.set_xticklabels(labels,rotation=50)
    plt.title("IOU to size of groundtruth scatterplot - " + experiment["rescaled"])
    plt.savefig("ae_dive3_scatter_box_plot", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(15,5))
    for value in instance_level_graph_settings:
        plt.clf()
        g = sns.boxplot(
            x=ts_experiments[value[0]],
            y=ts_experiments["overlap"],
            palette=value[1],
            order=value[2],
            fliersize=1,
            hue=ts_experiments["rescaled"],
        )
        print(value[2])

        # g.get_legend().remove()
        g.set_xticklabels(value[3], rotation=0)
        g.set_xlabel("")
        g.set_ylim([0,1])
        # plt.title(value[0])
        # plt.xlabel('Epoch')
        plt.ylabel("Segmentation IOU Scores")

        plt.savefig(f"{value[0]}_IOU_boxplot", bbox_inches='tight')


        for ae_dataset in [[ae_dive1_experiments, "AE_area1"], [ae_dive2_experiments, "AE_area2"], [ae_dive3_experiments, "AE_area3"]]:
            plt.clf()
            g = sns.boxplot(
                x=ae_dataset[0][value[0]],
                y=ae_dataset[0]["overlap"],
                hue=ae_dataset[0]["rescaled"],
                order=value[2],
                palette=value[1],
            )

            # g.get_legend().remove()
            g.set_xticklabels(value[3], rotation=0)
            g.set_xlabel("")
            g.set_ylim([0,1])
            # plt.title(value[0])
            # plt.xlabel('Epoch')
            plt.ylabel("Segmentation IOU Scores")

            plt.savefig(f"{value[0]}_IOU_boxplot_{ae_dataset[1]}", bbox_inches='tight')






def plot_size_overlap_scatter(experiments):

    ts_experiments = []
    ae_dive1_experiments = []
    ae_dive2_experiments = []
    ae_dive3_experiments = []
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
            "overlaps_AE_area1",
            "overlaps_AE_area2",
            "overlaps_AE_area3",
            "gt_size_list",
            "overlaps",
            "gt_class_list",
            "gt_size_list_AE_area1",
            "gt_class_list_AE_area1",
            "gt_size_list_AE_area2",
            "gt_class_list_AE_area2",
            "gt_size_list_AE_area3",
            "gt_class_list_AE_area3",
        ]:
            print(value)
            experiment[value] = stringlist_to_list(experiment[value])

        for prediction in range(len(experiment["gt_class_list"])):
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
                "gt_size":experiment["gt_size_list"][prediction],
                "gt_class":experiment["gt_class_list"][prediction],
                "overlap":experiment["overlaps"][prediction],
            }
            ts_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area1"])):
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
                "gt_size":experiment["gt_size_list_AE_area1"][prediction],
                "gt_class":experiment["gt_class_list_AE_area1"][prediction],
                "overlap":experiment["overlaps_AE_area1"][prediction],
                "dataset":"ae_dive1"
            }
            ae_dive1_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area2"])):
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
                "gt_size":experiment["gt_size_list_AE_area2"][prediction],
                "gt_class":experiment["gt_class_list_AE_area2"][prediction],
                "overlap":experiment["overlaps_AE_area2"][prediction],
                "dataset":"ae_dive2"
            }
            ae_dive2_experiments.append(new_row)
        for prediction in range(len(experiment["gt_class_list_AE_area3"])):
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
                "gt_size":experiment["gt_size_list_AE_area3"][prediction],
                "gt_class":experiment["gt_class_list_AE_area3"][prediction],
                "overlap":experiment["overlaps_AE_area3"][prediction],
                "dataset":"ae_dive3"
            }
            ae_dive3_experiments.append(new_row)


    ts_experiments = pd.DataFrame(ts_experiments)
    ae_dive1_experiments = pd.DataFrame(ae_dive1_experiments)
    ae_dive2_experiments = pd.DataFrame(ae_dive2_experiments)
    ae_dive3_experiments = pd.DataFrame(ae_dive3_experiments)



    sns.set()

    axes = plt.gca()

    axes.set_ylim([-0.01, 1])
    axes.set_xscale('log')
    axes.set_xlim([1, 30000])
    axes.set_ylabel("IOU scores")
    axes.set_xlabel("Size of groundtruth segment in pixels")
    plt.legend()
    plt.title("IOU to size of groundtruth scatterplot - all data points")
    splot=sns.scatterplot(x="gt_size", y="overlap", hue="gt_class", data=ts_experiments, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(ts_experiments["gt_class"]))))
    # splot.set(xscale="log")


    plt.savefig("ts_all_points_scatter_plot", bbox_inches='tight')
    plt.clf()


    # axes = plt.gca()
    # # axes.set_xlim([0, 30000])
    # axes.set_ylim([-0.01, 1])
    # axes.set_ylabel("IOU scores")
    # axes.set_xscale('log')
    # axes.set_xlim([1, 3000000])
    # axes.set_xlabel("Size of groundtruth segment in pixels")
    # plt.legend()
    # plt.title("IOU to size of groundtruth scatterplot - all data points")
    # splot=sns.scatterplot(x="gt_size", y="overlap", hue="gt_class", data=ae_dive1_experiments, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(ae_dive1_experiments["gt_class"]))))
    # # splot.set(xscale="log")
    # plt.savefig("ae_dive1_all_points_scatter_plot", bbox_inches='tight')
    # plt.clf()


    # axes = plt.gca()
    # # axes.set_xlim([0, 30000])

    # axes.set_xscale('log')
    # axes.set_xlim([1, 3000000])
    # axes.set_ylim([-0.01, 1])
    # axes.set_ylabel("IOU scores")
    # axes.set_xlabel("Size of groundtruth segment in pixels")
    # plt.legend()
    # plt.title("IOU to size of groundtruth scatterplot - all data points")
    # splot=sns.scatterplot(x="gt_size", y="overlap", hue="gt_class", data=ae_dive2_experiments, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(ae_dive2_experiments["gt_class"]))))
    # # splot.set(xscale="log")
    # plt.savefig("ae_dive2_all_points_scatter_plot", bbox_inches='tight')
    # plt.clf()


    # axes = plt.gca()
    # axes.set_xscale('log')
    # axes.set_xlim([1, 3000000])
    # axes.set_ylim([-0.01, 1])
    # axes.set_ylabel("IOU scores")
    # axes.set_xlabel("Size of groundtruth segment in pixels")
    # plt.legend()
    # plt.title("IOU to size of groundtruth scatterplot - all data points")
    # splot=sns.scatterplot(x="gt_size", y="overlap", hue="gt_class", data=ae_dive3_experiments, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(ae_dive3_experiments["gt_class"]))))
    # # splot.set(xscale="log")
    # plt.savefig("ae_dive3_all_points_scatter_plot", bbox_inches='tight')
    # plt.clf()



    for r in set(ts_experiments["rescaled"]):
        rescaled_set = ts_experiments[ts_experiments['rescaled']==r]

        axes = plt.gca()
        axes.set_xscale('log')
        axes.set_xlim([1, 30000])
        axes.set_ylim([-0.01, 1])
        axes.set_ylabel("IOU scores")
        axes.set_xlabel("Size of groundtruth segment in pixels")
        plt.legend()
        plt.title("IOU to size of groundtruth scatterplot - all rescaling")
        splot=sns.scatterplot(x="gt_size", y="overlap", hue="gt_class", data=rescaled_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(rescaled_set["gt_class"]))))
        # splot.set(xscale="log")
        plt.savefig("ts_rescaled_scatter_plot_"+str(r), bbox_inches='tight')
        plt.clf()


    # for r in set(ae_dive1_experiments["rescaled"]):
    #     rescaled_set = ae_dive1_experiments[ae_dive1_experiments['rescaled']==r]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 30000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="class", data=rescaled_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(rescaled_set["class"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive1_rescaled_scatter_plot_"+str(r), bbox_inches='tight')
    #     plt.clf()


    # for r in set(ae_dive2_experiments["rescaled"]):
    #     rescaled_set = ae_dive2_experiments[ae_dive2_experiments['rescaled']==r]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 30000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="class", data=rescaled_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(rescaled_set["class"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive2_rescaled_scatter_plot_"+str(r), bbox_inches='tight')
    #     plt.clf()

    # for r in set(ae_dive3_experiments["rescaled"]):
    #     rescaled_set = ae_dive3_experiments[ae_dive3_experiments['rescaled']==r]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 30000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="class", data=rescaled_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(rescaled_set["class"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive3_rescaled_scatter_plot_"+str(r), bbox_inches='tight')
    #     plt.clf()


    for c in set(ts_experiments["gt_class"]):
        class_set = ts_experiments[ts_experiments['gt_class']==c]

        axes = plt.gca()
        axes.set_xscale('log')
        axes.set_xlim([1, 30000])
        axes.set_ylim([-0.01, 1])
        axes.set_ylabel("IOU scores")
        axes.set_xlabel("Size of groundtruth segment in pixels")
        plt.legend()
        plt.title("IOU to size of groundtruth scatterplot - all rescaling")
        splot=sns.scatterplot(x="gt_size", y="overlap", hue="rescaled", data=class_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(class_set["rescaled"]))))
        # splot.set(xscale="log")
        plt.savefig("ts_class_scatter_plot_"+str(c).split('.')[0], bbox_inches='tight')
        plt.clf()

    # for c in set(ae_dive1_experiments["class"]):
    #     class_set = ae_dive1_experiments[ae_dive1_experiments['class']==c]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 3000000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="rescaled", data=class_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(class_set["rescaled"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive1_class_scatter_plot_"+str(c).split('.')[0], bbox_inches='tight')
    #     plt.clf()



    # for c in set(ae_dive2_experiments["class"]):
    #     class_set = ae_dive2_experiments[ae_dive2_experiments['class']==c]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 3000000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="rescaled", data=class_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(class_set["rescaled"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive2_class_scatter_plot_"+str(c).split('.')[0], bbox_inches='tight')
    #     plt.clf()


    # # axes.set_xlim([0, 30000])
    # for c in set(ae_dive3_experiments["class"]):
    #     class_set = ae_dive3_experiments[ae_dive3_experiments['class']==c]

    #     axes = plt.gca()
    #     axes.set_xscale('log')
    #     axes.set_xlim([1, 3000000])
    #     axes.set_ylim([-0.01, 1])
    #     axes.set_ylabel("IOU scores")
    #     axes.set_xlabel("Size of groundtruth segment in pixels")
    #     plt.legend()
    #     plt.title("IOU to size of groundtruth scatterplot - all rescaling")
    #     splot=sns.scatterplot(x="size", y="overlap", hue="rescaled", data=class_set, s=5, marker='.', linewidth=0, palette=sns.color_palette("muted", n_colors=len(set(class_set["rescaled"]))))
    #     # splot.set(xscale="log")
    #     plt.savefig("ae_dive3_class_scatter_plot_"+str(c).split('.')[0], bbox_inches='tight')
    #     plt.clf()


def plot_colour_correction_stackedarea(experiments, name):
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
            "overlaps"
        ]:
            experiment[value] = stringlist_to_list(experiment[value])

        new_experiments.append(experiment)

    experiments_dataframe = pd.DataFrame(new_experiments)
    print(type(experiments_dataframe["val_mrcnn_class_loss"]))
    experiments_dataframe = group_experiments_by_epoch(experiments_dataframe, "rescaled")
    print(type(experiments_dataframe["val_mrcnn_class_loss"]))

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
        axes.set_ylabel("Training Loss")
        axes.set_xlabel("Training Epoch")
        print(x)
        print(y)
        plt.stackplot(x, y, labels=labels)
        plt.legend()
        # plt.title(experiment["rescaled"])
        print(name)
        plt.savefig(str(name) + str(experiment["rescaled"]) + "_stacked_loss", bbox_inches='tight')
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
        axes.set_ylabel("Validation Loss")
        axes.set_xlabel("Training Epoch")
        plt.legend()
        # plt.title(experiment["rescaled"])
        plt.savefig(str(name) + str(experiment["rescaled"]) + "_stacked_val_loss", bbox_inches='tight')
        plt.clf()




def compile_dataframes():
    all_filenames=os.listdir("./maskandclassloss")
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv("./maskandclassloss/"+f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "maskandclassloss_csv.csv", index=False, encoding='utf-8-sig')

def mAP_summary_table(experiments_dataframe):
    print(experiments_dataframe.index)
    mAP_table = experiments_dataframe.groupby(
                                    by=["colour_correction_type", "rescaled", "distortion_correction", "elastic_distortions", "separate_channel_ops"],
                                ).agg([np.mean, np.std])['mean_AP']
    mAP_table.to_csv("mAP_table.csv")

    ae_mAP_table = experiments_dataframe.groupby(
                                    by=["colour_correction_type", "rescaled", "distortion_correction", "elastic_distortions", "separate_channel_ops"],
                                ).agg([np.mean, np.std])[['mean_classless_AP_AE_area1', 'mean_classless_AP_AE_area2', 'mean_classless_AP_AE_area3']]
    ae_mAP_table.to_csv("ae_mAP_table.csv")

experiments = pd.read_csv("./experiments_dataframe_17.csv")
plot_boxplots(experiments)
# plot_colour_correction_stackedarea(experiments, "all_loss")
# plot_colour_correction_lossvsvalloss(experiments, "all_loss")
plot_size_overlap_scatter(experiments)
plot_size_overlap_boxplots(experiments)

# experiments = pd.read_csv("./maskandclassloss_csv.csv")
# plot_boxplots(experiments)
# plot_colour_correction_stackedarea(experiments, "maskandclass_loss")
# plot_colour_correction_lossvsvalloss(experiments, "maskandclass_loss")
# plot_size_overlap_scatter(experiments)
# plot_size_overlap_boxplots(experiments)


# experiments = pd.read_csv("./masklossonly_csv.csv")
# plot_boxplots(experiments)
# plot_colour_correction_stackedarea(experiments, "mask_loss")
# print(experiments)
# plot_colour_correction_lossvsvalloss(experiments, "mask_loss")
# plot_size_overlap_scatter(experiments)
# plot_size_overlap_boxplots(experiments)


