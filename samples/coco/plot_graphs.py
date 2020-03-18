import matplotlib.pyplot as plt
import csv
import os
import math
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA




def get_experiments():
	experiments=[]
	directories = [x for x in os.walk('../../../paper_experiments/weights/')][0][1]
	for directory in directories:
		colour_correction_type = directory.split('-')[0]

		if(directory.split('-')[1]=='distortion_correction'):
			distortion_correction = True
		else:
			distortion_correction = False

		rescaled = directory.split('-')[2]


		filenames=[f for f in os.listdir('../../../paper_experiments/weights/'+directory) if f[-4:] == '.csv']
		print filenames

		for filename in filenames:
			experiment={'colour_correction_type': colour_correction_type, 'distortion_correction': distortion_correction, 'rescaled': rescaled}
			with open('../../../paper_experiments/weights/'+directory+'/'+filename, 'r') as csvfile:
				plots= csv.reader(csvfile, delimiter=',')
				headers = next(plots, None)
				for header in headers:
					experiment[header] =[]
				for row in plots:
					for i, header in enumerate(headers):
						experiment[header].append(float(row[i]))

			experiment['minimum_val_loss']=min(experiment['val_loss'])
			experiment['minimum_loss']=min(experiment['loss'])
			number = int(filename.split('_')[1].split('.')[0])
			experiment['number'] = number
			experiment['repeat'] = math.floor(number/4)
			if((number%4)/2<1):
				experiment['elastic_distortions']=True
			else:
				experiment['elastic_distortions']=False
			if((number%4)%2 != 0):
				experiment['separate_channel_ops']=True
			else:
				experiment['separate_channel_ops']=False

			print(experiment)
			experiments.append(experiment)



	return experiments

def size_overlaps(size_overlaps_string):
	size_overlaps_list=[]
	split_overlaps = size_overlaps_string[1:-1].split(': ')
	print(split_overlaps)
	exit()
	for item in split_overlaps:
		numbers =[float(x) for x in item.split("]]")[0][2:].split(',')]
		overlaps_list.append(max(numbers))
	return(np.mean(overlaps_list))

def overlaps(overlaps_string):
	overlaps_list=[]
	split_overlaps = overlaps_string[1:-1].split('array(')[1:]
	for item in split_overlaps:
		numbers =[float(x) for x in item.split("]]")[0][2:].split(',')]
		overlaps_list.append(max(numbers))
	return(np.mean(overlaps_list))

def plot_boxplots(experiments):
	plt.clf()
	axes = plt.gca()
	axes.set_ylim([0,4.6])
	fig, ax = plt.subplots()

	new_experiments=[]
	print(experiments.keys())
	for index, experiment in experiments.iterrows():
		print(experiment)
		experiment['val_mrcnn_class_loss'] = experiment['val_mrcnn_class_loss'][1:-1].split(', ')
		experiment['val_loss'] = experiment['val_loss'][1:-1].split(', ')
		experiment['overlaps'] = overlaps(experiment['overlaps'])
		experiment['size_overlaps'] = size_overlaps(experiment['size_overlaps'])
		print(len(experiment['val_mrcnn_class_loss']))
		print(len(experiment['val_loss']))
		for epoch in range(len(experiment['val_loss'])):
			new_row = {'rescaled': experiment['rescaled'], 
						'distortion_correction': experiment['distortion_correction'], 
						'minimum_loss': experiment['minimum_loss'], 
						'number': experiment['number'],
						'minimum_val_loss': experiment['minimum_val_loss'],
						'elastic_distortions': experiment['elastic_distortions'],
						'colour_correction_type':experiment['colour_correction_type'],
						'separate_channel_ops': experiment['separate_channel_ops'],
						'epoch':epoch,
						'mrcnn_mask_loss':experiment['mrcnn_mask_loss'][epoch],
						'val_mrcnn_bbox_loss':experiment['val_mrcnn_bbox_loss'][epoch],
						'loss':experiment['loss'][epoch],
						'val_mrcnn_class_loss':experiment['val_mrcnn_class_loss'][epoch],
						'val_mrcnn_mask_loss':experiment['val_mrcnn_mask_loss'][epoch],
						'mrcnn_bbox_loss':experiment['mrcnn_bbox_loss'][epoch],
						'val_rpn_bbox_loss':experiment['val_rpn_bbox_loss'][epoch],
						'rpn_class_loss':experiment['rpn_class_loss'][epoch],
						'rpn_bbox_loss':experiment['rpn_bbox_loss'][epoch],
						'mrcnn_class_loss':experiment['mrcnn_class_loss'][epoch],
						'val_rpn_class_loss':experiment['val_rpn_class_loss'][epoch],
						'val_loss':experiment['val_loss'][epoch],
						'combo':experiment['colour_correction_type']+str(experiment['distortion_correction'])+str(experiment['rescaled']),
						'overlaps':experiment['overlaps']
						}
			new_experiments.append(new_row)

	experiments_dataframe = pd.DataFrame(new_experiments)
	print(experiments_dataframe)

	g = sns.boxplot(x=experiments_dataframe['colour_correction_type'],y=experiments_dataframe['overlaps'],hue=experiments_dataframe['distortion_correction'], palette="Blues")
	# g.get_legend().remove()
	# g.set_xticklabels(g.get_xticklabels(), rotation=45)
	plt.title('Colour correction type - across classes')
	# plt.xlabel('Epoch')
	plt.ylabel('Mean overlap')

	plt.show()

def plot_colour_correction_graphs(experiments):
	plt.clf()
	axes = plt.gca()
	axes.set_ylim([0,4.6])
	fig, ax = plt.subplots()

	new_experiments=[]
	print(experiments.keys())
	for index, experiment in experiments.iterrows():
		print(experiment)
		experiment['val_mrcnn_class_loss'] = experiment['val_mrcnn_class_loss'][1:-1].split(', ')
		experiment['val_loss'] = experiment['val_loss'][1:-1].split(', ')
		experiment['loss'] = experiment['loss'][1:-1].split(', ')
		experiment['overlaps'] = overlaps(experiment['overlaps'])
		for epoch in range(len(experiment['val_loss'])):
			new_row = {'rescaled': experiment['rescaled'], 
						'distortion_correction': experiment['distortion_correction'], 
						'minimum_loss': experiment['minimum_loss'], 
						'number': experiment['number'],
						'minimum_val_loss': experiment['minimum_val_loss'],
						'elastic_distortions': experiment['elastic_distortions'],
						'colour_correction_type':experiment['colour_correction_type'],
						'separate_channel_ops': experiment['separate_channel_ops'],
						'epoch':epoch,
						'mrcnn_mask_loss':experiment['mrcnn_mask_loss'][epoch],
						'val_mrcnn_bbox_loss':experiment['val_mrcnn_bbox_loss'][epoch],
						'loss':float(experiment['loss'][epoch]),
						'val_mrcnn_class_loss':experiment['val_mrcnn_class_loss'][epoch],
						'val_mrcnn_mask_loss':experiment['val_mrcnn_mask_loss'][epoch],
						'mrcnn_bbox_loss':experiment['mrcnn_bbox_loss'][epoch],
						'val_rpn_bbox_loss':experiment['val_rpn_bbox_loss'][epoch],
						'rpn_class_loss':experiment['rpn_class_loss'][epoch],
						'rpn_bbox_loss':experiment['rpn_bbox_loss'][epoch],
						'mrcnn_class_loss':experiment['mrcnn_class_loss'][epoch],
						'val_rpn_class_loss':experiment['val_rpn_class_loss'][epoch],
						'val_loss':float(experiment['val_loss'][epoch]),
						'combo':experiment['colour_correction_type']+str(experiment['distortion_correction'])+str(experiment['rescaled'])
						}
			new_experiments.append(new_row)

	experiments_dataframe = pd.DataFrame(new_experiments)


	plt.clf()
	axes = plt.gca()
	axes.set_ylim([0,4.6])
	fig, ax = plt.subplots()
	sns.lineplot(x='epoch', y='val_loss', data=experiments_dataframe, hue="rescaled")
	sns.lineplot(x='epoch', y='loss', data=experiments_dataframe, hue="rescaled")
	ax.lines[0].set_linestyle("--")
	ax.lines[1].set_linestyle("--")
	ax.lines[2].set_linestyle("--")
	ax.lines[3].set_linestyle("--")
	print(ax.lines[0].get_linestyle())
	plt.show()


	# 
# experiments=get_experiments()

experiments = pd.read_csv("./classless_dataframe_0.csv")
print(experiments)
# plot_boxplots(experiments)
plot_boxplots(experiments)


		# print(loss)
		# print(val_loss)
		# plt.clf()
		# axes = plt.gca()
		# axes.set_ylim([0,4.6])
		# plt.plot(range(len(loss)), loss, label='Loss')
		# plt.plot(range(len(loss)), val_loss, label='Validation Loss')
		# plt.legend()


		# plt.title(experiment_name + 'experiment '+filename.split('_')[1].split('.')[0])

		# plt.xlabel('Epoch')
		# plt.ylabel('Loss Value')

		# plt.savefig('./'+directory+'/'+filename.split('.')[0]+'.png')
