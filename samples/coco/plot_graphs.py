import matplotlib.pyplot as plt
import csv
import os
import math
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pandas as pd




def get_experiments():
	experiments=[]
	directories = [x for x in os.walk('../../../')][0][1]
	for directory in directories:
		colour_correction_type = directory.split('_')[0]

		if(directory.split('_')[1]=='dist'):
			distortion_correction = True
		else:
			distortion_correction = False

		if(directory.split('_')[2]=='rescaled'):
			rescaled = True
		else:
			rescaled = False

		filenames=[f for f in os.listdir(directory) if f[-4:] == '.csv']
		print filenames

		for filename in filenames:
			experiment={'colour_correction_type': colour_correction_type, 'distortion_correction': distortion_correction, 'rescaled': rescaled}
			with open('./'+directory+'/'+filename, 'r') as csvfile:
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


def plot_boxplots(experiments):
	plt.clf()
	axes = plt.gca()
	axes.set_ylim([0,4.6])
	fig, ax = plt.subplots()

	experiments_dataframe = pd.DataFrame(experiments)
	print(experiments_dataframe)

	sns.boxplot(x=experiments_dataframe['distortion_correction'],y=experiments_dataframe['minimum_val_loss'],hue=experiments_dataframe['colour_correction_type'], palette="Blues")
	plt.title('Colour correction type')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Value')

	plt.show()

def plot_colour_correction_graphs(experiments):
	colour_correction_type_lists={'ac':{'loss':[], 'val_loss':[]}, 'gw':{'loss':[], 'val_loss':[]}, 'histogram':{'loss':[], 'val_loss':[]}}


	for experiment in experiments:
		colour_correction_type_lists[experiment['colour_correction_type']]['loss'].append(experiment['loss'])
		colour_correction_type_lists[experiment['colour_correction_type']]['val_loss'].append(experiment['val_loss'])



	plt.clf()
	axes = plt.gca()
	axes.set_ylim([0,4.6])
	fig, ax = plt.subplots()

	plt.plot(np.mean(colour_correction_type_lists['ac']['loss'], axis=0), label='Attenuation Correction', color='blue', linestyle='-')
	plt.plot(np.mean(colour_correction_type_lists['ac']['val_loss'], axis=0), color='blue', linestyle='--')
	plt.plot(np.mean(colour_correction_type_lists['gw']['loss'], axis=0), label='Greyworld Correction', color='red', linestyle='-')
	plt.plot(np.mean(colour_correction_type_lists['gw']['val_loss'], axis=0), color='red', linestyle='--')
	plt.plot(np.mean(colour_correction_type_lists['histogram']['loss'], axis=0), label='Histogram Normalisation', color='green', linestyle='-')
	plt.plot(np.mean(colour_correction_type_lists['histogram']['val_loss'], axis=0), color='green', linestyle='--')


	#TODO add legend items for training loss vs validation loss
	plt.legend()
	plt.title('Colour correction type')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Value')

	plt.show()




	# 

plot_boxplots(get_experiments())

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
