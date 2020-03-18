import os


def plot_loss():
	directories = ['./']
	print(directories)
	for directory in directories:

		filenames=[f for f in os.listdir(directory) if f[0:5] == 'slurm']
		print filenames

		for filename in filenames:
			log_file = open(directory+filename, 'r')
			print(log_file)
			print('./'+directory+filename)

			values_of_interest=["loss", "rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss", "val_loss", "val_rpn_class_loss", "val_rpn_bbox_loss", "val_mrcnn_class_loss", "val_mrcnn_bbox_loss", "val_mrcnn_mask_loss"]

			log=[]

			for line in log_file.readlines():
				if line[0:2] == "15":
					line_log={}
					for value in values_of_interest:
						line_log[value] = float(line.split(value+": ")[1].split(" - ")[0])
					print(line_log)
					log.append(line_log)

			import csv
			print(log)
			if log == []:
				break
			keys = log[0].keys()
			print(keys)
			results_filename =directory+'/results'+filename.split('.')[0]+'.csv'

			with open(results_filename, 'w') as output_file:
				dict_writer = csv.DictWriter(output_file, keys)
				dict_writer.writeheader()
				dict_writer.writerows(log)

if __name__ == '__main__':
	plot_loss()
