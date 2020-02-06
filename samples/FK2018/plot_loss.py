log_file = open("./slurm-182629_0.out", 'r')

values_of_interest=["loss", "rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss", "val_loss", "val_rpn_class_loss", "val_rpn_bbox_loss", "val_mrcnn_class_loss", "val_mrcnn_bbox_loss", "val_mrcnn_mask_loss"]

log=[]

for line in log_file.readlines():
	if line[0] == "5":
		line_log={}
		for value in values_of_interest:
			line_log[value] = float(line.split(value+": ")[1].split(" - ")[0])
		print(line_log)
		log.append(line_log)

import csv
keys = log[0].keys()
print(keys)
with open('results-182629.csv', 'w') as output_file:
	dict_writer = csv.DictWriter(output_file, values_of_interest)
	dict_writer.writeheader()
	dict_writer.writerows(log)
