import argparse
import FK2018

def job_array(array_id, data_path):
    elastic_transformations = False
    separate_channel_operations = 0

    augs_id = array_id%4

    if(augs_id==0):
       elastic_transformations = False
       separate_channel_operations = 0
    if(augs_id==1):
       elastic_transformations = False
       separate_channel_operations = 1
    if(augs_id==2):
       elastic_transformations = True
       separate_channel_operations = 0
    if(augs_id==3):
       elastic_transformations = True
       separate_channel_operations = 1

   
    FK2018.train_nnet(dataset=data_path, elastic_transformations=elastic_transformations, separate_channel_operations=separate_channel_operations, log_file=str(array_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass job array id')
    parser.add_argument('array_id', type=int, help='job array id')
    parser.add_argument('data_path', type=str, help='dataset directory')
    args = parser.parse_args()
    job_array(args.array_id, args.data_path)
