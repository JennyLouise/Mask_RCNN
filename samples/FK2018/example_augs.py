import imgaug.augmenters as iaa
import os
import cv2

def example_augs(add_freq=0.1, add_value=(-10,10), add_pc_freq=0.5, multiply_freq=0.1, 
                multiply_value=(0.75,1.25), multiply_pc_freq=0.5, snp_freq=0.1, snp_p=0.05, jpeg_freq=0.1, 
                jpeg_compression=(1,5), gaussian_freq=0.1, gaussian_sigma=(0.01,0.7), motion_freq=0.1, motion_k=(3,10), 
                contrast_freq=0.1, contrast_alpha=(0.5,1.5), fliplr=0.5, flipud=0.5, affine_freq=0.1, 
                affine_scale=(0,0.02), transform_freq=0.1, transform_scale=(0,0.05), elastic_freq=0.1, elastic_sigma=(4, 6), 
                elastic_alpha=(0,7), rotate=1, dataset="/scratch/jw22g14/FK2018/second_set/"):


    augmentation = iaa.Sequential([
        iaa.Sometimes(add_freq, iaa.Add(value=add_value, per_channel=add_pc_freq)),
        iaa.Sometimes(multiply_freq, iaa.Multiply(mul=multiply_value, per_channel=multiply_pc_freq)),
        iaa.Sometimes(snp_freq, iaa.SaltAndPepper(snp_p)),
        iaa.Sometimes(jpeg_freq, iaa.JpegCompression(compression=jpeg_compression)),
        iaa.Sometimes(gaussian_freq, iaa.GaussianBlur(sigma=gaussian_sigma)),
        iaa.Sometimes(motion_freq, iaa.MotionBlur(k=motion_k)),
        iaa.Sometimes(contrast_freq, iaa.LinearContrast(alpha=contrast_alpha)),
        iaa.Fliplr(fliplr),
        iaa.Flipud(flipud),
        iaa.Sometimes(affine_freq, iaa.PiecewiseAffine(scale=affine_scale, nb_rows=8, nb_cols=8)),
        iaa.Sometimes(transform_freq, iaa.PerspectiveTransform(scale=transform_scale, keep_size=True)),
        iaa.Sometimes(elastic_freq, iaa.ElasticTransformation(sigma=elastic_sigma, alpha=elastic_alpha)),
        iaa.Sometimes(rotate, iaa.Rot90([0,1,2,3]))
        ], random_order=True)

    images=[]
    image_names=[]
    for filename in os.listdir(dataset):
        if(filename[-4:]=='.png'):
            image=cv2.imread(dataset+filename)
            images.append(image)
            image_names.append(filename)

    print("running augmentation")
    images_aug=augmentation(images=images)
    print("augmented!")
    for i, image in enumerate(images_aug):
        cv2.imwrite("./augmented/"+image_names[i], image)



example_augs(dataset="/Users/jenny/Documents/uni/data/FK2018/subset/")