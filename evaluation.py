from keras_segmentation.models.all_models import model_from_name
from imgaug import augmenters as iaa
from keras_segmentation.predict import model_from_checkpoint_path

################# Configure Here ################
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

checkpoints_saving_path = "checkpoints/cow_iris_3/set"
dataset_abbr = "set"

path_base = "/home/heemoon/Desktop/0_DATABASE/3_IRIS/cow/"
inp_images_path = path_base+ "rgb/test"
inp_annotations_path = path_base + "iii_format"

model_list = [
            "fcn_16_vgg",
            "vgg_unet",
            "vgg_segnet",
            "fcn_32_vgg",
            "fcn_8_vgg",

            "fcn_16_resnet50",
            "resnet50_unet",
            "resnet50_segnet",
            "fcn_32_resnet50", 
            "fcn_8_resnet50",
    
            "fcn_8_mobilenet",
            "fcn_16_mobilenet",
            "mobilenet_unet",
            "mobilenet_segnet",
            "fcn_32_mobilenet",

            # "pspnet", # core dump error
            # "vgg_pspnet", # core dump error
            # "resnet50_pspnet", # core dump error
            # "pspnet_50", # big size over 11GB
            # "pspnet_101",
    
            # "unet_mini",
            # "unet",
            # "segnet",
            ]

DO_Augment = True
def custom_augmentation():
    return  iaa.Sequential(
        [
            # apply the following augmenters to most images
            # https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
            iaa.AddToBrightness((-10, 10)),  
            #iaa.CropAndPad(percent=(-0.25, 0.25)),
            #iaa.ContrastNormalization(0.5),
            #iaa.AllChannelsHistogramEqualization(),
            iaa.Affine(rotate=(-40, 40))
        ])

################################################

f = open(dataset_abbr+"_test_result_.txt", "w")

for model_name in model_list:
    for i in range(1,6):
        #get model file name
        model_file_name = model_name+"_"+dataset_abbr+str(i)

        # model define
        print("------------ Define Model:"+model_file_name+" ------------")
        
        try:
            # evaluating the model 
            model = model_from_checkpoint_path(checkpoints_saving_path+str(i)+"/"+model_file_name)
            result = model.evaluate_segmentation( inp_images_dir=inp_images_path,
                             annotations_dir=inp_annotations_path )
            f.write(model_file_name+"==========="+str(result)+"\n")
            print("==============="+model_file_name+"==============="+"\n"+str(result))
        except Exception as e:
            print("Error: "+model_file_name+"\n",e)
            f.write("Error: "+model_file_name+"==========="+"\n")
    f.write("\n")
    
f.close()

print("end of evaluation.py")
