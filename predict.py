from keras_segmentation.predict import predict,predict_multiple,predict_video
from keras_segmentation.models.all_models import model_from_name

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

test_image_path = "/home/heemoon/Desktop/0_DATABASE/3_IRIS/cow/rgb/set1/11_9.png"
checkpoints_saving_path = "checkpoints/cow_iris_3/set"
dataset_abbr = "set"
out_folder = "out_frame/"+dataset_abbr

model_list = [
#            "fcn_16_vgg",
#            "fcn_32_vgg",
#            "fcn_8_vgg",
#            "fcn_8_resnet50",  # big size over 11GB
#            "fcn_16_resnet50",
#            "fcn_32_resnet50", # big size over 11GB
#            "fcn_8_mobilenet",
#            "fcn_16_mobilenet",
#            "fcn_32_mobilenet",

#            "pspnet", # core dump error
#            "vgg_pspnet", # core dump error
#            "resnet50_pspnet", # core dump error
#            "pspnet_50", # big size over 11GB
#            "pspnet_101",

#            "unet_mini",
#            "unet",
#            "vgg_unet",
            "resnet50_unet",
#            "mobilenet_unet",

#            "segnet",
#            "vgg_segnet",
#            "resnet50_segnet",
#            "mobilenet_segnet"
            ]


for model_name in model_list:
    for i in range(1,2):
        #get model file name
        model_file_name = model_name+"_"+dataset_abbr+str(i)
        checkpoints_path_ = checkpoints_saving_path+str(i)+"/"+model_file_name
        
        # model define
        print("------------ Define Model:"+model_file_name+" ------------")

        try:
            # Single Predict
            predict( 
                checkpoints_path = checkpoints_path_, 
                inp = test_image_path, 
                out_fname = out_folder+str(i)+"/"+model_file_name+"_11_9.png",
                overlay_img=True
            )

            # Multi Predict
            # predict_multiple( 
            #    checkpoints_path=checkpoints_path_, 
            #    inp_dir=test_image_path, 
            #    out_dir=out_folder+str(i)+"/",
            #    overlay_img=True,
            #    class_names=None, show_legends=False,
            #    prediction_width=None, prediction_height=None,
            # )

            # Video Predict
            # predict_video(
            #     checkpoints_path=checkpoints_path_, 
            #     inp=test_image_path, # should be avi file! 
            #     out_fname="output.avi"
            # )
        except Exception as e:
            print("Error: "+model_file_name+"\n",e)



