import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import MatSegReader
parser = argparse.ArgumentParser(description='Train on MatSim')
parser.add_argument('--MatSim_dir_object', default= r"sample_data/VesselTrainData/", type=str, help='input folder 1 MatSim synthethic Objects dataset main dir')
parser.add_argument('--MatSim_dir_vessel', default= r"sample_data/ObjectTrainData/", type=str, help='input folder 2 MatSim synthethic Vessels dataset main dir')
parser.add_argument('--MaxPixels', default= 800*800*12, type=int, help='max Size of input matrix in pixels H*W*BatchSize (reduce to solve cuda out of memory)')
parser.add_argument('--MaxImagesInBatch', default = 15, type=int, help='max images in a a batch (reduce to solve cuda out of memory)')
parser.add_argument('--temp', default= 0.2, type=float, help='temperature for softmax')
parser.add_argument('--weight_decay', default= 4e-5, type=float, help='optimizer weight decay')
parser.add_argument('--learning_rate', default= 1e-5, type=float, help='optimizer learning rate')
args = parser.parse_args()
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
###############################################################################################
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
#****************************************************************************************************************************
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
#****************************************************************************************************************************
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
#****************************************************************************************************************************
def load_image_list(path,im_size=(1000,1000)):
    imgs=[]
    # dir_path = "/media/breakeroftime/2T/Data_zoo/ChemPicsV1.0/Simple/Test/Image/"
    # list_path = ['images/truck.jpg', ]
    list_im=os.listdir(path)

    for i in range(3):
        image = cv2.imread(path+"/"+ list_im[np.random.randint(len(list_im))])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image,im_size)
        imgs.append(img)
    return imgs


#*****************************************************************
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"
import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # Build model all parts, load trained weight, set model to eval
sam.to(device=device)

#--------Image encoder------------------------------------------------------------------------------------------------------------

predictor = SamPredictor(sam)

optimizer = torch.optim.AdamW(params=predictor.model.mask_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
for itr in range(1000):
    with torch.no_grad():
        image_list=load_image_list(path=r"/media/breakeroftime/2T/Data_zoo/ChemPicsV1.0/Simple/Train/Image",im_size=[1000,1000])
        predictor.model.image_encoder.eval()
        predictor.set_multiple_image(image_list)#image(image_list)


    #--------Decoder with training------------------------------------------------------------------------------


    with torch.enable_grad():
        input_point = np.array([[[500, 500]],[[500, 500]],[[500, 500]]])
        input_label = np.array([[1],[1],[1]])


        predictor.model.mask_decoder.requires_grad_(True)
        predictor.model.mask_decoder.train()
        predictor.model.prompt_encoder.requires_grad_(True)
        predictor.model.prompt_encoder.train()

        # Create  optimizer
        #scaler = torch.cuda.amp.GradScaler()
        #with torch.cuda.amp.autocast():
            #predictor.model.train()
        predictor.model.zero_grad()
       # predictor.model.mask_decoder.zero_grad()
        masks, iou_predictions, low_res_masks = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=None,#input_box[None, :],
                multimask_output=True,)
        before2 = predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy() + 0
        before = predictor.model.mask_decoder.transformer.final_attn_token_to_image.q_proj.weight.sum().abs().detach().cpu().numpy() + 0
      #  print("before,", predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy())
        loss=low_res_masks.mean().abs()#.requires_grad(True)
        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()  # Backpropogate loss caler used for mix precision
        # scaler.step(optimizer)  # Apply gradient descent change to weight scaler used for mix precision
        # scaler.update()
        after2 = predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy() + 0
        after = predictor.model.mask_decoder.transformer.final_attn_token_to_image.q_proj.weight.sum().abs().detach().cpu().numpy() + 0
        print("before",before,"after", after,"dif",after-before)
#-----------------------------------------------------------------------------------------------------------------------------
    for i in range(len(image_list)):
        image=image_list[i]
        plt.figure(figsize=(10,10))
        plt.imshow(image)


        show_points(input_point[i], input_label[i], plt.gca())
        plt.axis('on')
        plt.show()
        #
        # # input_box = np.array([425, 600, 700, 875])
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        for kk in range(masks.shape[1]):
            show_mask(masks[i][kk], plt.gca())
            show_points(input_point[i][0], input_label[i][0], plt.gca())
            plt.axis('off')
            plt.show()