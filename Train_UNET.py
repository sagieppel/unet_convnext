import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import MatSegReader
import Unet as unet
import scipy#.optimize import linear_sum_assignment as hungarian_matching

parser = argparse.ArgumentParser(description='Train on MatSim')
parser.add_argument('--MatSim_dir_object', default= r"sample_data/VesselTrainData/", type=str, help='input folder 1 MatSim synthethic Objects dataset main dir')
parser.add_argument('--MatSim_dir_vessel', default= r"sample_data/ObjectTrainData/", type=str, help='input folder 2 MatSim synthethic Vessels dataset main dir')
parser.add_argument('--MaxPixels', default= 800*800*12, type=int, help='max Size of input matrix in pixels H*W*BatchSize (reduce to solve cuda out of memory)')
parser.add_argument('--MaxImagesInBatch', default = 15, type=int, help='max images in a a batch (reduce to solve cuda out of memory)')
parser.add_argument('--temp', default= 0.2, type=float, help='temperature for softmax')
parser.add_argument('--weight_decay', default= 4e-5, type=float, help='optimizer weight decay')
parser.add_argument('--learning_rate', default= 1e-5, type=float, help='optimizer learning rate')
parser.add_argument('--log_dir', default= r"logs_unetPointer/", type=str, help='log folder were train model will be saved')
parser.add_argument('--resume_training_from', default= r"", type=str, help='path to model to resume training from')
parser.add_argument('--sam_weight_path', default= r"sam_vit_h_4b8939.pth", type=str, help='path to model to resume training from')
parser.add_argument('--auto_resume', default= True, type=bool, help='start training from existing last saved model (Defult.torch)')
args = parser.parse_args()
if not os.path.exists(args.log_dir):os.mkdir(args.log_dir)
InitStep=0
if args.auto_resume:
    if os.path.exists(args.log_dir + "/Defult.torch"):
        args.resume_training_from=args.log_dir  + "/Defult.torch"
    if os.path.exists(args.log_dir +"/Learning_Rate.npy"):
        args.learning_rate=np.load(args.log_dir +"/Learning_Rate.npy")
    if os.path.exists(args.log_dir +"/itr.npy"): InitStep=int(np.load(args.log_dir +"/itr.npy"))
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
###########################################################################################################
# def Get_Loss(prd_bin_masks,high_res_masks, PointMasks,imgs,  PointsXY, NumGTMask):
#      with torch.no_grad():
#          high_res_masks=torch.tensor(prd_bin_masks, requires_grad=False, dtype=torch.float32).cuda()
#          GT_mask = torch.tensor(PointMasks, requires_grad=False, dtype=torch.float32).cuda()
#          bsize,num_msk,h,d = PointMasks.shape
#          inter = torch.einsum('yaij,ybij->yab', GT_mask, high_res_masks) #intersection
#       ###   inter = torch.einsum('aij,bij->ab', GT_mask[0], high_res_masks[0])
#          gt_sum=GT_mask.sum(2).sum(2)
#          gt_sum = torch.tile(gt_sum.unsqueeze(2), (1, 1, inter.shape[2]))
#          prd_sum=high_res_masks.sum(2).sum(2)
#          prd_sum=torch.tile(prd_sum.unsqueeze(1),[1,inter.shape[1],1])
#          iou=inter/(prd_sum+gt_sum-inter)
#          for i in range(bsize):
#              gt_indx,prd_indx=scipy.optimize.linear_sum_assignment(-iou[i].cpu().numpy())
#              print(iou[i])
#              for f,i in enumerate(gt_indx):
#                  print("gt->prd",i,"->",prd_indx[f] )
#              print(["----"])

###########################################################################################################


def Get_Loss_one_channel(high_res_masks, GTMasks, imgs, PointsXY):#, NumGTMask, prd_iou, mask_loss_type="substractive",unmatched_mask_loss=False, binary_iou=True, matched_IOU="best"):
    probs=torch.nn.functional.softmax(high_res_masks, dim=1)
    batch_size, num_msk, h, d = GTMasks.shape
    loss=0

    dtloss = {"total_loss": 0, "iou": 0}
    # liou=[]
    # lintr = []
    # for ib in range(batch_size):
    #     # create IOU matrix
    #     gt = torch.tensor(GTMasks[ib,0]>0.5, requires_grad=False, dtype=torch.float32).cuda()
    #     dtloss["total_loss"] -= (gt * torch.log(probs[ib,0] + 0.0001)).mean()
    #     dtloss["total_loss"] -= ((1 - gt) * torch.log((1-probs[ib,0]) + 0.0001)).mean()
    #
    #     gmask = gt.cpu().numpy()
    #     pmask = (probs[ib,0]>0.5).detach().cpu().numpy()
    #     inter= (gmask*pmask).sum()
    #     iou = inter/(gmask.sum()+pmask.sum()-inter+0.001)
    #     dtloss['iou']+=iou/batch_size
    #     liou.append(iou)
    #     lintr.append(inter)


    gt = torch.tensor(GTMasks[:, 0] > 0.5, requires_grad=False, dtype=torch.float32).cuda()
    dtloss["total_loss"] = - (gt * torch.log(probs[:,0] + 0.0001)    +     (1 - gt) * torch.log((1 - probs[:,0]) + 0.0001)).mean()
    gmask = gt.cpu().numpy()
    pmask = (probs[:,0]>0.5).detach().cpu().numpy()
    inter = (gmask * pmask).sum(1).sum(1)
    dtloss['iou'] = (inter/(gmask.sum(1).sum(1)+pmask.sum(1).sum(1)-inter+0.001)).mean()

    return  dtloss

    #                      dtloss["loss_iou"] -= ((1-gt_mask_list[f][gt2pr_list[f]['gt']]) * torch.log(1-high_res_masks[f][gt2pr_list[f]['pr']]+ 0.0001)).mean()

# def Get_Loss(prd_bin_masks, high_res_masks, GTMasks, imgs, PointsXY, NumGTMask,prd_iou,mask_loss_type = "substractive",unmatched_mask_loss=False,binary_iou=True, matched_IOU="best"):
#                     # '''
#                     # prd_bin_masks: thrsholds binary masks predictions
#                     # high_res_masks: high res probability predicted probablity  masks (0-1 sigmoid)
#                     # GTMasks: Ground true maps
#                     # imgs, PointsXY: images and pointer points for segmention
#                     # NumGTMask: number of GT masks
#                     # prd_iou: predicted iou
#                     # mask_loss_type = "crossentropy"/"substractive", Loss between to mask cross entropy or simple substractive
#                     # unmatched_mask_loss =True  , for unmatched predicted masks caclulate loss as an empty mask, else ignore unmatched mask
#                     # binary_iou=False , Calculate IOU match to binary (thresholds) masks
#                     # matched_IOU="hungarian" /"best"  # the GT IOU for given segment, best meaning its the IOU of the best match GT segment, hungarian mean its the IOU with the segment it was matched to in the hungarian matching
#                     # '''
#
#
#              #===Match mask to image using hungarian matching====================================================
#              with torch.no_grad():
#                 # high_res_masks = torch.tensor(prd_bin_masks, requires_grad=False, dtype=torch.float32).cuda()
#                 # GT_mask = torch.tensor(PointMasks, requires_grad=False, dtype=torch.float32).cuda()
#                  batch_size, num_msk, h, d = GTMasks.shape
#                  gt_mask_list=[]
#                  gt2pr_list=[]
#                  for ib in range( batch_size):
#                      # create IOU matrix
#                      gt= torch.tensor(GTMasks[ib][:NumGTMask[ib]], requires_grad=False, dtype=torch.float32).cuda()
#                   #   gt=GT_mask[ib][:NumGTMask[ib]]
#                      if binary_iou==True:
#                          prd_mask = torch.tensor(prd_bin_masks[ib], requires_grad=False, dtype=torch.float32).cuda() # binary mask for IOU
#                      else:
#                          prd_mask=high_res_masks[ib] # coninues mask for IOU
#                      inter = torch.einsum('aij,bij->ab', gt, prd_mask)  # intersection iou matrix per one single prediction
#
#                      ###   inter = torch.einsum('aij,bij->ab', GT_mask[0], high_res_masks[0]) # IOU matching for all batcg
#
#                      gt_sum = gt.sum(1).sum(1)
#                      gt_sum = torch.tile(gt_sum.unsqueeze(1), (1,inter.shape[1]))
#                      prd_sum = prd_mask.sum(1).sum(1)
#                      prd_sum = torch.tile(prd_sum.unsqueeze(0), [inter.shape[0], 1])
#                      iou = inter / (prd_sum + gt_sum - inter+0.0001)
#                      # hungarian  matching find which gt index match which prediction match index
#                      gt_indx, prd_indx = scipy.optimize.linear_sum_assignment(-iou.cpu().numpy())
#                      #try:
#                      if iou.shape[0]==0:
#                           print("what")
#                      print(torch.max(iou,axis=0))
#                      gt2pr_list.append({"gt":gt_indx,"pr":prd_indx,"best iou":torch.max(iou,axis=0),"iou mat":iou, "hungarian iou":iou[gt_indx,prd_indx]})
#                      # gt/prd_indx list of matching indexes for gt and prd masks in the hungarian matching
#                      #* note best iou is for each predition by order while  hungarian iou is for rhe corresponding indexes in gt_indx/prd_indx
#                      # except:
#                      #     print("$4")
#                      gt_mask_list.append(gt)
#                      print(iou)
#                      for f, i in enumerate(gt_indx):
#                          print("gt->prd", i, "->", prd_indx[f])
#              #=======Calculate loss============================================================================================================
#
#
#                  # for f in range(len(gt_mask_list)):
#                  #     for i1 in range(gt_mask_list[f].shape[0]):
#                  #         for i2 in range(high_res_masks[f].shape[0]):
#                  #             gtmsk = gt_mask_list[f][i1].cpu().numpy()
#                  #             prdmsk = high_res_masks[f][i2].cpu().numpy()
#                  #             im1=imgs[f].copy()
#                  #             im1[:,:,0][gtmsk>0.5]=255
#                  #             np.hstack([im1, imgs[f]]).astype(np.uint8)
#                  #             im2 = imgs[f].copy()
#                  #             im2[:, :, 1][prdmsk > 0.5] = 255
#                  #             cv2.destroyAllWindows()
#                  #             cim=np.hstack([im1,im2, imgs[f]]).astype(np.uint8)
#                  #             cv2.resize(cim,(int(im2.shape[1]/4),int(im2.shape[0]/4)))
#                  #             cv2.imshow("prd indx"+str(i2)+", gt ind:"+str(i1)+"iou="+str(gt2pr_list[f]['iou mat'][i1,i2].cpu().numpy()+0),cim)
#                  #             cv2.waitKey()
#
#                 # iou loss
#              # loss_iou=0
#              # loss_unmatched = 0
#              # eval_loss = 0
#              dtloss= {"total_loss": 0, "eval_loss": 0, "loss_unmatched": 0, "loss_iou": 0, "mean iou hungarian":0, "mean iou best":0, "mean iou error":0,"mean occupancy unmatched":0}
#
#              for f in range(batch_size):
#                  #===Calculate loss between matching ===============================
#                  if mask_loss_type=="crossentropy":
#                      dtloss["loss_iou"] -= (gt_mask_list[f][gt2pr_list[f]['gt']]*torch.log(high_res_masks[f][gt2pr_list[f]['pr']]+0.0001)).mean()
#                      dtloss["loss_iou"] -= ((1-gt_mask_list[f][gt2pr_list[f]['gt']]) * torch.log(1-high_res_masks[f][gt2pr_list[f]['pr']]+ 0.0001)).mean()
#                  elif  mask_loss_type ==  "substractive":
#                      dtloss["loss_iou"] += torch.abs(gt_mask_list[f][gt2pr_list[f]['gt']]-high_res_masks[f][gt2pr_list[f]['pr']]).mean()
#            ###      dtloss["loss_iou"] -= torch.log(1-high_res_masks[f][gt2pr_list[f]['pr']]+ 0.0001).mean()
#                  #loss_iou += -((gt_mask_list[f][gt2pr_list[f]['gt']] * torch.log(high_res_masks[f][gt2pr_list[f]['pr']]) + 0.0001).mean()
#                  #loss_iou += -(1-gt_mask_list[f][gt2pr_list[f]['gt'])# * torch.log((1-high_res_masks[f][gt2pr_list[f]['pr']] + 0.0001).mean()
#                  unmatched = torch.ones(high_res_masks[f].shape[0],dtype=torch.bool)  # binary list of all masks who are not masked
#                  unmatched[gt2pr_list[f]['pr']] = False # binary true/false list of unmatched
#                  if unmatched_mask_loss:
#                      if mask_loss_type == "crossentropy":
#                           dtloss["loss_unmatched"] -= torch.log(1-high_res_masks[f][unmatched]+0.001).mean()
#                      elif mask_loss_type == "substractive":
#                           dtloss["loss_unmatched"] += torch.abs(high_res_masks[f][unmatched]).mean()
#                      #gt_iou=gt2pr_list[f]["iou mat"][gt_indx[f]][prd_indx[f]]
#                  if matched_IOU=="hungarian":
#                      dtloss["eval_loss"] += torch.abs(gt2pr_list[f]["hungarian iou"]-prd_iou[f][gt2pr_list[f]['pr']]).mean() # iou error in matched samples
#                      dtloss["eval_loss"]+= torch.abs( prd_iou[f][unmatched]).mean() # iou error in unmatched
#                  elif matched_IOU == "best":
#                      gt_iou =gt2pr_list[f]["best iou"].values
#                      dtloss["eval_loss"] += torch.abs(prd_iou[f]-gt_iou).mean()
#                  dtloss["mean iou error"] += dtloss["eval_loss"].detach().cpu().numpy()/batch_size
#                  dtloss["mean iou hungarian"]+= gt2pr_list[f]["hungarian iou"].mean().detach().cpu().numpy()/batch_size
#                  dtloss["mean iou best"] += gt2pr_list[f]["best iou"].values.mean().detach().cpu().numpy()/batch_size
#                  dtloss["mean occupancy unmatched"]+=high_res_masks[f][unmatched].mean().detach().cpu().numpy()/batch_size
#                  #*****************************Display matches for loss****************************************************************************************
#                  # for uu in range(len(gt2pr_list[f]['gt'])):
#                  #     prdmsk = high_res_masks[f][gt2pr_list[f]['pr'][uu]].data.cpu().numpy()
#                  #     gtmsk = gt_mask_list[f][gt2pr_list[f]['gt'][uu]].data.cpu().numpy()
#                  #     GTiouHung = gt2pr_list[f]["hungarian iou"][uu].cpu().numpy()+0
#                  #     GTiouBest = gt2pr_list[f]["best iou"].values[gt2pr_list[f]['pr'][uu]].cpu().numpy()+0
#                  #     PRDiou = prd_iou[f][gt2pr_list[f]['pr'][uu]].detach().cpu().numpy()+0
#                  #     (x, y) = PointsXY[f]
#                  #
#                  #     im1 = imgs[f].copy()
#                  #     im1[:,:,0][gtmsk>0.5]=0
#                  #     im1[:, :, 1][gtmsk > 0.5] = 255
#                  #     im1[y-5:y+5,x-5:x+5]=0
#                  #     #np.hstack([im1, imgs[f]]).astype(np.uint8)
#                  #     im2 = imgs[f].copy()
#                  #     im2[:, :, 0][prdmsk > 0.5] = 255
#                  #     im2[:, :, 1][prdmsk > 0.5] = 0
#                  #     im2[y - 5:y + 5, x - 5:x + 5] = 255
#                  #     cv2.destroyAllWindows()
#                  #     cim=np.hstack([im1,im2, imgs[f]]).astype(np.uint8)
#                  #     cv2.resize(cim,(int(im2.shape[1]/4),int(im2.shape[0]/4)))
#                  #     cv2.imshow("prd iou"+str(PRDiou)+", gt iou best:"+str(GTiouBest)+" hungarian="+str(GTiouHung)+"gt max"+str(gtmsk.max()),cim)
#                  #     cv2.waitKey()
#
#                     #**************************************************************************************************************************************************
#        #      print(dtloss)
#              dtloss["total_loss"] = dtloss["loss_iou"]#dtloss["eval_loss"]*0.0+dtloss["loss_unmatched"]*0.0+dtloss["loss_iou"]
#              return dtloss

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
#sam_checkpoint = "sam_vit_h_4b8939.pth"

device = "cuda"
model_type = "default"
import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
if False:
    predictor.model.eval()
    sam = sam_model_registry[model_type](checkpoint=  args.sam_weight_path) # Build model all parts, load trained weight, set model to eval
    sam.to(device=device)
    predictor = SamPredictor(sam)

unet=unet.Net()
if args.resume_training_from!="": # Optional initiate full net
    unet.load_state_dict(torch.load(args.resume_training_from))
optimizer = torch.optim.AdamW(params= unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

#optimizer = torch.optim.AdamW(params= predictor.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
reader = MatSegReader.Reader(TrainDir=r"/media/breakeroftime/2T/Data_zoo/OutFolderMaterial_Segmentation/", MaxBatchSize=4,MinSize=300, MaxSize=1200, MaxPixels=800 * 800 * 4, TrainingMode=True)
InitStep=0
AVGLoss={}
#---------Main traininf loop----------------------------------------------------------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler() # For mixed precision
for itr in range(InitStep,1000000):
    print("--------------",itr,"-----------------------------------------------------------")
    print("---------start reading----------------")
    Imgs, ROIMask, PointMasks, AllMasks, PointsXY, NumPointMasks, NumAllMasks = reader.LoadBatch()

    print("---------Finished reading----------------")
    # for i in range(Imgs.shape[0]):
    #     reader.Display(img=Imgs[i], ROI=ROIMask[i], MatMask=PointMasks[i], txt=str(i) + " pointers only",x=PointsXY[i][0], y=PointsXY[i][1])
    # with torch.no_grad():
    #     # image_list=load_image_list(path=r"/media/breakeroftime/2T/Data_zoo/ChemPicsV1.0/Simple/Train/Image",im_size=[1000,1000])
    #     predictor.model.image_encoder.eval()
    #
    # # with torch.enable_grad():
    # #     predictor.model.requires_grad_(True)
    # #     predictor.model.train()
    #     # --------Image encoder------------------------------------------------------------------------------------------------------------
    #     predictor.set_multiple_image(Imgs)#image(image_list)
    #

    #--------Decoder with training------------------------------------------------------------------------------


    with torch.enable_grad():
        # input_point = np.array([[[500, 500]],[[500, 500]],[[500, 500]]])
        # input_label = np.array([[1],[1],[1]])


        # predictor.model.mask_decoder.requires_grad_(True)
        # predictor.model.mask_decoder.train()
        # predictor.model.prompt_encoder.requires_grad_(True)
        # predictor.model.prompt_encoder.train()

        # Create  optimizer
        #scaler = torch.cuda.amp.GradScaler()
        #with torch.cuda.amp.autocast():
            #predictor.model.train()
        with torch.cuda.amp.autocast():
            unet.zero_grad()
            logits = unet.forward(Imgs, ROIMask*0, PointsXY, TrainMode=True)
       # predictor.model.mask_decoder.zero_grad()
       #  bin_masks, iou_predictions, low_res_masks,high_res_masks = predictor.predict(
       #          point_coords=np.expand_dims(PointsXY,1),#input_point,
       #          point_labels=np.ones([PointsXY.shape[0],1]),#input_label,
       #          box=None,#input_box[None, :],
       #          multimask_output=True,)
       #  if torch.isnan(high_res_masks).any() == True:
       #      print("ga")

            loss = Get_Loss_one_channel(logits,PointMasks,Imgs, PointsXY)#, NumPointMasks)#,iou_predictions)
         #   print("loss",loss)


           # before2 = predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy() + 0
          #  before = predictor.model.mask_decoder.transformer.final_attn_token_to_image.q_proj.weight.sum().abs().detach().cpu().numpy() + 0
         #  #  print("before,", predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy())
         # #   loss=low_res_masks.mean().abs()#.requires_grad(True)
         #    loss["total_loss"].backward()
         #    optimizer.step()
            # -----------------------ba
            scaler.scale(loss["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

       # after2 = predictor.model.mask_decoder.output_upscaling[0].weight.sum().abs().detach().cpu().numpy() + 0
       # after = predictor.model.mask_decoder.transformer.final_attn_token_to_image.q_proj.weight.sum().abs().detach().cpu().numpy() + 0
#        print("before", before, "after", after, "dif", after - before)
        #========update statitics=============================================================================
        fr = 1 / np.min([itr - InitStep + 1, 2000])
        for ky in loss:
            if 'torch' in str(loss[ky]):  loss[ky] = loss[ky].data.cpu().numpy()
            if ky not in AVGLoss:
                if loss[ky] > 0:AVGLoss[ky]=loss[ky]
            else:
                if loss[ky] > 0: AVGLoss[ky] = AVGLoss[ky] * (1 - fr) + fr * float(loss[ky])  # Average loss
            if ky in AVGLoss: print(ky,":",AVGLoss[ky])

        # ===================save statitics and displaye loss======================================================================================
        # --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 200 == 0 and itr > 0:  # Save model weight and other paramters in temp file once every 1000 steps
            print("Saving Model to file in " + args.log_dir + "/Defult.torch")
            torch.save(unet.state_dict(), args.log_dir + "/Defult.torch")
            torch.save(unet.state_dict(), args.log_dir + "/DefultBack.torch")
            print("model saved")
            np.save(args.log_dir + "/Learning_Rate.npy", args.learning_rate)
            np.save(args.log_dir + "/itr.npy", itr)
            torch.cuda.empty_cache()  # clean memory
        if itr % 2000 == 0 and itr > 0:  # Save model weight once every 30k steps permenant (not temp)
            print("Saving Model to file in " + args.log_dir + "/" + str(itr) + ".torch")
            torch.save(unet.state_dict(), args.log_dir + "/" + str(itr) + ".torch")
            print("model saved")
        #==========================================================================================================

#
#-----------------------------------------------------------------------------------------------------------------------------
    input_point = np.expand_dims(PointsXY, 1)  # input_point,
    input_label = np.ones([PointsXY.shape[0], 1])  # input_label,
    # for i in range(Imgs.shape[0]):
    #     #image=image_list[i]
    #     image=Imgs[i].astype(np.uint8)
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #
    #
    #     show_points(input_point[i], input_label[i], plt.gca())
    #     plt.axis('on')
    #     plt.show()
    #     #
    #     # # input_box = np.array([425, 600, 700, 875])
    #     # plt.figure(figsize=(10, 10))
    #     # plt.imshow(image)
    #     for kk in range(bin_masks.shape[1]):
    #         show_mask(bin_masks[i][kk], plt.gca())
    #         show_points(input_point[i][0], input_label[i][0], plt.gca())
    #         plt.axis('off')
    #         plt.show()