# Model and Loss functions
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self): # Load pretrained encoder and prepare net layers
            self.skip_con_stitch_mode = "add" #"add"
            self.transpose_mode = True
            self.features2keep = {

                 2: {"layer": 5, "downscale": 16, "upscale": 2}
                ,4: {"layer": 3, "downscale": 8, "upscale": 4}
                ,8: {"layer": 1, "downscale": 4, "upscale": 8}

            } # features to keep for the skip connection
            self.upscaling_layers = {

                 2: {"upscale": 2, "indepth": 1024, "outdepth": 512, "skip_depth": 512}
                ,4: {"upscale": 4, "indepth": 512, "outdepth": 256, "skip_depth": 256}
                ,8: {"upscale": 4, "indepth": 256, "outdepth": 128, "skip_depth": 128}
            }

            self.build_encoder()
            self.build_decoder()
            self.build_upsample()
            self.logits_layer = nn.Conv2d(128, 2, stride=1, kernel_size=3, padding=1, bias=False)

################################################################################################################################################################
        def build_encoder(self):
            super(Net, self).__init__()
# ---------------Load pretrained net----------------------------------------------------------
            self.Encoder1 = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)#resnet50(pretrained=True)
           # self.Encoder1 = models.convnext_large(weights=True)  # resnet50(pretrained=True)
            # weight_shape = torch.tensor(self.Encoder1.features[0][0].weight.shape)
            # weight_shape[1] = 1
            # mask_weight=torch.zeros(tuple(weight_shape.numpy()))
            # total_weight= torch.cat([self.Encoder1.features[0][0].weight.data, mask_weight], 1)

#--------------Replace First layer from 3 channel input (RGB) to 4 channel (RGB,ROI)
            old_bias = copy.deepcopy(self.Encoder1.features[0][0].bias.data)
            old_weight = copy.deepcopy(self.Encoder1.features[0][0].weight.data)
            self.Encoder1.features[0][0]= torch.nn.Conv2d(5, 128, kernel_size=(4, 4), stride=(4, 4)) # Add layers to masks and pointer point
            self.Encoder1.features[0][0].weight.data[:,:3,:,:] = old_weight
            self.Encoder1.features[0][0].weight.data[:, 3, :, :] = 0
            self.Encoder1.features[0][0].bias.data = old_bias
            print("new_bias", self.Encoder1.features[0][0].bias.data.sum())
            print("new_weight", self.Encoder1.features[0][0].weight.data.sum())
#----------------Change final layer to predict 512 descriptor------------------------------------------------------------------------------------------
            #self.Encoder1.fc=nn.Sequential(nn.Linear(2048, 512),nn.ReLU())
            #self.Encoder1.classifier[2]=torch.nn.Linear(in_features=1024, out_features=512, bias=True)
 ########################################################################################################################################
        def build_decoder(self,mode="aspp"):
            self.mode = "aspp"
            if self.mode=="psp":
            # ---------------------------------PSP layer----------------------------------------------------------------------------------------
                self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
                self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]  # scalesPyramid Scene Parsing PSP layer
                for Ps in self.PSPScales:
                    self.PSPLayers.append(nn.Sequential(
                        nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(512),
                        ##nn.nn.LayerNorm((512), eps=1e-06),
                        nn.GELU()))
            # ----------------------------------------ASPP  deeplab layers-----------------------------------------------------------------------
            elif  self.mode=="aspp":
                self.ASPPLayers = nn.ModuleList()
                self.ASPPScales = [1, 4, 8, 16]  # scales ASPP deep lab 3 net
                for scale in self.ASPPScales:
                    self.ASPPLayers.append(nn.Sequential(
                        nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=(scale, scale), dilation=(scale, scale),bias=False),
                        nn.BatchNorm2d(512),
                        ##nn.LayerNorm((512), eps=1e-06),
                        nn.GELU()))
                        #, nn.BatchNorm2d(512), nn.GELU()))


            # -------------------------------------------------------------------------------------------------------------------
            self.SqueezeLayers = nn.Sequential(
                nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU())  # ,
                # nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(512),
                # nn.ReLU()
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------

 ###############################################################################################################################


        def build_upsample(self):




            self.upsample_ModuleList = nn.ModuleList()


            for i in  self.upscaling_layers:
                dt= self.upscaling_layers[i]
                if self.transpose_mode:
                    layer = nn.Sequential(
                                      nn.ConvTranspose2d(dt["indepth"], dt["outdepth"], 4, stride=2),
                                      nn.BatchNorm2d(dt["outdepth"]),
                                      #nn.LayerNorm((512,), eps=1e-06),
                                      nn.GELU())
                    self.upscaling_layers[i]["upscale_layer"] =   layer
                    self.upsample_ModuleList.append(layer)

                                      # nn.BatchNorm2d(512),
                else:
                    layer =  nn.Sequential(
                    nn.Conv2d(dt["indepth"], dt["outdepth"], stride=1, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dt["outdepth"]),
                    #nn.LayerNorm((dt["outdepth"],), eps=1e-06),# 1e-05
                    #nn.BatchNorm2d(512),
                    nn.GELU())
                    self.upscaling_layers[i]["upscale_layer"] = layer
                    self.upsample_ModuleList.append(layer)

                skiplayer = nn.Sequential(
                    nn.Conv2d(dt["skip_depth"], dt["outdepth"], stride=1, kernel_size=1, padding=0, bias=False),
                    #nn.BatchNorm2d(512),
                    nn.BatchNorm2d(dt["outdepth"]),
                  #  nn.LayerNorm((dt["outdepth"],), eps=1e-06),
                    nn.GELU())
                self.upscaling_layers[i]["skip_layer"] = skiplayer
                self.upsample_ModuleList.append(skiplayer)

            #    if self.skip_con_stitch_mode == "cat":


#######################################Pre process  input#############################################################################################3
        def preproccess_input(self,Images, Mask,PointerXY,TrainMode=True):
            # ------------------------------- Convert from numpy to pytorch-------------------------------------------------------
            if TrainMode:
                mode = torch.FloatTensor
            else:
                mode = torch.half

            self.type(mode)
            InpImages1 = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,
                                                                                                          3).transpose(
                1, 2).type(mode)
            ROIMask1 = torch.autograd.Variable(torch.from_numpy(Mask.astype(np.float32)),
                                               requires_grad=False).unsqueeze(dim=1).type(mode)

            pointer_mask = np.zeros_like(Mask)
            x, y = PointerXY[:, 0].astype(np.int32), PointerXY[:, 1].astype(np.int32)
            pointer_mask[:, y, x] = 1
            ###########################################################################
            # for ii in range(Mask.shape[0]):
            #     im=Images[ii].copy().astype(np.uint8)
            #     im[:,:,0][pointer_mask[ii]>0]=0
            #     im[y[ii]-5:y[ii]+5, x[ii]-5:x[ii]+5, 1] = 255
            #     im[:, :, 2][Mask[ii] > 0] = 0
            #     cv2.imshow(str(ii),np.hstack([im,Images[ii].astype(np.uint8)]))
            #     cv2.waitKey()

            #########################################################################
            pointer_mask = torch.autograd.Variable(torch.from_numpy(pointer_mask.astype(np.float32)),
                                                   requires_grad=False).unsqueeze(dim=1).type(mode)

            InpImages1 = InpImages1.to(device)
            ROIMask1 = ROIMask1.to(device)
            pointer_mask = pointer_mask.to(device)
            self.to(device)

            # -------------------------Normalize image-------------------------------------------------------------------------------------------------------
            RGBMean = [123.68, 116.779, 103.939]
            RGBStd = [65, 65, 65]
            for i in range(len(RGBMean)):
                InpImages1[:, i, :, :] = (InpImages1[:, i, :, :] - RGBMean[i]) / RGBStd[
                    i]  # Normalize image by std and mean
            # ============================Run net layers===================================================================================================
            inp_concat = torch.cat([InpImages1, ROIMask1, pointer_mask], 1)
            return inp_concat
###########################################Run encoder########################################################################################333
        def forward_encoder(self,x):
           for i in range(len(self.Encoder1.features)):
               x = self.Encoder1.features[i](x)
               for f in self.features2keep:
                   if self.features2keep[f]["layer"] == i:
                       self.features2keep[f]["features"] = x
               print(i, ")", x.shape)
           return x
###########################################Run encoder########################################################################################333
        def forward_midlayer(self,x):
            if self.mode == "psp":
                PSPSize = (x.shape[2], x.shape[3])  # Size of the original features map
                PSPFeatures = []  # Results of various of scaled procceessing
                for i, PSPLayer in enumerate(
                        self.PSPLayers):  # run PSP layers scale features map to various of sizes apply convolution and concat the results
                    NewSize = (np.array(PSPSize) * self.PSPScales[i]).astype(np.int32)
                    if NewSize[0] < 1: NewSize[0] = 1
                    if NewSize[1] < 1: NewSize[1] = 1

                    # print(str(i)+")"+str(NewSize))
                    y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear', align_corners=False)
                    # print(y.shape)
                    y = PSPLayer(y)
                    y = nn.functional.interpolate(y, PSPSize, mode='bilinear', align_corners=False)
                    PSPFeatures.append(y)
                x = torch.cat(PSPFeatures, dim=1)
                x = self.SqueezeLayers(x)
            elif self.mode == "aspp":

                # ---------------------------------ASPP Layers--------------------------------------------------------------------------------
                ASPPFeatures = []  # Results of various of scaled procceessing
                for ASPPLayer in self.ASPPLayers:
                    y = ASPPLayer(x)
                    ASPPFeatures.append(y)
                x = torch.cat(ASPPFeatures, dim=1)
                x = self.SqueezeLayers(x)
            return x
###############################################Upsampling forward#########################################################################################################
        def forward_upsample(self,x):
            for ii in range(1, 12):
                if ii in self.upscaling_layers:
                    if "upscale_layer" in self.upscaling_layers[ii]:
                        x = self.upscaling_layers[ii]["upscale_layer"](x)
                        if (ii in self.features2keep) and ("skip_layer" in self.upscaling_layers[ii]):
                            y = self.upscaling_layers[ii]["skip_layer"](self.features2keep[ii]["features"])
                            if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
                                print("inconsistant upsampling scale")
                                x = nn.functional.interpolate(x, size=(y.shape[2], y.shape[3]), mode='bilinear',align_corners=False)
                                if self.skip_con_stitch_mode == "add":
                                    x += y
                                elif self.skip_con_stitch_mode == "cat":
                                    x = torch.cat((y, x), dim=1)
            return x

###############################################Run prediction inference using the net ###########################################################################################################



        def forward(self,Images, Mask,PointerXY,TrainMode=True):
                x = self.preproccess_input(Images, Mask,PointerXY,TrainMode=True)
                x = self.forward_encoder(x)
                x = self.forward_midlayer(x)
                x = self.forward_upsample(x)
                logits = self.logits_layer(x)
                logits_upsample = nn.functional.interpolate(logits, size=(Images.shape[1], Images.shape[2]), mode='bilinear', align_corners=False)
                return logits_upsample
if __name__ == "__main__":
    net=Net()