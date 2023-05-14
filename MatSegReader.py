# Reader for the for annotation for object parts (data need to be generated using  the script at GenerateTrainingData
import numpy as np
import os
import random
import cv2
import json
import threading
import random


############################################################################################################
#########################################################################################################################
class Reader:
    # Initiate reader and define the main parameters for the data reader
    def __init__(self, TrainDir, MaxBatchSize=100, MinSize=500, MaxSize=2000, MaxPixels=800 * 800 * 5,TrainingMode=True, Suffle=False):

        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and hight in pixels
        self.MaxSize = MaxSize  # Max image width and hight in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.Epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        #    self.ClassBalance=ClassBalance
        self.Findx = 0
        #  self.MinCatSize=1
        # ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.annlist = []
        #      self.AnnotationByCat = {}

        # for i in range(NumClasses):
        #     self.AnnotationByCat.append([])

        # Ann_name.replace(".","__")+"##Class##"+str(seg["Class"])+"##IsThing##"+str(seg["IsThing"])+"IDNum"+str(ii)+".png"
        print("Creating file list for reader this might take a while")

        for uu, dname in enumerate(os.listdir(TrainDir)):
          #  if uu > 300: continue
            print(uu, dname)
            s = {}
            if os.path.exists(TrainDir + "/" + dname + "/Finished.txt"):
                s["dir"] = TrainDir + "/" + dname + "/"
                s["masks"] = []
                for fl in os.listdir(s["dir"]):
                    if fl == "RGB__RGB.jpg":  s["ImageFile"] = s["dir"] + "/" + fl
                    if fl[:4] == "mask":  s["masks"].append(s["dir"] + "/" + fl)
                    if fl == "ObjectMaskOcluded.png":  s["ROI"] = s["dir"] + "/" + fl

                if len(s.keys()) == 4 and len(s['masks']) > 0:
                    self.annlist.append(s)

        # # tt=0
        # # uu=0
        # # for i,ct in enumerate(self.AnnotationByCat):
        # #         print(str(i) + ")" + str(ct)+" "+str(len(self.AnnotationByCat[ct])))
        # #         if (ct % 7) == 0  or (ct % 9) == 0 == 0:
        # #             uu+=len(self.AnnotationByCat[ct])
        # #             tt+=1
        # #             self.AnnotationByCat[ct]=[]
        #
        if Suffle:
            np.random.shuffle(self.annlist)
        # print("All cats "+str(len(self.annlist)))
        # print("done making file list")
        # iii=0
        if TrainingMode: self.StartLoadBatch()
        # self.AnnData=False

    #############################################################################################################################
    # Crop and resize image and mask and Object mask to feet batch size
    def CropResize(self, Img, MatMasks, ROImask, Hb, Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox = cv2.boundingRect(ROImask.astype(np.uint8))
        [h, w, d] = Img.shape
        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Wbox == 0: Wbox += 1
        if Hbox == 0: Hbox += 1

        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or Bs < 1 or np.random.rand() < 0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in range(len(MatMasks)):
                MatMasks[i] = cv2.resize(MatMasks[i].astype(float), dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            ROImask = cv2.resize(ROImask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float32)).astype(np.int32)

        # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox) - 1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb) + 1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox) - 1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb) + 1))

        if Ymax <= Ymin:
            y0 = Ymin
        else:
            y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax <= Xmin:
            x0 = Xmin
        else:
            x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=PartMask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        for i in range(len(MatMasks)):
            MatMasks[i] = MatMasks[i][y0:y0 + Hb, x0:x0 + Wb]
        ROImask = ROImask[y0:y0 + Hb, x0:x0 + Wb]
        # ------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            for i in range(len(MatMasks)):
                MatMasks[i] = cv2.resize(MatMasks[i][y0:y0 + Hb, x0:x0 + Wb, :], interpolation=cv2.INTER_LINEAR)
            ROImask = cv2.resize(ROImask, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img, MatMasks, ROImask
        # misc.imshow(Img)

    #################################################Generate Annotaton mask#############################################################################################################333
    #################################################Generate Pointer mask#############################################################################################################333
    # Return point and the masks corresponding to this point
    def GeneratePointer(self, MatMasks, ROIMask, use_eqaual_prob=0.6, ocupancy_thresh=0.5,max_seg_per_points=1):
        for attemps in range(200):
        #MatMasks# Segment masks
        #ROIMask #roi masks
        # use_eqaual_prob = 0.6 probablity for picking segments with equal probability,
        # ocupancy_thresh = 0.5, # minimimum accupany for point to be used for selection
        # max_seg_seg_per_point maximum number of segments belonging to a point
            if np.random.rand() > use_eqaual_prob:  # pick random point from ROI
                point_list = np.where(ocupancy_thresh > 0.9)
            else:
                indx = np.random.randint(0, MatMasks.shape[0])
                point_list = np.where(MatMasks[indx] >0.9)
            if len(point_list) == 0 or  point_list[0].__len__() < 40: return False, None, None, None
            for xx in range(100):
                pind = np.random.randint(0, point_list[0].__len__())
                y = point_list[0][pind]
                x = point_list[1][pind]
                masksPoints = MatMasks[MatMasks[:, y, x] > ocupancy_thresh] # masks belonging to the point
                if masksPoints.shape[0]<=max_seg_per_points and  masksPoints.shape[0]>=1: break # if point match critetrion break else pick again
            if masksPoints.shape[0] <= max_seg_per_points and masksPoints.shape[0] >= 1: break# if point match critetrion break else pick again
            if attemps>50: return False,False,False # if couldnt read point that match criterion  return false
        return True, masksPoints, y, x

    ######################################################Augmented mask##################################################################################################################################
    def Augment(self, Img, MatMasks, ROIMask, prob=1):
        Img = Img.astype(np.float32)
        if np.random.rand() < 0.5:  # flip left right
            Img = np.fliplr(Img)
            ROIMask = np.fliplr(ROIMask)
            for i in range(len(MatMasks)):
                MatMasks[i] = np.fliplr(MatMasks[i])

        if np.random.rand() < 0.5:  # flip up down
            Img = np.flipud(Img)
            ROIMask = np.flipud(ROIMask)
            for i in range(len(MatMasks)):
                MatMasks[i] = np.flipud(MatMasks[i])
        #
        # if np.random.rand() < prob: # resize
        #     r=r2=(0.6 + np.random.rand() * 0.8)
        #     if np.random.rand() < prob*0.2:  #Strech
        #         r2=(0.65 + np.random.rand() * 0.7)
        #     h = int(PartMask.shape[0] * r)
        #     w = int(PartMask.shape[1] * r2)
        #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        #     PartMask = cv2.resize(PartMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        #     AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        if np.random.rand() < 0.035:  # Add noise
            noise = np.random.rand(Img.shape[0], Img.shape[1], Img.shape[2]) * 0.2 + np.ones(Img.shape) * 0.9

            Img *= noise
            Img[Img > 255] = 255
            #
        if np.random.rand() < 0.2:  # Gaussian blur
            Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < 0.25:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img > 255] = 255
        # if np.random.rand() < prob:  # Dark light
        #     Img = Img * (0.5 + np.random.rand() * 0.7)
        #     Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr = Img.mean(axis=2)
            r = np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)

        return Img, MatMasks, ROIMask

    #######################################################################################################################################################
    def Display(self, img, MatMask, ROI, txt="", x=-1, y=-1):
        img_cat = img.copy()
        tmp = img.copy()
        tmp[:, :, 0][ROI < 0.2] = 255
        tmp[:, :, 1][ROI < 0.2] = 0
        img_cat = np.concatenate([img_cat, tmp], 1)
        if type(MatMask) == list:
            for msk in MatMask:
                if msk.sum() == 0: continue
                tmp = img.copy()
                tmp[:, :, 0][msk > 0.2] = 255
                tmp[:, :, 1][msk > 0.2] = 0
                img_cat = np.concatenate([img_cat, tmp], 1)
        else:
            for i in range(MatMask.shape[0]):
                tmp = img.copy()
                msk = MatMask[i]
                if msk.sum() == 0: continue
                tmp[:, :, 0][msk > 0.2] = 255
                tmp[:, :, 1][msk > 0.2] = 0
                if x > -1: tmp[y - 10:y + 10, x - 10:x + 10] = (0, 255, 255)
                img_cat = np.concatenate([img_cat, tmp], 1)
        h, w, d = img_cat.shape
        r = 1500 / w
        h = int(h * r)
        w = int(w * r)
        cv2.destroyAllWindows()
        cv2.imshow(txt, cv2.resize(img_cat, (w, h)).astype(np.uint8))
        cv2.waitKey()

    #######################################################################################################################################################\

    # Normalized

    #########################################################################################################################################################
    def Normalize(self, MatMask, ROImask):
        ROImask = (ROImask>30).astype(np.float32)
        for i, msk in enumerate(MatMask):
            MatMask[i] = MatMask[i].astype(np.float32) / 255
        return MatMask, ROImask

    ########################################################################################################################################################
    # ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
        # if self.ClassBalance: # pick with equal class probability
        #     while (True):
        #          CL=random.choice(list(self.AnnotationByCat))
        #          if not (CL in self.AnnotationByCat): continue
        #          CatSize=len(self.AnnotationByCat[CL])
        #          if CatSize>=self.MinCatSize: break
        #     Nim = np.random.randint(CatSize)
        #    # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
        #     Ann=self.AnnotationByCat[CL][Nim]
        # else: # Pick with equal probability per annotation
        while(True):
           Nim = np.random.randint(len(self.annlist))
           if np.random.rand()>2/len(self.annlist[Nim]): break
        Ann = self.annlist[Nim]
        # CatSize=100000000
        # --------------Read image--------------------------------------------------------------------------------
        Img = cv2.imread(Ann["ImageFile"])  # Load Image
        Img = Img[..., :: -1]
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        # -------------------------Read annotation--------------------------------------------------------------------------------
        MatMasks = []
        for msk_path in Ann['masks']:
            mastmsk = cv2.imread(msk_path, 0).astype(float)
            MatMasks.append(mastmsk.astype(float))
        ROIMask = cv2.imread(Ann["ROI"], 0)  # Load mask

        MatMasks, ROIMask = self.Normalize(MatMasks, ROIMask)

        # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
        #   self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask,txt="raw")
        if not Hb == -1:
            Img, MatMasks, ROIMask = self.CropResize(Img, MatMasks, ROIMask, Hb, Wb)
        # -------------------------Augment-----------------------------------------------------------------------------------------------
        #  self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="cropped")
        if np.random.rand() < 10.62:
            Img, MatMasks, ROIMask = self.Augment(Img, MatMasks, ROIMask)
        # self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="augmented")
        # -------------------------Turn Masks list into numpy matrix---------------------------------------------------------------------------------------------------------
        # Find the non-zero matrices
        non_zero_matrices = [matrix for matrix in MatMasks if not np.all(matrix < 0.1)]
        if len(non_zero_matrices) == 0:
            return self.LoadNext(batch_pos, Hb, Wb)
        # Stack the non-zero matrices into a 3D array
        MatMasks = np.stack(non_zero_matrices, axis=0)  # stack along last axis

        #self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="stacked")
        # ----------------Select pointer points------------------------------------------------------------------------------
        Succssess, PointerMasks, y, x = self.GeneratePointer(MatMasks=MatMasks, ROIMask=ROIMask)
        if not Succssess or PointerMasks.shape[0]==0:
            return self.LoadNext(batch_pos, Hb, Wb)
        #self.Display(img=Img, MatMask=PointerMasks, ROI=ROIMask, txt="points",x=x,y=y)
        #   self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="points", x=x, y=y)
        # ---------------------------------------------------------------------------------------------------------------------------------
        self.BROIMask[batch_pos] = ROIMask
        self.BImgs[batch_pos] = Img
        self.BAllMasks[batch_pos][:MatMasks.shape[0]] = MatMasks[:self.BAllMasks[batch_pos].shape[0]]
        self.BPointMasks[batch_pos][:PointerMasks.shape[0]] = PointerMasks[:self.BPointMasks[batch_pos].shape[0]]
        self.BPointsXY[batch_pos] = [x, y]
        self.BNumPointMasks[batch_pos] = PointerMasks.shape[0]
        self.BNumAllMasks[batch_pos] = MatMasks.shape[0]

    #  self.BCatID[batch_pos] = Ann["Class"]

    ############################################################################################################################################################
    # Start load batch of images, segment masks, ROI masks, and pointer points for training MultiThreading s
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb * Wb < self.MaxPixels: break
        BatchSize = np.int32(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        #  BatchSize=1
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  #
        self.BROIMask = np.zeros((BatchSize, Hb, Wb))
        self.BPointMasks = np.zeros((BatchSize, 5, Hb, Wb))
        self.BAllMasks = np.zeros((BatchSize, 10, Hb, Wb))
        self.BPointsXY = np.zeros((BatchSize, 2), dtype=np.uint32)
        self.BNumPointMasks = np.zeros((BatchSize), dtype=np.uint32)
        self.BNumAllMasks = np.zeros((BatchSize), dtype=np.uint32)
        # ===============Select images for next batch
        # ====================Start reading data multithreaded===========================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th = threading.Thread(target=self.LoadNext, name="thread" + str(pos), args=(pos, Hb, Wb))
            self.thread_list.append(th)
            th.start()
        self.itr += BatchSize

    ##################################################################################################################
    def SuffleFileList(self):
        random.shuffle(self.FileList)
        self.itr = 0

    ###########################################################################################################
    # Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
        for th in self.thread_list:
            th.join()

    ########################################################################################################################################################################################
    def LoadBatch(self):
        # Load batch for training (muti threaded  run in parallel with the training proccess)
        # For training
        self.WaitLoadBatch()

        Imgs = self.BImgs
        ROIMask = self.BROIMask
        PointMasks = self.BPointMasks
        AllMasks = self.BAllMasks
        PointsXY = self.BPointsXY
        NumPointMasks = self.BNumPointMasks
        NumAllMasks = self.BNumAllMasks
        self.StartLoadBatch()
        return Imgs, ROIMask, PointMasks, AllMasks, PointsXY, NumPointMasks, NumAllMasks

    ############################################################################################################################################################
    def Reset(self):
        self.Cindx = int(0)
        self.Findx = int(0)
        self.CindList = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Clepoch = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Epoch = int(0)  # not valid or


########################################

if __name__ == "__main__":
    print("G")
    read = Reader(TrainDir=r"/media/breakeroftime/2T/Data_zoo/OutFolderMaterial_Segmentation/", MaxBatchSize=100,
                  MinSize=250, MaxSize=1000, MaxPixels=800 * 800 * 20, TrainingMode=True)
    Imgs, ROIMask, PointMasks, AllMasks, PointsXY = read.LoadBatch()
    for i in range(Imgs.shape[0]):
        read.Display(img=Imgs[i], ROI=ROIMask[i], MatMask=PointMasks[i], txt=str(i) + " pointers only",x=PointsXY[i][0], y=PointsXY[i][1])




