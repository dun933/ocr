import os
pwd = os.getcwd()
opencvFlag = 'keras' 
IMGSIZE = (608,608)
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none','text',]
kerasTextModel=os.path.join(pwd,"models","text.h5")

darknetRoot = os.path.join(os.path.curdir,"darknet")
yoloCfg     = os.path.join(pwd,"models","text.cfg")
yoloWeights = os.path.join(pwd,"models","text.weights")
yoloData    = os.path.join(pwd,"models","text.data")


GPU = True
GPUID=0

nmsFlag='gpu'
if not GPU:
    nmsFlag='cython'


DETECTANGLE=True
AngleModelPb = os.path.join(pwd,"models","Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd,"models","Angle-model.pbtxt")




LSTMFLAG = True
ocrFlag = 'torch'
chinsesModel = True
ocrModelKeras = os.path.join(pwd,"models","ocr-dense-keras.h5")
if chinsesModel:
    if LSTMFLAG:
        ocrModel  = os.path.join(pwd,"models","ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd,"models","ocr-dense.pth")
else:
        LSTMFLAG=True
        ocrModel = os.path.join(pwd,"models","ocr-english.pth")
