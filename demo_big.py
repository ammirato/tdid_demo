# coding: utf-8

# An example using startStreams

import torch
import torch.utils.data
import torchvision.models as models
import importlib

from model_defs.TDID import TDID 
from utils import *
from model_defs.nms.nms_wrapper import nms

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

import matplotlib.pyplot as plt

logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(None)
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)



def get_target_image(listener):
    accept = 'n'
    while accept!='y':
        while True:
            frames = listener.waitForNewFrame()
            color = frames["color"]
            cv2.imshow("color", cv2.resize(color.asarray(),
                                           (int(1920 / 3), int(1080 / 3))))
            listener.release(frames)

            key = cv2.waitKey(delay=1)
            if key == ord('t'):
                break

        target_img = color.asarray().copy()


        #crop around target object
        target_img = target_img[:,:,:3]
        img = target_img.copy() 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)

        dil_k = np.ones((75,75))
        erd_k = np.ones((85,85))
        dilated = cv2.dilate(edges,dil_k)
        img = cv2.erode(dilated,erd_k)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
        target_area = np.sort(stats[:,-1])[-2]
        target_box = []
        for box in stats:
            if box[-1] == target_area:
                target_box = box[:-1]

        offsets = np.asarray([-50,-50,125,125])
        box = target_box + offsets
        box[0] = max(0,box[0])
        box[1] = max(0,box[1])
        box[2] = min(img.shape[1], box[0]+box[2])
        box[3] = min(img.shape[0], box[1]+box[3])

        target_img = target_img[box[1]:box[3], box[0]:box[2],:]
        
        scale_factor = 80.0/np.min(target_img.shape[:2])
        target_img = cv2.resize(target_img,(int(target_img.shape[1]*scale_factor),
                                            int(target_img.shape[0]*scale_factor)))
        #accept = raw_input('Accept target image?(y/n): ')
        accept = 'y'

    return target_img






def detect(net, target_data,im_data, im_info,score_thresh=.01, features_given=True):
    """ 
    Detect single target object in a single scene image.

    Input Parameters:
        net: (TDID) the network
        target_data: (torch Variable) target images
        im_data: (torch Variable) scene_image
        im_info: (tuple) (height,width,channels) of im_data
        
        features_given(optional): (bool) if true, target_data and im_data
                                  are feature maps from net.features,
                                  not images. Default: True
                                    

    Returns:
        scores (ndarray): N x 2 array of class scores
                          (N boxes, classes={background,target})
        boxes (ndarray): N x 4 array of predicted bounding boxes
    """

    cls_prob, rois = net(target_data, im_data, im_info,
                                    features_given=features_given)
    scores = cls_prob.data.cpu().numpy()[0,:,:]
    zs = np.zeros((scores.size, 1)) 
    scores = np.concatenate((zs,scores),1)
    boxes = rois.data.cpu().numpy()[0,:, :]


    #get scores for foreground, non maximum supression
   
    inds = np.where(scores[:, 1] > score_thresh)[0]
    fg_scores = scores[inds, 1]
    fg_boxes = boxes[inds,:]
    fg_dets = np.hstack((fg_boxes, fg_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
    keep = nms(fg_dets, cfg.TEST_NMS_OVERLAP_THRESH)
    fg_dets = fg_dets[keep, :]

    max_dets_per_target = 5 
    image_scores = np.hstack([fg_dets[:, -1]])
    if len(image_scores) > max_dets_per_target:
        image_thresh = np.sort(image_scores)[-max_dets_per_target]
        keep = np.where(fg_dets[:, -1] >= image_thresh)[0]
        fg_dets = fg_dets[keep, :]

#    if len(fg_dets) > 0:
#        box = fg_dets[0]
#    else:
#        box = None
    return fg_dets 















########################################################
# SETUP CAMERA #
########################################################

enable_rgb = True
enable_depth = False 

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.startStreams(rgb=enable_rgb, depth=enable_depth)


########################################################
#   Load Model #
########################################################
# load config
cfg_file = 'configAVD2' #NO FILE EXTENSTION!
cfg = importlib.import_module('configs.'+cfg_file)
cfg = cfg.get_config()

print('Loading ' + cfg.FULL_MODEL_LOAD_NAME + ' ...')
net = TDID(cfg)
load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
net.features.eval()#freeze batchnorms layers?
print('load model successfully!')

net.cuda()
net.eval()



########################################################
#   GET TARGET IMAGES #
########################################################

target_0 = get_target_image(listener)
target_1 = get_target_image(listener)
#target_0 = cv2.imread('./t0.jpg')
#target_1 = cv2.imread('./t1.jpg')

target_display = match_and_concat_images_list([target_0,
                                           target_1])
target_display = np.vstack((target_display[0,:,:,:],target_display[1,:,:,:]))


########################################################
# RUN DETECTION #
########################################################

#prepare target images for detection network
#target_img = np.stack([normalize_image(target_0,cfg),
#                       normalize_image(target_1,cfg)],
#                      axis=0)
target_img = match_and_concat_images_list([normalize_image(target_0,cfg),
                       normalize_image(target_1,cfg)])
#target_display = np.vstack((target_img[0,:,:,:],target_img[1,:,:,:]))
cv2.imshow('target images', target_display.astype(np.uint8))
target_img = np_to_variable(target_img, is_cuda=True)
target_img = target_img.permute(0, 3, 1, 2)

while True:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    if color.asarray().shape[0] < 2:
        continue
    #cv2.imshow("color", cv2.resize(color.asarray(),
    #                               (int(1920 / 1), int(1080 / 1))))
    listener.release(frames)

    scene_img = cv2.resize(color.asarray(),
                           (int(1920 / 2), int(1080 / 2)))
    display_img =color.asarray().copy()# scene_img.copy()
    scene_img = scene_img[:,:,:3]
    im_info = scene_img.shape[:]
    scene_img = normalize_image(scene_img,cfg)
    scene_img = np_to_variable(scene_img, is_cuda=True)
    scene_img = scene_img.unsqueeze(0)
    scene_img = scene_img.permute(0, 3, 1, 2)


    boxes = detect(net, target_img, scene_img, im_info,
                          features_given=False)
    print boxes
    for box in boxes:
        cv2.rectangle(display_img,(int(box[0]*2),int(box[1]*2)),(int(box[2]*2),int(box[3]*2)),(255,0,0),4)
    if len(boxes) > 0:
        box = boxes[-1]
        cv2.rectangle(display_img,(int(box[0]*2),int(box[1]*2)),(int(box[2]*2),int(box[3]*2)),(0,0,255),5)

    cv2.imshow("color",display_img)
    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break





########################################################
# END DEMO #
########################################################
device.stop()
device.close()

sys.exit(0)
