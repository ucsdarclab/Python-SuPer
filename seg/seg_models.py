import os
import numpy as np
import cv2

import torch.nn.functional as F

from utils.utils import *


border_range = [50, 550]


def file_as_index(fname):
    x = 0
    if 'l' in fname:
        x = 10000
    fname = fname.strip('-viewerleftright.png')
    return int(fname) + x


def HSVSeg(rgb, tool):
    # table:0, chicken:1, beef:2, tool:3
    seg_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    tool = torch_to_numpy(tool).astype('bool')
    seg_mask[np.invert(tool)] = 3
    
    rgb = 255*de_normalize(rgb)
    hsv = cv2.cvtColor(
        torch_to_numpy(rgb.permute(1,2,0)).astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    low = np.array([0, 100, 0])
    high = np.array([255, 255, 255])
    
    table = cv2.inRange(hsv, low, high)
    protect_border_mask = np.zeros([480, 640])
    corners = np.array([[[40, 0], [550, 0], [550, 480], [50, 480], [0, 400], [0, 50]]])
    cv2.fillPoly(protect_border_mask, corners, (255, 255, 255))
    table = table.astype('bool') | protect_border_mask.astype('bool')

    table = np.invert(table)
    contours, hier = cv2.findContours(table.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0:3]

    table = np.zeros((480, 640)).astype('uint8')
    cv2.drawContours(table, contour, -1, 255, -1)

    table = np.invert(table)
    contours, hier = cv2.findContours(table.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0:1]

    table = np.zeros((480, 640)).astype('uint8')
    cv2.drawContours(table, contour, -1, 255, -1)

    seg_mask[np.invert(table.astype('bool'))] = 0

    low = np.array([8, 0, 145])
    high = np.array([256, 256, 256])

    chicken = cv2.inRange(hsv, low, high)
    contours, hier = cv2.findContours(chicken.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0:1]

    chicken = np.zeros((480, 640)).astype('uint8')
    cv2.drawContours(chicken, contour, -1, 255, -1)

    chicken = np.invert(chicken)
    contours, hier = cv2.findContours(chicken.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0:2]

    chicken = np.zeros((480, 640)).astype('uint8')
    cv2.drawContours(chicken, contour, -1, 255, -1)

    seg_mask[np.invert(chicken.astype('bool'))] = 1

    beef = tool & table & chicken

    seg_mask[beef.astype('bool')] = 2

    # cv2.imwrite("test_seg.jpg", seg_mask*50)

    seg_conf = F.one_hot(numpy_to_torch(seg_mask).type(long_)).type(fl32_)

    return seg_conf