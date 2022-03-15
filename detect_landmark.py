# -*- coding: UTF-8 -*-
import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import copy
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_corner, scale_coords, xyxy2xywh, increment_path, \
    find_T_matrix, non_max_suppression_face
from utils.torch_utils import time_synchronized

lp_transform_dest = [None, np.float32([[0, 0], [470, 0], [470, 110], [0, 110]]),
                     np.float32([[0, 0], [600, 0], [600, 330], [0, 330]])]
lp_crop_size = [None, (470, 110), (600, 330)]


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


def show_results(img, xywh, conf, landmarks):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def crop_affine(img, landmarks, lp_type):
    h, w, c = img.shape
    points = []
    for i in range(4):
        points.append([landmarks[2 * i] * w, landmarks[2 * i + 1] * h])
    points = np.array(points, np.float32)
    M = cv2.getPerspectiveTransform(points, lp_transform_dest[lp_type])
    dst = cv2.warpPerspective(img, M, lp_crop_size[lp_type])
    return dst


def show_landmark(img, lands):
    
    width = img.shape[1]
    height = img.shape[0]
    tl = 1 or round(0.002 * (height + width) / 2) + 1  # line/font thickness
    # for land in lands:
    x_arr = []
    y_arr = []
    land = lands[0]
    for i in range(len(land)):
        if i%2==0:
            x_arr.append(int(land[i]*width))
        else:
            y_arr.append(int(land[i]*height))
    
    x_min = min(x_arr)
    x_max = max(x_arr)
    y_min = min(y_arr)
    y_max = max(y_arr)
            
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 10)
    for i in range(len(x_arr)):    
            cv2.circle(img, (x_arr[i], y_arr[i]), 10, (255, 0, 0), -1)
            cv2.putText(img, str(i), (x_arr[i], y_arr[i]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)
    
    cv2.line(img, (x_arr[0], y_arr[0]), (x_arr[1], y_arr[1]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[1], y_arr[1]), (x_arr[2], y_arr[2]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[2], y_arr[2]), (x_arr[3], y_arr[3]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[3], y_arr[3]), (x_arr[0], y_arr[0]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)

    new_width = int(width/4)
    new_height = int(height/4)
    img = cv2.resize(img, (new_width, new_height))

    return img


def detect_one(model, image_path, device, img_size, vis=False):
    # Load model
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    img1 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]
    # img = img0
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_corner(pred, conf_thres, iou_thres)

    import math

    # Process detections
    crop_images = []
    landmarkss = []

    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                landmarkss.append(landmarks)
                crop_images.append(crop_affine(img1, landmarks, 2))

    return orgimg, crop_images, landmarkss


def detect():
    img_ext = ['jpg', 'png', 'jpeg']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    if os.path.isfile(opt.image):
        print("ok")
        result, crop_images, lands = detect_one(model, opt.image, device, opt.img_size, vis=False)
        origin = cv2.imread(opt.image)

        if origin is not None:
            new_img = show_landmark(origin, lands=lands[0])
            cv2.imshow("Origin", origin)
            for i, crop_img in enumerate(crop_images):
                cv2.imshow("Image", crop_img)
                cv2.waitKey(0)
        
    elif os.path.isdir(opt.image):
        for image_name in os.listdir(opt.image):
            if image_name[-3:] not in img_ext:
                continue
            image_path = os.path.join(opt.image, image_name)
            origin = cv2.imread(image_path)
            if origin is not None:
                time1 = time.time()
                result, crop_images, lands = detect_one(model, image_path, device, opt.img_size, vis=False)
                print(time.time() - time1)
                new_img = show_landmark(origin, lands=lands)
                cv2.imshow("Origin", new_img)
                for i, crop_img in enumerate(crop_images):
                    cv2.imshow("Image", crop_img)
                    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/yolo_landmark_960_v3.pt',
                        help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='/home/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--project', default='runs/inference', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    detect()
