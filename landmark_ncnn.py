from detection.yolov5_ncnn.utils import *
from detection.yolov5_ncnn.utils.functional import *
from etection.yolov5_ncnn.utils.general import *
from common.base import DetectionBase
import ncnn
import numpy as np
import time
import torch
import cv2
from time import time

class YoloLandmark(DetectionBase):
    def __init__(self, model_bin, model_param, img_size=416, conf_thres=0.7, iou_thres=0.5):
        super().__init__("barcode", "yolov5_landmark_ncnn")

        anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
        self.na = len(anchors[0]) // 2
        self.nl = len(anchors)
        self.nc = 3
        self.num_landmarks = 4
        self.no = self.nc + 5 + self.num_landmarks * 2 
        self.target_size = img_size 
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.landmark_net = ncnn.Net()
        self.landmark_net.load_param(model_param)
        self.landmark_net.load_model(model_bin)
        self.device = torch.device('cpu')

        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

        self.anchor_grid = torch.FloatTensor(anchors)
        self.anchor_grid = torch.reshape(self.anchor_grid, (self.nl, 1, self.na, 1, 1, -1))

        self.confThreshold = conf_thres
        self.nmsThreshold = iou_thres
        self.class_names = ["qr","ocr","barcode"]

    def __del__(self):
        self.landmark_net = None

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
    def _detect(self, imgs):
        img = imgs[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_w = img.shape[1]
        img_h = img.shape[0]
        w = img_w
        h = img_h
        scale = 1.0
        if w > h:
            scale = float(self.target_size) / w
            w = self.target_size
            h = int(h * scale)
        else:
            scale = float(self.target_size) / h
            h = self.target_size
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, w, h
        )
        wpad = (w + 31) // 32 * 32 - w
        hpad = (h + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )

        
        start_time = time()

        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.landmark_net.create_extractor()
        ex.input("data", mat_in_pad)

        # anchor setting from yolov5/models/yolov5s.yaml
        # output_1 = ncnn.Mat()
        # output_2 = ncnn.Mat()
        # output_3 = ncnn.Mat()
        ret1, mat_out1 = ex.extract("onnx::Reshape_949")  # stride 8
        ret2, mat_out2 = ex.extract("onnx::Reshape_963")  # stride 16
        ret3, mat_out3 = ex.extract("onnx::Reshape_977")  # stride 32

        np_out_data_1 = np.array(mat_out1)
        np_out_data_1 = np.expand_dims(np_out_data_1, axis=0)

        np_out_data_2 = np.array(mat_out2)
        np_out_data_2 = np.expand_dims(np_out_data_2, axis=0)

        np_out_data_3 = np.array(mat_out3)
        np_out_data_3 = np.expand_dims(np_out_data_3, axis=0)

        x = [np_out_data_1, np_out_data_2, np_out_data_3]
        pred = []
        for i in range(self.nl):
            x[i] = torch.from_numpy(x[i]).to(self.device)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            # break 
            y = torch.full_like(x[i], 0)
            class_range = list(range(5)) + list(range(13,13+self.nc))
            y[..., class_range] = x[i][..., class_range].sigmoid()
            y[..., 5:13] = x[i][..., 5:13]
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y[..., 5:7]   = y[..., 5:7] *   self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i] # landmark x1 y1
            y[..., 7:9]   = y[..., 7:9] *   self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x2 y2
            y[..., 9:11]  = y[..., 9:11] *  self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x3 y3
            y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x4 y4
            # y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x5 y5
            pred.append(y.view(bs, -1, self.no))

        pred = torch.cat(pred, 1)
        pred = non_max_suppression_face(pred, self.confThreshold, self.nmsThreshold)
        # print('Time to detect: ', time() - start_time)
        results= []
        detect_objects = []
        for i, det in enumerate(pred):
            class_nums = []
            landmarks = [] 
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
            gn_lks = torch.tensor(img.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(
                self.device)  # normalization gain landmarks
            if len(det):
                det[:, :4] = scale_coords([mat_in_pad.h, mat_in_pad.w], det[:, :4], img.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                det[:, 5:13] = scale_coords_landmarks([mat_in_pad.h, mat_in_pad.w], det[:, 5:13], img.shape).round()
                class_nums = det[:, 13].cpu().numpy()
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    # landmark = det[j, 5:13].view(-1).cpu().numpy().astype(int).tolist()
                    landmark = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                    class_num = int(det[j, 13].cpu().numpy())
                    landmarks.append(landmark)
                    obj = Detect_Object(label=class_num, prob=conf, x=xyxy[0], y=xyxy[1], w=xyxy[2] - xyxy[0], h=xyxy[3] - xyxy[1])
                    obj.landmark = landmark
                    detect_objects.append(obj)
                results.append({'image': img,
                                'coordinates': landmarks,
                                'class_nums': class_nums,
                                })
        return results
    
