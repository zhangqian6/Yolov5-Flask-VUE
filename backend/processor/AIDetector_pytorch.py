import torch
import numpy as np
import argparse

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator,colors
import cv2
from random import randint




class Detector(object):

    def __init__(self):
        self.model_init()

# 模型初始化
    def model_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='/weights/yolov5s.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default='',     #X:\毕业设计\源码\yolov5_7.0\yolov5\data\images\zidane.jpg
                            help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default='',  #X:\毕业设计\源码\yolov5_7.0\yolov5\data\coco128.yaml
                            help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_false', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand


        source = str(self.opt.source)
        self.save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images

        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(self.opt.weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data,
                                        fp16=self.opt.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size


    def detect(self, img):
        # Run inference对导入的图片进行推理
        label = ...
        im = letterbox(img, self.imgsz)[0]  # padded resize将原图变成特定大小的图片(640,480,3)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=self.opt.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                   self.opt.agnostic_nms, max_det=self.opt.max_det)

        self.names = self.model.names
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process prediction
        name = ...
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(img, line_width=self.opt.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_img or self.opt.save_crop or self.opt.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.opt.hide_labels else (
                            self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy,label,color=colors(c, True))
                        xxx = label.split(' ')
                        name = xxx[0]

            img = annotator.result()
        return img,name

if __name__=='__main__':
    detector = Detector()
    img_file = '../tmp/ct/children.jpg'
    img = cv2.imread(img_file)
    img, info = detector.detect(img)
    print(type(info))
    cv2.imshow(info,img)
    cv2.waitKey(0)
