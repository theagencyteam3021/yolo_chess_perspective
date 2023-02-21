from cv2 import cv2
import torch._C as torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

weights = 'runs/weights/chess.pt'
device = select_device(0)
half = device != 'cpu'
trace = True
imgsz=640

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16

img0 = None
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Failed to open camera stream'

rect = True

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

warmup = True

conf_thres = 0.2
iou_thres = 0.45

def getBoard():
    #get image
    cap.grab()
    success, im = cap.retrieve()
    img0 = im if success else 0
    #resize and convert
    img = [letterbox(img0, imgsz, auto=rect, stride=stride)[0]]
    img = np.stack(img, 0)
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
    img = np.ascontiguousarray(img)

    #get ready for inference 
    img = img[0]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    #Do a "warmup" for some reason, idk
    if warmup and device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for _ in range(3):
            #Not nograd, maybe callibrating the gradients? Not sure, I didn't write this.
            model(img, augment=False)[0]

    #Actual inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
     # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                center = (xywh[0], xywh[1]-xywh[3]+xywh[2]/2)
    
    