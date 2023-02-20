from cv2 import cv2
import torch
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


    

