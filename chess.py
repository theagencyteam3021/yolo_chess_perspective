import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from chesscorner import get_matrix_from_img, calculate_new_points

from stockfish import Stockfish
import time
import os
stockfish = Stockfish()


camera_mode = True
show_board = True

weights = 'weights/chess1.pt'
device = select_device('0')
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
img0 = cv2.imread("nonblank2.jpg")

if camera_mode:
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Failed to open camera stream'

rect = True
names = ['b','k','n','p','q','r','B','K','N','P','Q','R']

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

warmup = True

conf_thres = 0.1
iou_thres = 0.45



def set_orientation(camera_mode):
    
    if not camera_mode:
        print("Using stored image for perspective transformation")
        im = cv2.imread("blank2.jpg")
        M, rect_base = get_matrix_from_img(im)
        im = cv2.warpPerspective(im, M, (int(rect_base), int(rect_base)))
        cv2.imshow("im",im)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
        return M, rect_base
    n = input("Take a picture of the empty board")
    if not n:
        cap.grab()
        success, im = cap.retrieve()
        if not success:
            print("Failed to get blank board")
            return set_orientation(False)
        cv2.imshow("im0", im)
        cv2.waitKey(5000)
        cv2.imwrite("latest_blank.jpg", im)
    else:
        im = cv2.imread("latest_blank.jpg")
    M, rect_base = get_matrix_from_img(im)
    im = cv2.warpPerspective(im, M, (int(rect_base), int(rect_base)))
    cv2.imshow("im",im)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    return M, rect_base
    
    
    

def get_board(M, rect_base):
    #get image
    img0=cv2.imread("nonblank2.jpg")
    if camera_mode:
        cap.grab()
        for i in range(10):
            success, im = cap.read()
        img0 = im if success else img0
        print("Writing image")
        cv2.imwrite(str(time.time())+'.jpg',img0)
        
        #cv2.imshow('img0',img0)
        #cv2.waitKey(10000)
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
        
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    #Do a "warmup" for some reason, idk
    if warmup and device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for _ in range(3):
            #Not nograd, maybe callibrating the gradients? Not sure, I didn't write this.
            model(img, augment=False)[0]
    print("completed warmup")
    #Actual inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
     # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    og_pieces = [] #(cls, (x, y))
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                center = (int(xywh[0]), int(xywh[1]+xywh[3]/2-xywh[2]/2))
                og_pieces.append((int(cls), center))
                #cv2.circle(img0, center, radius=10, color=(0, 0, 255))
                #print(int(cls), center)
    #Perspective transform
    #print(og_pieces)
    trans_img = cv2.warpPerspective(img0, M, (int(rect_base), int(rect_base)))
    cv2.imshow("img", trans_img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    piece_centers = [p[1] for p in og_pieces]
    new_centers = calculate_new_points(np.array(piece_centers),M).astype("int64").tolist()
    transformed_pieces = [(og_pieces[i][0], tuple(c)) for i, c in enumerate(new_centers)]
    #print(transformed_pieces)
    board_coors = [] 
    for i, p in enumerate(new_centers):
        cv2.circle(trans_img, p, radius=10, color=(0, 0, 255))
        board_coors.append((transformed_pieces[i][0], (int(8*p[0]/rect_base), int(8*p[1]/rect_base))))
    #print(board_coors)
    
    board = [[' ' for i in range(8)] for j in range(8)]
    fen_board = ""
    for p in board_coors:
    	board[p[1][1]][p[1][0]]=names[p[0]]
    for l in board:
        print(l)
        fen_line = ''
        run = 0
        for i, p in enumerate(l):
    	    if p != ' ':
    	        if run != 0:
    	            fen_line += str(run)
    	        fen_line += p
    	        run = 0
    	        
    	    else:
    	        run += 1
    	        if i == 7:
    	            fen_line += str(run)
        fen_board += fen_line + '/'
    fen_board = fen_board[:-1] + " w - - 0 0"
    	
    
        
    #cv2.imshow("img",trans_img)
    #cv2.waitKey(600)
    #print(fen_board)
    return board
    
    '''names = model.module.names if hasattr(model, 'module') else model.names
    print(names)'''
def to_fen(board):
    fen_board = ""
    for l in board:
        #print(l)
        fen_line = ''
        run = 0
        for i, p in enumerate(l):
    	    if p != ' ':
    	        if run != 0:
    	            fen_line += str(run)
    	        fen_line += p
    	        run = 0
    	        
    	    else:
    	        run += 1
    	        if i == 7:
    	            fen_line += str(run)
        fen_board += fen_line + '/'
    fen_board = fen_board[:-1] + " w - - 0 0"
    if show_board and stockfish.is_fen_valid(fen_board):
        stockfish.set_fen_position(fen_board)
        print(stockfish.get_board_visual())
    return fen_board, stockfish.is_fen_valid(fen_board)
    
    
def next_move(fen_board):
    
    #print(fen_board)
    if stockfish.is_fen_valid(fen_board):
        stockfish.set_fen_position(fen_board)
        move = str(stockfish.get_best_move())
        start = move[:2].upper()
        end = move[2:].upper()
        capture = bool(stockfish.get_what_is_on_square(end.lower()))
        return start, end, capture 
    
        
if __name__ == '__main__':    
    M, rect_base=set_orientation(camera_mode)
    input("set up pieces")   
    board = get_board(M, rect_base)
    print(next_move(to_fen(board)))
