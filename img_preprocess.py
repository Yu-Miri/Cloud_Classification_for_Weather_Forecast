!pip install opencv-python
import cv2
import numpy as np

# 함수 적용 시 cv2.imread 된 이미지

def resize_img(img_path, img_size=640):    
    img = cv2.imread(img_path)
    
    if(img.shape[1] > img.shape[0]) : 
        ratio = img_size/img.shape[1]
    else :
        ratio = img_size/img.shape[0]

    img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    # 그림 주변에 검은색으로 칠하기
    w, h = img.shape[1], img.shape[0]

    dw = (img_size-w)/2 # img_size와 w의 차이
    dh = (img_size-h)/2 # img_size와 h의 차이

    M = np.float32([[1,0,dw], [0,1,dh]])  #(2*3 이차원 행렬)
    img_re = cv2.warpAffine(img, M, (640, 640)) #이동변환  
    return img_re

def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(224, 224, 1)
    return img

def adaptive_binarization(img):
    gray_img = gray_scale(img)
    adap_bin_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    return adap_bin_img

def histogram_equalization(img):
    gray_img = gray_scale(img)
    equal_img = cv2.equalizeHist(img)
    return equal_img