!pip install ultralytics
!pip install opencv-python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from img_preprocess import resize_img

def bbox2points(bbox):
    x, y, w, h = bbox
    xmin = int(round(x-(w/2)))
    xmax = int(round(x+(w/2)))
    ymin = int(round(y-(h/2)))
    ymax = int(round(y+(h/2)))
    return xmin, xmax, ymin, ymax

def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3) # 이미지를 darknet style로 전처리하여 darknet_image 변수에 저장

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BRG에서 RGB 순서로 변환

    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR) # input image의 size를 512x512로 변환

    img_height, img_width, _ = img.shape # bounding box의 크기 조정을 위한 이미지 크기 비율 계산 
    width_ratio = img_width/width # 원본 이미지 width/ network width
    height_ratio = img_height/height # 원본 이미지 height/ network height

    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)

    return detections, width_ratio, height_ratio


def yolo4_inference(img, cfg_dic, data_dic, weight_dic):
    # cfg = './cfg/yolov4-custom_test.cfg'
    # data = './data/cloud_data.data'
    # weights = './data/yolov4-custom_60.weights'
    network, class_names, class_colors = load_network(cfg_dic, data_dic, weight_dic)
    width = network_width(network) # -> 512, 512
    height = network_height(network)

    detections, width_ratio, height_ratio = darknet_helper(img, width, height)

    for label, confidence, bbox in detections:
        # center x, center y, width, height로 반환 받은 bounding box를 corner 정보 (left, top, right, bottom)로 변환
        left, top, right, bottom = bbox2points(bbox)
        # 앞서 계산한 이미지 비율을 사용하여 bounding box의 크기를 조정
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        # bounding box 좌표에 해당하는 사각형을 이미지 위에 그림
        cv2.rectangle(imagee, (left, top), (right, bottom), class_colors[label], 2)
        # 클래스 이름과 confidence level을 사각형 위에 입력
        cv2.putText(imagee, "{} [{:.2f}]".format(label, float(confidence)),
                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            class_colors[label], 2)

    image = cv2.cvtColor(imagee, cv2.COLOR_RGB2BGR)
    cv2_imshow(image)