!pip install split-folders
!pip install opencv-python

import os
import cv2
import shutil
import splitfolders
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compute_mean(imgs)
    mean_img = np.mean(imgs, axis = 0)
    return mean_img

def mean_img(folders): # mean_img를 만들기 위해 합칠 폴더 목록
    content_list = {} 
    for index, folder in enumerate(folders):
        path = os.path.join('./Cloud_Classification_for_Weather_Forecast/data', folder)
        filenames = os.listdir(path)
        content_list[folders[index]] = filenames # {'test' : [test filenames], 'train' : [train filenames], 'valid' : [valid filenames]}
        
        !mkdir "./data/images"
        !mkdir "./data/texts"

        merge_folder_path = './data/images'

    for sub_dir in content_list: # [test], [train], [valid]
        for contents in content_list[sub_dir]:
            if contents.split('.')[-1] == 'jpg':
                path_to_content = sub_dir + "/" + contents # 이동시킬 파일 폴더 위치
                dir_to_move = os.path.join('./data', path_to_content) # 현재 위치
                shutil.move(dir_to_move, merge_folder_path) # 현재 위치에서 images 파일 위치로 이동

    merge_folder_path_txt = './data/texts'

    for sub_dir in content_list:
        for contents in content_list[sub_dir]:
            if contents.split('.')[-1] != 'jpg':
                path_to_content = sub_dir + "/" + contents
                dir_to_move = os.path.join('./data', path_to_content)
                shutil.move(dir_to_move, merge_folder_path)

    splitfolders.ratio('./data/images', output = 'train_dataset', seed = 69, ratio = (0.7, 0.2, 0.1))

    for folder in folders:
        with open(f'{folder}.txt', 'w') as f:
            for content in content_list[folder]:
                if content[-3:] == 'jpg':
                    f.write(f'./data/images/{content}\n')

    train_paths = content_list['train']

    img_paths = []
    for train_path in train_paths:
        if train_path.split('.')[-1] == 'jpg':
            img_paths.append('.data/train_dataset/train'+ train_path)

    train_data = []

    for img_file in img_paths:
        img = cv2.imread(img_file)
        train_data.append(img)

    mean_img = compute_mean(train_data)
    mean_img = mean_img.astype(int)
    with open('./data/mean_img.pickle', 'wb') as f:
        pickle.dump(mean_img, f)

    train_paths = content_list['train']
    test_paths = content_list['test']
    val_paths = content_list['valid']

    for label in content_list:
        img_paths = []
        sub_path = './data/'+ label + '/'
        for files in content_list[label]:
            if files.split('.')[-1] == 'jpg':
                img_paths.append(sub_path + files)
        for img_file in img_paths:
            
            img = cv2.imread(img_file)
            zero_img = img- mean_img
            cv2.imwrite(img_file, zero_img)
    return zero_img