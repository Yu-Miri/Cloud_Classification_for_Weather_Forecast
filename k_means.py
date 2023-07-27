import numpy as np
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(txt_file):

    with open(txt_file, 'r') as f: #txt_file = './darknet/data/cloud_train_annotation.txt'
        total_ws = []
        total_hs = []
        for num, line in enumerate(f):
            annotation = line.strip().split()
            bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]])
            if len(bbox_data_gt) == 0:
                ws = []
                hs = []
            else:
                ws, hs = bbox_data_gt[:, 2], bbox_data_gt[:, 3]
            total_ws.extend(ws)
            total_hs.extend(hs)

    train_df = pd.DataFrame(zip(total_ws, total_hs), columns= ['width', 'height'])

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10))
    visualizer.fit(train_df)

    k = 9 # 그룹 수, random_state 설정
    model = KMeans(n_clusters = k, random_state = 10)

    model.fit(train_df) # 정규화된 데이터에 학습

    train_df['cluster'] = model.fit_predict(train_df) # 클러스터링 predict, 각 데이터가 몇 번째 그룹에 속하는지 저장

    plt.figure(figsize = (8, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(k):
        plt.scatter(train_df.loc[train_df['cluster'] == i, 'width'], train_df.loc[train_df['cluster'] == i, 'height'], 
                    label = 'cluster ' + str(i),
                    c=colors[i])

    plt.legend()
    plt.title('K = %d results'%k , size = 15)
    plt.xlabel('width', size = 12)
    plt.xlim([0, 650])
    plt.ylabel('height', size = 12)
    plt.ylim([0, 650])
    plt.show()


    centers_df = pd.DataFrame(model.cluster_centers_, columns=['C-width', 'C-height', '표준편차?'])
    print(centers_df)
    return centers_df