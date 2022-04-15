import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import preprocess_data
from clss import Point
from sklearn.cluster import KMeans
from genetic import genetic_shortest_path
import math



#
# def k_means(data, k, max_iter=10000):
#     centers = {}  # 初始聚类中心
#     # 初始化，随机选k个样本作为初始聚类中心。 random.sample(): 随机不重复抽取k个值
#     n_data = data.shape[0]  # 样本个数
#     for idx, i in enumerate(random.sample(range(n_data), k)):
#         # idx取值范围[0, k-1]，代表第几个聚类中心;  data[i]为随机选取的样本作为聚类中心
#         centers[idx] = data[i]
#
#         # 开始迭代
#     clusters = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
#     for i in range(max_iter):  # 迭代次数Í
#         # print("开始第{}次迭代".format(i + 1))
#         for j in range(k):  # 初始化为空列表
#             clusters[j] = []
#
#         for sample in data:  # 遍历每个样本
#             distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
#             for c in centers:  # 遍历每个聚类中心
#                 # 添加该样本点到聚类中心的距离
#                 distances.append(distance(sample, centers[c]))
#             idx = np.argmin(distances)  # 最小距离的索引
#             clusters[idx].append(sample)  # 将该样本添加到第idx个聚类中心
#
#         pre_centers = centers.copy()  # 记录之前的聚类中心点
#
#         for c in clusters.keys():
#             # 重新计算中心点（计算该聚类中心的所有样本的均值）
#             x_center = []
#             y_center = []
#             angle_center = []
#             for point in clusters[c]:
#                 x_center.append(point.x)
#                 y_center.append(point.y)
#                 angle_center.append(point.angle)
#             centers[c] = Point(x=np.mean(np.array(x_center)), y=np.mean(np.array(y_center)), angle=np.mean(np.array(angle_center)))
#
#         is_convergent = True
#         for c in centers:
#             if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
#                 is_convergent = False
#                 break
#         if is_convergent:
#             # 如果新旧聚类中心不变，则迭代停止
#             break
#     return centers, clusters

def draw_finger_print(finger_prints, index, tag, centers):
    num = (index - 10000)*2 + tag
    for center in centers:
        plt.scatter(center.x, center.y, color='r')

    for i in range(finger_prints[num].point_num.astype("int")):
        point_set = finger_prints[num].point_set
        plt.quiver(point_set[i].x, point_set[i].y, math.cos(math.pi / 180 * point_set[i].angle),
                   math.sin(math.pi / 180 * point_set[i].angle), angles='uv')
        ax = plt.gca()
        ax.set_aspect(1)
    plt.show()


def tsp_quick(point_list: list, start_index: int):
    sum_distance, seq_result, n = 0, [start_index, ], len(point_list)
    for path_index in range(n-1):
        min_dis = float('inf')
        for i in range(n):
            distance = point_list[start_index].get_distance(point_list[i])
            if (i not in seq_result) and (distance < min_dis):
                min_dis = distance
                start_index = i
        sum_distance += min_dis
        seq_result.append(start_index)
    return sum_distance, seq_result

def cal_shortest_path_of_centers(points, class_num, finger_index):
    pointlist = []
    for point in points:
        list = []
        list.append(point.x)
        list.append(point.y)
        list.append(point.angle*10)
        pointlist.append(list)
    cluster = KMeans(class_num, random_state=1).fit(np.array(pointlist))
    centers = []
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']  # 画图颜色
    # markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']  # 画图形状
    # for i, l in enumerate(cluster.labels_):
    #     plt.plot(points[i].x, points[i].y, markersize=2, color=colors[l], marker=markers[l], ls='None')
    for item in cluster.cluster_centers_:
        centers.append(Point(x=item[0], y=item[1], angle=item[2]))
    # draw_finger_print(finger_prints, index=finger_index//10, tag=finger_index-100000, centers=np.array(centers))

    min, path, time = genetic_shortest_path(centers, 10, 100)
    return min


random.seed(2022)
finger_prints = preprocess_data(data_path='./data/1.csv')
# finger1 = finger_prints[11]
# finger2 = finger_prints[10]
# pointset1 = finger1.point_set
# id1 = finger1.id
# # print("id : {}, min_distance:{}".format(id1, min_dis1))
# pointset2 = finger2.point_set
# id2 = finger2.id
# min_dis2 = cal_shortest_path_of_centers(pointset2, 10, finger_index=finger2.id)
# # print("id : {}, min_distance:{}".format(id2, min_dis2))
val = []
record = []
# for index in range(0, 1000, 2):
#     finger1 = finger_prints[index]
#     pointset1 = finger1.point_set
#     min_dis1 = cal_shortest_path_of_centers(pointset1, 13, finger_index=finger1.id)
#
#     finger2 = finger_prints[index+1]
#     pointset2 = finger2.point_set
#     min_dis2 = cal_shortest_path_of_centers(pointset2, 13, finger_index=finger2.id)
#     val.append(math.fabs(min_dis1-min_dis2))
#     print('{}'.format(math.fabs(min_dis1-min_dis2)))
num = 0
for index in tqdm(range(0, 1000, 2)):
    finger1 = finger_prints[index]
    pointset1 = finger1.point_set
    min_dis1 = cal_shortest_path_of_centers(pointset1, 5, finger_index=finger1.id)
    for i in range(1000):
        finger2 = finger_prints[i]
        pointset2 = finger2.point_set
        min_dis2 = cal_shortest_path_of_centers(pointset2, 5, finger_index=finger2.id)
        val.append(math.fabs(min_dis1-min_dis2))
        # print('{}'.format(math.fabs(min_dis1-min_dis2)))
        if math.fabs(min_dis1-min_dis2) < 50:
            record.append(i)
            num += 1
    # print(len(record))
    # print(record)
    print((index+1) in record)
    # plt.scatter(range(0, 1000), val, marker='.')
    # plt.show()

print('[Result] The total num of result that has been matched correctly is {}'.format(num))
# collect = []
# num = 0
#
#
#
#
# # for i in range(0, 1000):
# #     finger2 = finger_prints[i]
# #     pointset2 = finger2.point_set
# #     min_dis2 = cal_shortest_path_of_centers(pointset2, 5, finger_index=finger2.id)
# #     collect.append(math.fabs(min_dis2-min_dis1))
# #     # print(math.fabs(min_dis2-min_dis1))
# #     if math.fabs(min_dis1 - min_dis2) < 30:
# #         collect.append(i)
# #         num += 1
#
# print(index+1 in collect)
# print(num)
# plt.scatter(range(len(collect)), collect)
# plt.show()
