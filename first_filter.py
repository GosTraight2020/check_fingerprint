from tools import preprocess_data
from clss import get_distance, Point, Finger_print
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

DELTA_X = 10
DELTA_DISTANCE = 10
DELTA_ANGLE = 5
# 判断两个细节点是否匹配
def judge_similarity(a, b, delta_x, delta_y, dleta_angle):
    if math.fabs(b.x - a.x) < delta_x \
            and math.fabs(b.y - a.y) < delta_y \
            and math.fabs(b.angle - a.angle) < dleta_angle:
        return True
    else:
        return False

# dist 为距离矩阵，start_index 为起始位置
def tsp_quick(dist: list, start_index: int):
    sum_distance, seq_result, n = 0, [start_index, ], len(dist)
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





finger_prints = preprocess_data("./data/TZ_同指.csv")
a = Point(0, 0, 0)
b = Point(2, 77, 0)
c = Point(3, 9, 0)
d = Point(12, 56, 0)
e = Point(199, 3, 0)

point_list = [a, b, c, d, e]

sum_distance, seq_list = tsp_quick(point_list, 0)  # dist为距离矩阵，3表示从下标为3开始
print(sum_distance)
print(seq_list)
# x_list = []
# y_list = []
# for point in [a, b, c, d, e]:
#     x_list.append(point.x)
#     y_list.append(point.y)
#
# plt.scatter(x_list, y_list)
# # plt.show()
#
# #使用方法：
# dist = [
#     [a.get_distance(a), a.get_distance(b), a.get_distance(c), a.get_distance(d), a.get_distance(e)],
#     [b.get_distance(a), b.get_distance(b), b.get_distance(c), b.get_distance(d), b.get_distance(e)],
#     [c.get_distance(a), c.get_distance(b), c.get_distance(c), c.get_distance(d), c.get_distance(e)],
#     [d.get_distance(a), d.get_distance(b), d.get_distance(c), d.get_distance(d), d.get_distance(e)],
#     [e.get_distance(a), e.get_distance(b), e.get_distance(c), e.get_distance(d), e.get_distance(e)],
# ]
#
# min = float('inf')
# for i in range(5):
#     sum_distance,seq_list = tsp_quick(dist, i) # dist为距离矩阵，3表示从下标为3开始
#     print(min)
#     if sum_distance < min:
#         min = sum_distance
#         path = seq_list
#
# print('min distance:{}, the path is {}'.format(min, path))
# dic = {}
# for (i, point) in enumerate([a, b, c, d, e]):
#     dic[i] = point
#
# for i in range(5):
#     if i == 4:
#         break
#     plt.plot([dic[i].x, dic[i+1].x], [dic[i].y, dic[i+1].y], label=i)
#     plt.legend()
# plt.show()
# print(sum_distance)
# print(seq_list)
# #返回sum_distance 即为最短距离
# #返回序列 [3,2,1,0,4] 表示 3 -> 2 -> 1 -> 0 -> 4




