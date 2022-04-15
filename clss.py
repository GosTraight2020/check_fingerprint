import numpy as np
import math
# 特征点类
class Point:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.neighbours = []
        self.nb_angles = []
        self.nb_distances = []
        self.id = -1

    def set_id(self, id:int):
        self.id = id

    def get_id(self)->int:
        return self.id

    def set_neighbour(self, point):
        self.neighbours.append(point)

    def cal_angle_distance(self):
        if len(self.neighbours) == 0:
            raise ValueError('[Error] There is no members in the neighbour set!')
        else:
            # print('[DEBUG] The num of neighbour set is {}'.format(len(self.neighbours)))
            neighbours = np.array(self.neighbours)
            angle_list = []
            distance_list = []
            for i in range(neighbours.shape[0]):
                for j in range(i+1, neighbours.shape[0]):
                    angle = get_angle(self, neighbours[i], neighbours[j])
                    distance = get_distance(neighbours[i], neighbours[j])
                    angle_list.append(angle)
                    distance_list.append(distance)
            self.nb_angles = angle_list
            self.nb_distances = distance_list

    def get_distance(self, point)->np.float32:
        return np.sqrt(np.sum([(self.x - point.x)**2, (self.y - point.y)**2]))




# 指纹类
class Finger_print:
    def __init__(self, id, point_num):
        self.id = id
        self.point_num = point_num
        self.point_set = []
        self.NEIGHOUTR_LIMIT = 5
        self.shortest_path = 0
        self.seq_path = []

    def get_id(self):
        return self.id

    def get_point_num(self):
        return self.point_num

    def get_all_x(self):
        x_list = []
        for point in self.point_set:
            x_list.append(point.x)
        return np.array(x_list)

    def get_all_y(self):
        y_list = []
        for point in self.point_set:
            y_list.append(point.y)
        return np.array(y_list)

    def set_neighbours(self, k):
        point_set = np.array(self.point_set)
        if point_set.shape[0] != 0:
            # 为所有细节点创建自己的邻居集合
            for i in range(point_set.shape[0]):
                list=[]
                dic = {}
                point_A = point_set[i]
                # 遍历所有点 收集距离
                for (i, point_B) in enumerate(point_set):
                    distance = get_distance(point_A, point_B)
                    # 筛除掉离得太近的点
                    if distance != 0.0001:
                        list.append(distance)
                        dic[distance] = i
                # 对距离进行排序
                list.sort()
                for j in range(k):
                    index = dic[list[j]]
                    point_A.set_neighbour(point_set[index])



def get_distance(a:Point, b:Point)->float:
    temp = np.square(a.x - b.x)+np.square(a.y - b.y)
    return np.sqrt(temp+1e-8)

# 两个点相对于中心点所组成向量的点积
def dot(core:Point, a:Point, b:Point)->float:
    x1 = a.x - core.x
    x2 = b.x - core.x
    y1 = a.y - core.y
    y2 = b.y - core.y
    return x1 * x2 + y1 * y2


# 计算两个近邻的细节点相对于中心细节点的角度
def get_angle(core:Point, a:Point, b:Point):
    temp =  math.acos((dot(core, a , b)/(get_distance(core, a)*get_distance(core, b))))
    return temp/math.pi*180
