import numpy as np
from clss import Finger_print, Point
import math
from clss import get_distance

# 将txt中的数据转化成指纹类的数组
def preprocess_data(data_path):
    data = np.genfromtxt(data_path, delimiter=',',)
    finger_prints = []
    for line in data:
        line = [item.astype("int") for item in line if not np.isnan(item)]
        line = np.array(line)
        len = line.shape[0]
        fp = Finger_print(id=line[0], point_num=line[1])
        x = []
        y = []
        angle = []
        for index in range(len - 2):
            if index % 3 == 0:
                x.append(line[index + 2])
            elif index % 3 == 1:
                y.append(line[index + 2])
            else:
                angle.append(line[index + 2])
        points = []
        for i in range(line[1].astype('int')):
            point = Point(x=x[i], y=y[i], angle=angle[i])
            points.append(point)
        fp.point_set = points
        finger_prints.append(fp)

    return np.array(finger_prints)

