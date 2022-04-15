# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:06:59 2019

@author: 1
"""
# 导入需要用到的包
import random
import matplotlib.pyplot as plt
import datetime
from tools import preprocess_data
import numpy as np

random.seed(2022)
np.random.seed(2022)

def genetic_shortest_path(pointset, species, iters):

    def getListMaxNumIndex(num_list, topk=int(0.2 * species)):
        '''
        获取列表中最大的前n个数值的位置索引
        '''
        num_dict = {}
        for i in range(len(num_list)):
            num_dict[i] = num_list[i]
        res_list = sorted(num_dict.items(), key=lambda e: e[1])
        max_num_index = [one[0] for one in res_list[::-1][:topk]]
        return max_num_index

    # 适应度函数
    def calfit(trip, num_city):
        total_dis = 0
        for i in range(num_city):
            cur_city = trip[i]
            next_city = trip[i + 1] % num_city
            temp_dis = distance[cur_city][next_city]
            total_dis = total_dis + temp_dis
        return 1 / total_dis

    def dis(trip, num_city):
        total_dis = 0
        for i in range(num_city):
            cur_city = trip[i]
            next_city = trip[i + 1] % num_city
            temp_dis = distance[cur_city][next_city]
            total_dis = total_dis + temp_dis
        return total_dis

    # 交叉函数
    def crossover(father, mother):
        num_city = len(father)
        # indexrandom = [i for i in range(int(0.4*cronum),int(0.6*cronum))]
        index_random = [i for i in range(num_city)]
        pos = random.choice(index_random)
        son1 = father[0:pos]
        son2 = mother[0:pos]
        son1.extend(mother[pos:num_city])
        son2.extend(father[pos:num_city])

        index_duplicate1 = []
        index_duplicate2 = []

        for i in range(pos, num_city):
            for j in range(pos):
                if son1[i] == son1[j]:
                    index_duplicate1.append(j)
                if son2[i] == son2[j]:
                    index_duplicate2.append(j)
        num_index = len(index_duplicate1)
        for i in range(num_index):
            son1[index_duplicate1[i]], son2[index_duplicate2[i]] = son2[index_duplicate2[i]], son1[index_duplicate1[i]]

        return son1, son2

    # 变异函数
    def mutate(sample):
        num_city = len(sample)
        part = np.random.choice(num_city, 2, replace=False)
        if part[0] > part[1]:
            max_ = part[0]
            min_ = part[1]
        else:
            max_ = part[1]
            min_ = part[0]
        after_mutate = sample[0:min_]
        temp_mutate = list(reversed(sample[min_:max_]))
        after_mutate.extend(temp_mutate)
        after_mutate.extend(sample[max_:num_city])
        return after_mutate

    def get_distance(point1, point2):  # 计算距离（欧几里得距离）
        sq1 = (point1.x - point2.x) ** 2
        sq2 = (point1.y - point2.y) ** 2
        return np.sqrt(np.sum([sq1, sq2]))

    starttime = datetime.datetime.now()

    pointlist = []
    for point in pointset:
        tmp = []
        tmp.append(point.x)
        tmp.append(point.y)
        pointlist.append(tmp)

    # 计算各城市邻接矩阵。
    n = len(pointset)
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance[i][j] = get_distance(pointset[i], pointset[j])

    pointset = np.array(pointlist)

    # 初始化种群，生成可能的解的集合
    x = []
    counter = 0
    # for i in range(species):
    while counter < species:
        dna = np.random.permutation(range(n)).tolist()
        start = dna[0]
        dna.append(start)
        if dna not in x:
            x.append(dna)
            counter = counter + 1

    ctlist = []
    dislist = []
    ct = 0
    while ct < iters:
        ct = ct + 1
        f = []
        for i in range(species):
            f.append(calfit(x[i], n))

        # 计算选择概率
        sig = sum(f)
        p = (f / sig).tolist()

        # 选取最优秀的20%
        test = getListMaxNumIndex(p)
        testnum = len(test)
        newx = []
        for i in range(testnum):
            newx.append(x[test[i]])
            # newx.append(x[test[i]])
        index = [i for i in range(species)]
        # 20%优秀的必须选入， 再随机挑选80%（尽可能挑选比较优秀的）
        news = random.choices(index, weights=p, k=int(0.8 * species))
        newsnum = len(news)
        for i in range(newsnum):
            newx.append(x[news[i]])
        # 从新挑选的200个中进行交叉变异
        m = int(species / 2)
        for i in range(0, m):
            j = i + m - 1
            # j=i+1
            # 路径长度 numx
            numx = len(newx[0])
            if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) < 8:
                # 去掉尾部
                tplist1 = newx[i][0:numx - 1]
                tplist2 = newx[j][0:numx - 1]
                # 70%概率交叉
                crosslist1, crosslist2 = crossover(tplist1, tplist2)
                # 30%概率变异
                if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) < 4:
                    crosslist1 = mutate(crosslist1)
                    crosslist2 = mutate(crosslist2)
                end1 = crosslist1[0]
                end2 = crosslist2[0]
                crosslist1.append(end1)
                crosslist2.append(end2)
                newx[i] = crosslist1
                newx[j] = crosslist2
        x = newx
        res = []
        for i in range(species):
            res.append(calfit(x[i], n))
        # 最短距离
        result = 1 / max(res)
        res1 = []
        for i in range(species):
            # 计算总路径长度
            res1.append(dis(x[i], n))
        result1 = min(res1)
        # print(ct)
        # print(result)
        # print(result1)
        ctlist.append(ct)
        dislist.append(result)

    endtime = datetime.datetime.now()

    time = endtime - starttime

    plk1 = []
    plk2 = []
    for i in range(len(x[0])):
        plk2.append(pointset[x[0][i], 0])
        plk1.append(pointset[x[0][i], 1])

    # plot = plt.plot(plk1, plk2, c='r')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # plt.plot(ctlist, dislist)
    # plt.xlabel('iters')
    # plt.ylabel('distance')
    # plt.show()

    return result1, x[0], time


