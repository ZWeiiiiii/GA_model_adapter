# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_utils import *
import random
from quantization_and_noise.quant_layer import QuanModule
from quantization_and_noise.quant_util import *


# 权重替换
def replace_weights(model, model_weights, args):
    for n, m in model.named_modules():
        if type(m) in QuanModule:
            if hasattr(m, 'weight'):
                # print(m.weight)
                # print("-----------------------")
                set_weights = model_weights['{}.weights'.format(n)][0]
                m.weight.data = set_weights.to(args.device.type)
                # print(m.weight)
    return model


#获得初始种群---多个加不同噪声的模型（卷积层加相同的噪声，全连接层加相同的噪声）
def get_initial_population(model):
    populations = {}
    model_weight = {}
    model_num = 0
    for i in range(1, 6):
        for j in range(1, 6):
            model_num = model_num + 1
            set_noise_for_conv = i * 0.05
            set_noise_for_fc = j * 0.05
            # 设置权重量化器---加不同的噪声
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    print(name)
                    print(module)
                    # print(module.w_quantizer.noise_scale)
                    model_weight['{}.weights'.format(name)] = []
                    module.w_quantizer.noise_scale = set_noise_for_conv
                    # print(module.w_quantizer.noise_scale)
                    weight_q_conv = module.get_weight_quant()
                    # print(weight_q_conv.flatten().detach().numpy())
                    # print(len(weight_q_conv.flatten().detach().numpy()))
                    model_weight['{}.weights'.format(name)].append(weight_q_conv)
                elif isinstance(module, nn.Linear):
                    model_weight['{}.weights'.format(name)] = []
                    module.w_quantizer.noise_scale = set_noise_for_fc
                    weight_q_fc = module.get_weight_quant()
                    model_weight['{}.weights'.format(name)].append(weight_q_fc)
            populations['m{}'.format(model_num)] = model_weight

    return populations


def get_initial_populations_new(model):
    populations = {}
    model_weight = {}
    model_num = 0
    for i in range(1, 6):
        for j in range(1, 6):
            model_num = model_num + 1
            set_noise_for_conv = i * 0.05
            set_noise_for_fc = j * 0.05
            # 设置权重量化器---加不同的噪声
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    print(name)
                    print(module)
                    # print(module.w_quantizer.noise_scale)
                    model_weight['{}.weights'.format(name)] = []
                    ####
                    # module.w_quantizer.noise_scale = set_noise_for_conv
                    weight_data = module.weight.data
                    # print('-------------------weight data:',weight_data)
                    weight_data_add = add_noise(weight_data, 'add', set_noise_for_conv, "max_min")
                    # print('--------------add noise-----weight data:', weight_data_add)
                    # weight_q_conv = module.get_weight_quant()
                    # print(weight_q_conv.flatten().detach().numpy())
                    # print(len(weight_q_conv.flatten().detach().numpy()))
                    model_weight['{}.weights'.format(name)].append(weight_data_add)
                elif isinstance(module, nn.Linear):
                    model_weight['{}.weights'.format(name)] = []
                    weight_data = module.weight.data
                    weight_data_add = add_noise(weight_data, 'add', set_noise_for_fc, "max_min")
                    model_weight['{}.weights'.format(name)].append(weight_data_add)
            populations['m{}'.format(model_num)] = model_weight

    return populations


# 适应度函数,网络训练的准确率
def fitness(child, model,train_loader, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    model = replace_weights(model, child, args)
    model.to(args.device.type)
    t_top1, t_top5, t_loss = train(model, train_loader, criterion, optimizer,
                                   lr_scheduler, epoch, monitors, args)
    child_fitness = t_top1
    return child_fitness


# 选择过程
def selection(N):
    # 种群中随机选择2个个体进行变异（这里没有用轮盘赌，直接用的随机选择）
    return np.random.choice(N, 2)

def selection_new(m_num_acc, N, populations):
    p = []
    q =[]
    sum_acc = 0
    populations_after_select = {}
    for i in range(N):
        sum_acc = sum_acc + m_num_acc['m{}'.format(i + 1)]
    for i in range(N):
        p.append(m_num_acc['m{}'.format(i + 1)]/sum_acc)
    print(p)
    for i in range(N):
        if i == 0:
            q.append(p[i])
        else:
            q.append(q[i - 1] + p[i])
    print(q)
    for i in range(N):
        random_num = random.random()
        for k in range(N):
            if random_num < q[0]:
                populations_after_select['m{}'.format(i + 1)] = populations['m1']
            elif random_num < q[k] and random_num > q[k - 1]:
                populations_after_select['m{}'.format(i + 1)] = populations['m{}'.format(k + 1)]

    return populations_after_select





# 结合/交叉过程
def crossover(parent1, parent2, probability):  #输入两个父体，模型m1, m2，得到两个子代m1',m2'
    child1 = parent1
    child2 = parent2

    for key, _ in child1.items():
        for j in range(len(child1['{}'.format(key)])):
            # 以一定的概率进行交叉互换
            if random.random() < probability:
                #随机选择一个位置进行交换
                child1['{}'.format(key)][j], child2['{}'.format(key)][j] = child2['{}'.format(key)][j], child1['{}'.format(key)][j]
    return child1, child2

def crossover_new(populations, probability, N):
        a, b = np.random.choice(N, 2)  # 随机选择两个个体
        print(a, b)
        parent1, parent2 = populations['m{}'.format(a + 1)], populations['m{}'.format(b + 1)]
        child1 = parent1
        child2 = parent2
        print(child1)
        print(child2)
        for key, _ in child1.items():
            for j in range(len(child1['{}'.format(key)])):
                # 以一定的概率进行交叉互换
                if random.random() < probability:
                    # 随机选择一个位置进行交换
                    child1['{}'.format(key)][j], child2['{}'.format(key)][j] = child2['{}'.format(key)][j], child1['{}'.format(key)][j]
        return child1, child2, a, b




# 变异过程
def mutation(population, probability):
    # 种群中随机选择一个进行变异
    for key, _ in population.items():  #遍历模型
        for j in range(len(population['{}'.format(key)])):  #遍历层
            # 以一定的概率进行交叉互换
            if random.random() < probability:
                # 随机选择一个位置进行交换
                population['{}'.format(key)][j] = random.uniform(max(population['{}'.format(key)]),
                                                                 min(population['{}'.format(key)]))
                # if j == (len(population['{}'.format(key)]) -1):
                #     population['{}'.format(key)][j] = population['{}'.format(key)][j - 1]   #随机数未确定---变为相邻的数值
                #
                # else:
                #     population['{}'.format(key)][j] = population['{}'.format(key)][j + 1]   #随机数未确定---变为相邻的数值
    return population




# dict1 = {'m1':{'c1':[1, 2, 3, 4, 5]}, 'm2':{'c1':[6, 7, 8, 9, 10]}}
# m_num_acc={'m1': 23.76, 'm2':23.76}
# populations_new = selection_new(m_num_acc,2,dict1)
# print(populations_new)
# dict1_new, dict2_new, a, b = crossover_new(populations_new, 0.5, 2)
#
# dict3_new = mutation(dict1_new, 0.5)
# print(dict1_new, dict2_new)
# print(dict3_new)

