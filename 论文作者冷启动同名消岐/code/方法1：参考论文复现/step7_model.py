import torch.nn as nn
import torch
import json
import step1_pre_pro
import numpy as np
import random


class LogisticRegression(nn.Module):  # 定义一个类，完成triplet loss 神经网络
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        # nn.init.xavier_uniform_

        self.w1 = nn.Parameter(nn.init.kaiming_normal_(torch.empty(input_size,128)), requires_grad=True).type(torch.float32)
        self.b1 = nn.Parameter(torch.randn(128).uniform_(-0.1, 0.1), requires_grad=True).type(torch.float32)
        self.w2 = nn.Parameter(nn.init.kaiming_normal_(torch.empty(128,output_size)), requires_grad=True).type(torch.float32)
        self.b2 = nn.Parameter(torch.randn(output_size).uniform_(-0.1, 0.1), requires_grad=True).type(torch.float32)

    def forward(self, inp):
        a1 = torch.matmul(inp, self.w1) + self.b1  # shape (1,128)
        h1 = torch.relu(a1)
        a2 = torch.matmul(h1, self.w2) + self.b2  # shape(1,64)
        pred = torch.relu(a2)
        return pred


def computer_loss(pred_pos, pred_neg, label):
    m = torch.tensor(1.0)  # 网上示例代码中的margin
    # print(pred_pos)
    # print(pred_pos.shape)
    # pred_pos = torch.squeeze(pred_pos, dim=1)
    # pred_neg = torch.squeeze(pred_neg, dim=1)
    # -----------------------------------------分别计算两个例子的损失值
    loss_pos = torch.norm(label - pred_pos, p=2)
    loss_neg = torch.norm(label - pred_neg, p=2)
    # -----------------------------------------
    loss = loss_pos - loss_neg + m
    if loss < 0:  # 说明正例与负例相距甚远
        loss = torch.tensor(0.0)

    # pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    # accuracy = torch.mean(torch.eq(pred, label).float())
    return loss


def cluster2_posneg(clusters):  # 给定一个簇的列表，生成所有的正例对，负例对
    middle_triple = []  # 这里存放所有的三元组，三元组经过处理之后的正例与负例对将要存放进final里面去
    for index_1, cluster_pos in enumerate(clusters):  # 遍历每一个簇（簇中的芦苇嫩都是同一个人写的）
        cluster_neg = []  # 将所有与当前簇不同的簇中的元素集中在一起
        for index_2, c in enumerate(clusters):  # 遍历所有的非当前簇的簇
            if index_1 != index_2:  # 非当前簇
                for i in c:  # 遍历该簇中的每个元素
                    cluster_neg.append(i)

        count = 0
        for index_p1, a1 in enumerate(cluster_pos):  # 遍历正例簇。制作三元组。
            if len(cluster_pos) == 1:  # 整个簇只有一篇论文
                continue
            samples_posnum = min(5, len(cluster_pos))  # 决定抽样数字
            idx_pos = random.sample(range(len(cluster_pos)), samples_posnum)  # 决定正例下标的list
            for i_pos in idx_pos:  # 生成的list中的每个元素都是下标（cluster_pos）
                if i_pos != index_p1:  # 正例与当前a1的下标应该不等
                    a2 = cluster_pos[i_pos]
                    idx_neg = random.sample(range(len(cluster_neg)), 1)  # 负例的下标
                    a3 = cluster_neg[idx_neg[0]]  # 得到负例
                    middle_triple.append([a1, a2, a3])
    return middle_triple


def train_gen_cluster_posneg():  # 从（训练集，测试集）中给定真实人名，生成正例对，负例对
    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as f_author:  # 获取所有的作者信息
        author_data = json.load(f_author)
        name_set = step1_pre_pro.name_over5(author_data)

    count = 0  # 计数人名
    triplet_dict = {}  # 字典，key是人名，value是簇中所有的三元组
    for real_name in name_set:  # 遍历每个人名，相当于遍历每一个cluster
        clusters = []  # 里面的每个元素是一个数组，数组里的元素是所有同一个人写的论文pub_id
        for name_id in author_data[real_name]:  # 对于每一个name id
            clusters.append(author_data[real_name][name_id])
        t = cluster2_posneg(clusters)
        triplet_dict[real_name] = t
        count = count + len(t)
        print(count)

    with open("triplet_sample/triplet_sample.json", "w", encoding="UTF-8") as wf:
        json.dump(triplet_dict, wf, indent=4)


#train_gen_cluster_posneg()

def convert_tripet_pubid2vec():  # 将三元组内的pubid转换成vec，然后将每个簇单独写在一个json文件中
    with open("triplet_sample/triplet_sample.json", "r", encoding="UTF-8") as rf:
        triplet_pubid = json.load(rf)
    with open("word2vec_result/trainpub_word2vec_result.json", "r", encoding="UTF-8") as rv:
        train_pubvec = json.load(rv)

    count=0
    for real_name in triplet_pubid:  # 遍历每个簇中的所有三元组
        triplet_pubvec = {}  # 字典，key是real name，value是三元组的list，然而三元组中的元素都是vec（word2vec后的论文）
        t_vec = []  # 里面的每一个元素都是三元组，三元组的元素都是vec
        for triplet in triplet_pubid[real_name]:  # 遍历所有的三元组
            t_vec.append([train_pubvec[triplet[0]], train_pubvec[triplet[1]], train_pubvec[triplet[2]]])
        triplet_pubvec[real_name] = t_vec
        with open("triplet_sample/triplet_sample2vec/" + real_name+".json", "w", encoding="UTF-8") as rf:
            json.dump(triplet_pubvec, rf, indent=4)
        count=count+1
        print(count)
#convert_tripet_pubid2vec()
