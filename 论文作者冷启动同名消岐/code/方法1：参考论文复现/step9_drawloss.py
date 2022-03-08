import json
import matplotlib.pyplot as plt
import torch
import step7_model
import numpy as np
import sys
from sklearn.cluster import DBSCAN

np.set_printoptions(threshold=sys.maxsize)


def draw_loss_pic():  # 画出5次迭代的损失值曲线图
    loss = []
    for i in range(5):  # 遍历每一个损失文件
        path = "triplet_model_ckp/losses_Epoch{}.json".format(str(i))
        with open(path, "r", encoding="UTF-8") as rf:
            loss_part = json.load(rf)[str(i)]
            for index, e in enumerate(loss_part):
                if index % 1000 == 0:
                    loss.append(e / 16)
    x = range(len(loss))

    plt.plot(x, loss, color="blue")
    plt.show()


draw_loss_pic()


def transform_data(model, flag):  # 将数据传入model，输出的数据存储在文件里
    if flag == "train":  # 转化训练集
        with open("word2vec_result/trainpub_word2vec_result.json", "r", encoding="UTF-8") as rf:
            vec_data = json.load(rf)
    if flag == "vali":
        with open("word2vec_result/valipub_word2vec_result.json", "r", encoding="UTF-8") as rf:
            vec_data = json.load(rf)

    transformed_dict = {}  # 转化后的字典 key是pubid，value是转化后的数值
    count = 0
    l = len(vec_data)
    for pub_id in vec_data:
        pub_vec = vec_data[pub_id]  # 需要去转化的word2vec向量
        tran_result = model(torch.tensor(pub_vec)).detach().numpy().tolist()  # 转化后的结果
        transformed_dict[pub_id] = tran_result
        count = count + 1
        print("\r", "Trans {} Loading >>> {}%".format(flag, 100 * count / l), end="", flush=True)

    if flag == "train":
        with open("triplet_traned_result/triplet_train.json", "w", encoding="UTF-8") as wf:
            json.dump(transformed_dict, wf, indent=4)
    if flag == "vali":
        with open("triplet_traned_result/triplet_vali.json", "w", encoding="UTF-8") as wf:
            json.dump(transformed_dict, wf, indent=4)


def triplet_feature2pubid(flag, triplet_feature):  # 通过triplet net后的特征找到对应的pubid
    if flag == "train":  # 训练集
        with open("triplet_traned_result/triplet_train.json", "r", encoding="UTF-8") as rf:
            triplet_feature_dict = json.load(rf)
    if flag == "vali":  # 训练集
        with open("triplet_traned_result/triplet_vali.json", "r", encoding="UTF-8") as rf:
            triplet_feature_dict = json.load(rf)
    for pub_id in triplet_feature_dict:
        if triplet_feature_dict[pub_id] == triplet_feature:  # 找到了
            return pub_id


# if __name__ == '__main__':
#     # 加载模型
#     model = step7_model.LogisticRegression(100, 64)
#     torch.load("triplet_model_ckp/tripletLoss_model_Epoch4.pth")
#     model.load_state_dict(torch.load("triplet_model_ckp/tripletLoss_model_Epoch4.pth")["model"])
#     # print(torch.load("triplet_model_ckp/tripletLoss_model_Epoch4.pth")["model"]["b1"].numpy())
#     # print(torch.load("triplet_model_ckp/tripletLoss_model_Epoch4.pth")["model"]["b2"].numpy())
#     model.eval()
#     # 将vec 100维的数据通过model转换为64维的数据，写入文件中
#     transform_data(model, "train")
#     transform_data(model, "vali")
#     #进行聚类分析
