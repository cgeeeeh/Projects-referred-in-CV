import json
import step1_pre_pro
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def real_name2pubnums(real_name, author_data):  # 计算出一个人名字下有多少篇论文
    count = 0
    for pub_id in author_data[real_name]:
        count = count + len(author_data[real_name][pub_id])
    return count


def detectless300():  # 寻找样本数量小于300的作者姓名，然而该函数没有意义
    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as rf:
        author_data = json.load(rf)
    name_set = step1_pre_pro.name_over5(author_data)
    for real_name in name_set:  # 遍历训练集合每一个名字，输出他们内的总论文数量
        count = real_name2pubnums(real_name, author_data)
        if count < 300:  # 不满足抽样条件
            print("warning", real_name)


class RNN_sampler(object):  # RNN神经网络的抽象类，负责抽样本提供给LSTM神经网络进行训练

    def __init__(self, name_set):  # 初始化
        self.sample_num = 300  # 对于每一个real name，我们抽样300个样本
        self.K = None  # 我们暂时不知道要从一个real name中抽出几个簇，这取决于real name中本身有几个簇
        self.real_names = name_set

    def sample_real_name(self, author_data, pub_data, real_name):  # 对一个real name 进行抽样：
        sample_result = []  # 这将是一个300*100的张量
        name_ids = author_data[real_name]  # 获取一个字典，key 是name_id, value是pub_id
        cluster_num = len(author_data[real_name])  # 这一个簇本身有多少个簇

        self.K = random.sample(range(5, cluster_num + 1), 1)[0]  # roll一个数字，决定从中抽取多少个簇
        # print(real_name, cluster_num, self.K)
        cluster_chosen = random.sample(range(cluster_num), self.K)  # 从所有簇中抽取下标，选择K个簇的下标
        samplepub_pool = []  # 这里存放所有被选中的簇中的所有pub_id

        for index, name_id in enumerate(name_ids):  # 遍历所有的簇
            if index in cluster_chosen:  # 当前簇（name id）被选中了
                samplepub_pool.extend(name_ids[name_id])  # 将该簇所有的样本pub_id 都放入样本池子里面

        pub_chosen = np.random.choice(samplepub_pool, self.sample_num, replace=True)
        # print(pub_chosen)
        for pub_id in pub_chosen:  # 遍历每一个被选中的pub_id
            sample_result.append(self.tran_pubid2wordvec(pub_data, pub_id))
        # print(sample_result)
        return sample_result, [self.K] * self.sample_num

    def sample_wholedataset(self, author_data, pub_data):  # 对整个数据集中的每一个真名进行抽样
        whole_sample = []  # real names数量 *300 *100 记得转换成tensor
        K_sample = []  # 存储对应的k值
        for real_name in self.real_names:  # 遍历每一个全名
            sample_result, K = self.sample_real_name(author_data, pub_data, real_name)
            # self.check_samples(author_data, real_name, sample_result)
            whole_sample.append(sample_result)
            K_sample.append(K)
        return torch.tensor(whole_sample), torch.tensor(K_sample)

    def check_samples(self, author_data, real_name, pub_chosen):  # 检查所有的sample结果是否在同一个簇里面
        name_ids = author_data[real_name]
        pub_ids = []  # 存储所有的pub_id
        for name_id in name_ids:  # 遍历每一个name_id
            pub_ids.extend(name_ids[name_id])
        for pub_id in pub_chosen:  # 遍历每一篇被选中的论文id
            if pub_id not in pub_ids:  # 采样的样本不在本该在的作者名下
                print(real_name, pub_id)
                exit()

    def tran_pubid2wordvec(self, pub_data, pub_id):  # 将pubid转换成wordvec
        return pub_data[pub_id]  # 返回一个100维的向量


class RNN_model(nn.Module):
    def __init__(self, sequence_size, input_size, hidden_size, output_size):
        super(RNN_model, self).__init__()
        self.sequence_size = sequence_size  # 输入的序列长度
        self.rnnoutput_size = hidden_size  # 经过lstm层之后的输出维度（单层）
        self.rnn_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                                   batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.decoder = nn.Linear(in_features=2 * hidden_size, out_features=output_size)
        # nn.init.uniform_(self.decoder.weight)

    def forward(self, batch_input):
        out_put, (h_n, c_n) = self.rnn_encoder(batch_input)
        out_put = self.drop(out_put)
        out_put.contiguous().view(-1, self.sequence_size, 2 * self.rnnoutput_size)
        out = self.decoder(out_put)
        return out.contiguous().view(-1, self.sequence_size)


class RNN_dataset(Dataset):  # RNN数据集类
    def __init__(self, x_data, y_data):
        super(RNN_dataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]


def compute_loss(pred, y):
    pred = torch.log(torch.clamp(pred, 1e-07, np.inf) + 1)
    y = torch.log(torch.clamp(y, 1e-07, np.inf) + 1)
    loss = torch.sqrt(torch.mean(torch.pow(pred - y,2), dim=-1))
    return loss.mean()


if __name__ == '__main__':
    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as rf:
        author_data = json.load(rf)
    with open("word2vec_result/trainpub_word2vec_result.json", "r", encoding="UTF-8") as rf1:
        pub_data = json.load(rf1)
    name_set = step1_pre_pro.name_over5(author_data)

    sampler = RNN_sampler(name_set)
    rnn = RNN_model(300, 100, 64, 1)
    optimizer = torch.optim.RMSprop(rnn.parameters(), lr=0.001)
    loss_fun = torch.nn.MSELoss()

    loss_Epoch = []  # 记录下一个epoch的损失值
    rnn.train()
    for Epoch in range(960):  # 要将整个数据集重复采样960遍
        one_sample, one_K = sampler.sample_wholedataset(author_data, pub_data)
        one_dataset = RNN_dataset(one_sample, one_K)
        one_loader = DataLoader(one_dataset, batch_size=16, shuffle=True, num_workers=11)
        for index, (x, y) in enumerate(one_loader):  # 每次迭代一个batch的数据 16*300*100 16*300
            optimizer.zero_grad()
            pred = rnn(x)  # 向前传播
            pred = pred.float()
            y = y.float()
            loss = compute_loss(pred, y)
            loss.backward()
            optimizer.step()
            print("\rEpoch {} batch {} loss {}".format(Epoch, index, loss), end="", flush=True)
            if Epoch % 10 == 0:  # 每隔10次迭代将loss值压入栈,同时画出损失函数曲线
                loss_Epoch.append(loss.detach().numpy().tolist())
                # 将模型保存，写入所有的损失值
                loss_dict = {"loss": loss_Epoch}
                with open("LSTM_model_ckp/model_wizout_log/losses.json", "w", encoding="UTF-8") as wf:
                    json.dump(loss_dict, wf, indent=4)

            if Epoch % 120 == 119:  # 每120个迭代将模型保存一次
                savepath = "LSTM_model_ckp/model_wizout_log/LSTM_model_Epoch" + str(Epoch) + ".pth"
                state_dict = {"model": rnn.state_dict(), "optimizer": optimizer.state_dict(), "Epoch": Epoch}
                torch.save(state_dict, savepath)

        # 画出损失函数曲线
        if Epoch % 10 == 0:
            plt.plot(range(len(loss_Epoch)), loss_Epoch)
            print(loss_Epoch)
            plt.draw()
            plt.pause(0.5)

