import step7_model
import torch.optim as optim
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Mydataset(Dataset):  # 这里定义了一个数据集类，协助数据迭代函数
    def __init__(self, file_name):
        super(Mydataset, self).__init__()
        self.name = file_name  # 生成多少个点（多少个数据）

        def get_triplet(file_name):  # 给定一个文件名曾，获取里面所有的数据
            with open("triplet_sample/triplet_sample2vec/" + file_name, "r", encoding="UTF-8") as rf:
                data = json.load(rf)[file_name.split(".json")[0]]
            return data

        self.data = torch.tensor(get_triplet(file_name))
        self.num = len(self.data)

    # indexing
    def __getitem__(self, index):
        return self.data[index]

    # 返回数据集大小，应该是（x_transpose,y_transpose）大小即num*2，这里我直接返回了num
    def __len__(self):
        return self.num


def train_one_step(model, optimizer, triplet):
    optimizer.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True)
    for i in range(len(triplet)):  # 对一个batch'中的每一个三元组遍历
        triplet_anchor = triplet[i][0]
        triplet_pos = triplet[i][1]
        triplet_neg = triplet[i][2]
        pred_anchor = model(triplet_anchor)
        pred_pos = model(triplet_pos)
        pred_neg = model(triplet_neg)
        # print(step7_model.computer_loss(pred_pos, pred_neg, pred_anchor))
        loss = loss + step7_model.computer_loss(pred_pos, pred_neg, pred_anchor)
    loss.backward()
    optimizer.step()
    return loss.detach()


if __name__ == '__main__':
    model = step7_model.LogisticRegression(100, 64)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    triplet_files = os.listdir("triplet_sample/triplet_sample2vec")  # 遍历每一个文件，读取其中的三元组作为train集合
    save_path = "triplet_model_ckp/"
    losses = []
    for i in range(5):  # 每个训练集训练epoch遍
        losses_Epoch = []
        for file_index, file in enumerate(triplet_files):  # 对于每个文件遍历，分别读取数据训练
            triplet_data_set = Mydataset(file)
            # print(triplet_data_set.__len__())
            losses_file = []
            triplet_loader = DataLoader(triplet_data_set, batch_size=16, shuffle=True, num_workers=11)
            count = 0
            for triplet in triplet_loader:  # 对每一个batch的三元组进行训练batch*3
                count = count + 1
                print("\r", "Epoch {} File {}".format(i, file_index), file,
                      " Load >>>{}%".format(1600 * count / triplet_data_set.__len__()), end="", flush=True)
                loss = train_one_step(model, optimizer, triplet)
                losses_Epoch.append(loss.numpy().tolist())  # yi定要注意转换
                losses_file.append(loss)
            print(torch.tensor(losses_file).mean())
        losses.append(losses_Epoch)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": i}  # 定义保存的内容
        Epoch_save_path = save_path + "tripletLoss_model_Epoch" + str(i) + ".pth"
        torch.save(state, Epoch_save_path)
        for i in range(len(losses)):  # 将所有的损失值全部写进一个文档里
            loss_dict = {}  # key 是epoch,value是loss数组
            loss_dict[i] = losses[i]
            with open("triplet_model_ckp/losses_Epoch" + str(i) + ".json", "w", encoding="UTF-8") as wf:
                json.dump(loss_dict, wf, indent=4)


