import step10_RNNsamples
import json
import step1_pre_pro
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import step11_draw_evaluate
if __name__ == '__main__':  # 继续训练
    step11_draw_evaluate.draw_LSTM_wizoutlog()
    exit()
    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as rf:
        author_data = json.load(rf)
    with open("word2vec_result/trainpub_word2vec_result.json", "r", encoding="UTF-8") as rf1:
        pub_data = json.load(rf1)
    name_set = step1_pre_pro.name_over5(author_data)

    sampler = step10_RNNsamples.RNN_sampler(name_set)
    rnn = step10_RNNsamples.RNN_model(300, 100, 64, 1)
    rnn.load_state_dict(torch.load("LSTM_model_ckp/model_wizout_log/LSTM_model_Epoch959.pth")["model"])
    optimizer = torch.optim.RMSprop(rnn.parameters(), lr=0.005,momentum=0.05)
    loss_fun = torch.nn.MSELoss()

    loss_Epoch = []  # 记录下一个epoch的损失值
    rnn.train()
    for Epoch in range(960):  # 要将整个数据集重复采样960遍
        one_sample, one_K = sampler.sample_wholedataset(author_data, pub_data)
        one_dataset = step10_RNNsamples.RNN_dataset(one_sample, one_K)
        one_loader = DataLoader(one_dataset, batch_size=16, shuffle=True, num_workers=11)
        for index, (x, y) in enumerate(one_loader):  # 每次迭代一个batch的数据 16*300*100 16*300
            optimizer.zero_grad()
            pred = rnn(x)  # 向前传播
            pred = pred.float()
            y = y.float()
            loss = step10_RNNsamples.compute_loss(pred, y)
            loss.backward()
            optimizer.step()
            print("\rEpoch {} batch {} loss {}".format(Epoch, index, loss), end="", flush=True)
            if Epoch % 10 == 0:  # 每隔10次迭代将loss值压入栈,同时画出损失函数曲线
                loss_Epoch.append(loss.detach().numpy().tolist())
                # 将模型保存，写入所有的损失值
                loss_dict = {"loss": loss_Epoch}
                with open("LSTM_model_ckp/model_wizout_log2/losses.json", "w", encoding="UTF-8") as wf:
                    json.dump(loss_dict, wf, indent=4)

            if Epoch % 120 == 119:  # 每120个迭代将模型保存一次
                savepath = "LSTM_model_ckp/model_wizout_log2/LSTM_model_Epoch" + str(Epoch) + ".pth"
                state_dict = {"model": rnn.state_dict(), "optimizer": optimizer.state_dict(), "Epoch": Epoch}
                torch.save(state_dict, savepath)

        # 画出损失函数曲线
        if Epoch % 10 == 0:
            plt.plot(range(len(loss_Epoch)), loss_Epoch)
            print(loss_Epoch)
            plt.draw()
            plt.pause(0.5)
