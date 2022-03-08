import step10_RNNsamples
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import step1_pre_pro


def draw_LSTM_wizoutlog():  # 画出非loglstm自编码器的损失函数图像
    with open("LSTM_model_ckp/model_wizout_log2/losses.json", "r", encoding="UTF-8") as rf:
        loss = json.load(rf)["loss"]
    plt.plot(range(len(loss)), loss)
    plt.show()


def sample_train(name_set, train_author_data, train_pub_data):  # 测试一下训练集
    all_sample = []
    for real_name in name_set:  # 遍历每一个真名
        samples = []
        pubs_pool = []
        name_ids = train_author_data[real_name]
        for name_id in name_ids:  # 遍历每一个nameid
            pubs_pool.extend(name_ids[name_id])
        indexes = np.random.choice(pubs_pool, 300, replace=True)
        for index in indexes:  # 每一个被选中的下标
            samples.append(train_pub_data[index])
        all_sample.append(samples)
    return all_sample


if __name__ == '__main__':
    draw_LSTM_wizoutlog()
    exit()
    rnn = step10_RNNsamples.RNN_model(300, 100, 64, 1)  # 生成一个rnn模型
    rnn.eval()
    rnn.load_state_dict(torch.load("LSTM_model_ckp/model_wizout_log2/LSTM_model_Epoch959.pth")["model"])

    with open("word2vec_result/valipub_word2vec_result.json", "r", encoding="UTF-8") as rf:
        vali_pub_data = json.load(rf)

    with open("OAG-v2-track1/valid/sna_valid_author_raw.json", "r", encoding="UTF-8") as rf1:
        vali_author_data = json.load(rf1)

    with open("word2vec_result/trainpub_word2vec_result.json", "r", encoding="UTF-8") as rf2:
        train_pub_data = json.load(rf2)

    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as rf3:
        train_author_data = json.load(rf3)

    name_set = step1_pre_pro.name_over5(train_author_data)
    # draw_LSTM_wizoutlog()  # 效果还是挺让人满意的
    all_preds=[]
    count=0
    for time in range(20):
        count=count+1
        train_samples = sample_train(name_set, train_author_data, train_pub_data)
        preds = []
        for input_data in train_samples:  # 遍历训练集中每一个被选中的data
            pred = rnn(torch.tensor([input_data], dtype=torch.float32)).mean()
            preds.append(int(pred.detach().numpy().tolist()))
        all_preds.append(preds)
        print(count)
    print(np.array(all_preds).mean(axis=0))

    ys = []
    for real_name in name_set:  # 遍历每一个人名输出种类数字
        ys.append(len(train_author_data[real_name]))
    print(ys)

    minus=[]
    for i in range(len(ys)):
        minus.append(np.array(all_preds).mean(axis=0)[i]-ys[i])
    print(minus)

