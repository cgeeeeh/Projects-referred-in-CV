import step10_RNNsamples
import torch
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


class estimate_cluster():
    def __init__(self, rnn_model):
        self.rnn_model = rnn_model

    def sample_validata(self, real_name, vali_author_data, vali_wordvec_data):  # 从validata中抽样出一个300*100的样本
        pubids = vali_author_data[real_name]  # 去除所有的pubid集合
        chosen_data = []
        sample_pool = []
        for pub_id in pubids:
            sample_pool.append(pub_id)
        chosen_pub_id = np.random.choice(sample_pool, 300, replace=True)  # 被选中的300*100的数据
        for pub_id in chosen_pub_id:
            chosen_data.append(vali_wordvec_data[pub_id])
        return chosen_data

    def sample_whole_dataset(self, vali_author_data, vali_wordvec_data):  # 对整个vali数据集进行采样
        whole_sample = []
        for time in range(20):  # 重复采样20次
            print(time)
            time_sample = []
            for real_name in vali_author_data:  # 遍历验证作者集合中的每一个作者真名
                chosen_data = self.sample_validata(real_name, vali_author_data, vali_wordvec_data)  # 抽出一个300*100的data
                time_sample.append(chosen_data)
            whole_sample.append(time_sample)  # 20*real_name_num*300*100
        return whole_sample

    def estimate_cluster_num(self, vali_author_data, vali_wordvec_data):  # 预测粗的数量
        whole_sample = self.sample_whole_dataset(vali_author_data, vali_wordvec_data)
        print(len(whole_sample), len(whole_sample[0]), len(whole_sample[0][0]), len(whole_sample[0][0][0]))
        estimate_result = []
        for epoch_data in whole_sample:  # real_name_num*300*100
            pred = torch.mean(self.rnn_model(torch.tensor(epoch_data, dtype=torch.float32)), dim=1)  # real_name_num*1
            estimate_result.append(pred.detach().numpy().tolist())
        estimate_result = np.array(estimate_result).mean(axis=0).tolist()  # real_num*1
        estimate_result = list(map(int, estimate_result))
        return estimate_result


class gen_cluster():  # 生成簇
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num

    def estimate(self, vali_author_data, vali_wordvec_data):  # 记得使用新的特征话结果 使用kmeans聚类分析
        x_data = {}  # 存储所有的待分类
        pub_data = {}  # 存储所有的论文名称
        for real_name in vali_author_data:  # 遍历验证集中的每一个真名
            data = []  # 每一个real name中所有的pubdata数据

            pub_ids = vali_author_data[real_name]  # 取出改名字下所有的pubid
            for pub_id in pub_ids:  # 对于改名字下的每一篇论文
                data.append(vali_wordvec_data[pub_id])  # 把他的特征放入data中去

            pub_data[real_name] = pub_ids
            x_data[real_name] = data
        # print(len(x_data), len(x_data["weiping_liu"]), len(x_data["weiping_liu"][0]))
        # print(pub_data)
        # print(len(pub_data["weiping_liu"]))

        return self.A_cluster(x_data, pub_data)

    def A_cluster(self, x_data, pub_data):  # 使用层次聚类方法进行聚类分析
        final_result = {}  # 一个字典，key是人的真名，value是一个list，元素是一个list，list里面的元素是被聚类在一起的所有pub_id
        for i, real_name in enumerate(x_data):  # 遍历每一个人名,对每个人名下的数据进行聚类分析
            A_machine = AgglomerativeClustering(n_clusters=self.cluster_num[i]).fit(x_data[real_name])
            # print(A_machine.labels_)
            # print(len(A_machine.labels_))
            pub_clusters = {}  # 里卖弄的每一个元素都是论文簇,key值是num，value值是pubid们
            for j, pub_idx in enumerate(A_machine.labels_):  # 对每一个index进行操作 i 是第几个元素，pub_idx是内容
                if str(pub_idx) not in pub_clusters:  # 需要新开一个key
                    pub_clusters[str(pub_idx)] = [pub_data[real_name][j]]
                else:
                    pub_clusters[str(pub_idx)].append(pub_data[real_name][j])

            final_result[real_name] = []
            for key in pub_clusters:
                final_result[real_name].append(pub_clusters[key])
        return final_result

    def check_result(self, final_result, vali_author_data):  # 检查最后的结果
        # 首先确保没有论文遗漏
        # 其次确保没有论文重复
        # 确保论文的确属于该真名
        for real_name in final_result:  # 遍历每一个真名
            count = 0  # 计数君
            for cluster in final_result[real_name]:  # 遍历每一个小簇
                count = count + len(cluster)
            if count != len(vali_author_data[real_name]):
                print("count 不对")
                exit()
            for cluster in final_result[real_name]:  # 遍历每一个簇
                for pub_id in cluster:
                    if pub_id not in vali_author_data[real_name]:  # 论文不属于这个作者
                        print("不属于这个作者 ")
            pub_idset = []
            for cluster in final_result[real_name]:
                for pub_id in cluster:
                    if pub_id not in pub_idset:
                        pub_idset.append(pub_id)
                    else:
                        print("重复了吧")


if __name__ == '__main__':
    with open("OAG-v2-track1/valid/sna_valid_author_raw.json", "r", encoding="UTF-8") as rf:
        vali_author_data = json.load(rf)
    with open("word2vec_result/valipub_word2vec_result.json", "r", encoding="UTF-8") as rf:
        vali_wordvec_data = json.load(rf)
    rnn_model = step10_RNNsamples.RNN_model(300, 100, 64, 1)
    rnn_model.load_state_dict(torch.load("LSTM_model_ckp/model_wizout_log2/LSTM_model_Epoch959.pth")["model"])
    rnn_model.eval()
    e_machine = estimate_cluster(rnn_model)
    cluster_num_pred = e_machine.estimate_cluster_num(vali_author_data, vali_wordvec_data)

    # 预测出簇的数量之后，进行真正的测试。使用triplet net得到的结果
    with open("triplet_traned_result/triplet_vali.json", "r", encoding="UTF-8") as rf:
        vali_triplet_data = json.load(rf)
    gen_vali_cluster = gen_cluster(cluster_num_pred)
    final_result=gen_vali_cluster.estimate(vali_author_data, vali_wordvec_data)
    print(final_result)
    gen_vali_cluster.check_result(final_result, vali_author_data)
    with open("estimate_result/final_result.json","w",encoding="UTF-8") as wf:
        json.dump(final_result,wf,indent=4)