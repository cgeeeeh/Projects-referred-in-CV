from sklearn.cluster import DBSCAN
import json


def dbscan_estimate():  # 使用dbscan进行预测
    with open("triplet_traned_result/triplet_vali.json", "r", encoding="UTF-8") as rf:
        vali_pub_data = json.load(rf)
    with open("OAG-v2-track1/valid/sna_valid_author_raw.json", "r", encoding="UTF-8") as rf:
        vali_author_data = json.load(rf)
    for real_name in vali_author_data:  # 遍历每一个真名
        x_data=[]
        for pub_id in vali_author_data[real_name]:#遍历每一篇论文id
            x_data.append(vali_pub_data[pub_id])
        #print(x_data)
        d_machine = DBSCAN(eps=2, min_samples=5).fit_predict(x_data)
        print(d_machine)
        exit()

dbscan_estimate()