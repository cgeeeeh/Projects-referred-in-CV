import make_class
import json
import gensim
from sklearn.cluster import DBSCAN
import numpy as np

if __name__ == '__main__':
    """对数据预处理，将结果写到pub.dict里面"""
    # processor = make_class.process_data()
    with open("OAG-v2-track1/valid/sna_valid_pub.json", "r", encoding="UTF_8") as rf:
        raw_pub_data = json.load(rf)  # 所有的粗糙论文数据
    #processor.process_all_pubids(raw_pub_data, "pub_dict.json")

    """根据做出来的字典，制作coauthor 与orgword到pubid的字典"""
    with open("pub_dict.json", "r", encoding="UTF_8") as rf:
        pub_dict = json.load(rf)  # 经过处理过的论文数据
    # processor.make_2pubids_dict(pub_dict, "name2pubids.json", "orgword2pubids.json")

    # """验证一下停用词"""
    # with open("orgword2pubids.json", "r", encoding="UTF_8") as rf:
    #     org_dict = json.load(rf)
    # for word in org_dict:
    #     org_dict[word] = len(org_dict[word])
    # print(sorted(org_dict.items(), key=lambda x: x[1], reverse=True))

    """开始随机游走的表演"""
    # walker = make_class.path_maker(5, 20)
    with open("OAG-v2-track1/valid/sna_valid_author_raw.json", "r", encoding="UTF_8") as rf:
        author_data = json.load(rf)  # 验证集的作者合集
    with open("pub_dict.json", "r", encoding="UTF_8") as rf:
        pubs_data = json.load(rf)  # pub_dict
    with open("name2pubids.json", "r", encoding="UTF_8") as rf:
        coauthor_data = json.load(rf)  # 所有的合作者到pubid的向量
    with open("orgword2pubids.json", "r", encoding="UTF_8") as rf:
        orgword_data = json.load(rf)  # 所有的机构单词到pubid的向量
    #
    # walker.whole_walk("path.txt", author_data, pubs_data, coauthor_data, orgword_data)

    """将所有的语料进行训练"""
    # vec_machine=make_class.path2vec("path.txt")
    # vec_machine.train_data()

    """将训练的结果作用到所有的数据上"""
    # model = gensim.models.Word2Vec.load("word2vec.model")
    # writer = make_class.transform_and_write(model)
    # writer.tran_one_cluster(pubs_data, "new_pub_data.json")

    """计算每两个矩阵的相似程度"""
    with open("new_pub_data.json", "r", encoding="UTF_8") as rf:
        new_pub_data = json.load(rf)
    with open("word2vec_result/valipub_word2vec_result.json", "r", encoding="UTF_8") as rf:
        word2vec_pub_data = json.load(rf)

    return_dict = {}
    for real_name in author_data:  # 遍历每一个真名，制作该真名下的相似度矩阵
        print(real_name)
        # 先计算相似度矩阵
        caltor = make_class.similarity_calculator()
        walk_m = caltor.make_cos_matrix_real_name(author_data, real_name, new_pub_data)
        word2vec_m = caltor.make_cos_matrix_real_name(author_data, real_name, word2vec_pub_data)
        final_m = (np.array(walk_m) + np.array(word2vec_m)) / 2
        print(final_m)
        # 再对矩阵进行聚类
        pre = DBSCAN(eps=0.1, min_samples=4, metric="precomputed").fit_predict(final_m)
        # 聚类的结果转换成一个list
        part_dict_maker = make_class.make_final_cluster_dict()
        part_list = part_dict_maker.make_final_dict_real_name(pre, real_name, author_data, pub_dict)
        return_dict[real_name] = part_list

    with open("final_result.json", "w", encoding="UTF_8") as wf:
        json.dump(return_dict, wf, indent=4)
