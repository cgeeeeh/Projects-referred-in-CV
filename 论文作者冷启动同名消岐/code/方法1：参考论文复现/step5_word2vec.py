import gensim
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import pandas as pd
import logging
import json
import Cython
from gensim.models import KeyedVectors
import numpy as np


def write_line_txt():
    with open("pub_word_vector.json", "r", encoding="UTF-8") as rf:
        my_pub_data = json.load(rf)
    with open("line_data_train.txt", "w", encoding="UTF-8") as wt:
        count = 0
        for pub in my_pub_data:
            result = ""
            vec = my_pub_data[pub]
            for word in vec:
                result = result + word + " "
            result = result.strip() + "\n"
            wt.write(result)
            count = count + 1


def concat_line_data():  # 将line_data_train与line_data_vali合并，制作最终的gensim word2vec语料集合
    train_concat = []  # 存储所有的train和valid信息
    with open("line_data_train.txt", "r", encoding="UTF-8") as rt:
        line = rt.readline()
        while line:
            train_concat.append(line)
            line = rt.readline()
    print(len(train_concat))
    with open("line_data_vali.txt", "r", encoding="UTF-8") as rt:
        line = rt.readline()
        while line:
            train_concat.append(line)
            line = rt.readline()
    print(len(train_concat))
    with open("line_data_concat.txt", "w", encoding="UTF-8") as wt:
        for line in train_concat:
            wt.write(line)
    with open("line_data_concat.txt", "r", encoding="UTF-8") as rt:
        count = 0
        line = rt.readline()
        while line:
            count = count + 1
            line = rt.readline()
        print(count)


# write_line_txt()
# concat_line_data()


def word2vec_train():
    # inp为输入语料
    inp = 'line_data_concat.txt'
    sentences = gensim.models.word2vec.LineSentence(inp)
    path = get_tmpfile("word2vec.model")  # 创建临时文件
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, sg=1, iter=50, workers=11)
    model.save("word2vec.model")
    model.wv.save("model.wv")


#word2vec_train()
# #加载方式2
#     t1 = time.time()
#     model = KeyedVectors.load_word2vec_format('word2vec.vector')
#     t2 = time.time()
#     print(len(model.vectors))
#     print(".vector load time %.4f" % (t2 - t1))
def test_word2vec_model():  # 测试训练出的模型
    wv = KeyedVectors.load("model.wv", mmap='r')
    print(wv["computer"], type(wv["computer"]),len(wv["simple"]))
    print(wv.most_similar("simple"))


test_word2vec_model()


def generate_pub_vec_dic(flag):  # 对每篇论文 生成专属向量
    wv = KeyedVectors.load("model.wv", mmap='r')
    if flag == "train":  # 对训练集，每篇论文生成专属向量
        with open("pub_word_vector.json", "r", encoding="UTF-8") as rf:
            my_pub_data = json.load(rf)
    elif flag == "vali":
        with open("pub_word_vector_vali.json", "r", encoding="UTF-8") as rf:
            my_pub_data = json.load(rf)
    l = len(my_pub_data)  # 论文数量
    count = 0  # 计数，处理了多少论文
    word2vec_dict = {}
    for pub_id in my_pub_data:  # 遍历每一篇论文
        vec = my_pub_data[pub_id]
        pub_vec = [0] * 100
        pub_len = 0  # 论文内被查找到的单词（未被查找到的单词不算在内）
        for word in vec:  # 遍历每一个单词
            try:
                word_vec = wv[word]
                for i in range(100):
                    pub_vec[i] = pub_vec[i] + word_vec[i]
                pub_len = pub_len + 1
            except:
                continue
        count = count + 1
        for i in range(100):
            pub_vec[i] = pub_vec[i] / pub_len
        # print(pub_vec)
        print("\r", "Loading .....{}%".format(100 * count / l), end="", flush=True)
        word2vec_dict[pub_id] = pub_vec
    if flag == "train":
        with open("word2vec_result/trainpub_word2vec_result.json", "w", encoding="UTF-8") as wf:
            json.dump(word2vec_dict, wf, indent=4)
    elif flag == "vali":
        with open("word2vec_result/valipub_word2vec_result.json", "w", encoding="UTF-8") as wf:
            json.dump(word2vec_dict, wf, indent=4)


# generate_pub_vec_dic("train")
# generate_pub_vec_dic("vali")
