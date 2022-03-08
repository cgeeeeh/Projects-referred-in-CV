import json
from multiprocessing import Process
from multiprocessing import Manager
import math


def gen_idf_dict(my_pub_data, part_word_set, i, l, return_dict):
    part_word_idf_dict = {}  # 存放部分单词的idf值的地方

    for word in part_word_set:  # 对集合中的每一个单词计算idf值
        in_count = 0
        for pub in my_pub_data:  # 遍历每一篇论文
            vec_pub = my_pub_data[pub]
            if word in vec_pub:
                in_count = in_count + 1
        idf = math.log10(l / in_count)

    # result_dict[str(i)]=
    return


def get_part_word_set(word_set, i):  # 返回子list，为并行做准备
    start = 32111 * i
    if i == 11:
        end = 385339
    else:
        end = 32111 * (i + 1)
    return word_set[start:end]


if __name__ == '__main__':
    wt = open("global_word_set.txt", "r", encoding="UTF-8")
    word = wt.readline()
    word_set = []
    while word:
        word = word.strip()
        word_set.append(word)
        word = wt.readline()
    print(len(word_set))
    with open("pub_word_vector.json", "r", encoding="UTF-8") as pf:
        my_pub_data = json.load(pf)
    l = len(my_pub_data)

    # 开启多进程，并行地为每个word计算idf值，每个计算的结果放在一个字典里，字典放进一个新的字典
    single_l = int(len(word_set) / 12)  # 单个进程需要完成的任务数量（最后一个进程除外）
    print(single_l)
    process_list = []  # 存放开启的进程
    manager = Manager()
    result_dict = manager.dict()
    for i in range(12):
        part_word_set = get_part_word_set(word_set, i)
        p = Process(target=gen_idf_dict, args=(my_pub_data, part_word_set, i, l, result_dict,))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    #
    # word_idf = {}
    # for word in word_set:  # 遍历每个单词，计算idf
    #     # log（文档总数量/包含该单词的文档数量）
    #     idf = cal_idf(my_pub_data, word)
    #     word_idf[word] = idf
