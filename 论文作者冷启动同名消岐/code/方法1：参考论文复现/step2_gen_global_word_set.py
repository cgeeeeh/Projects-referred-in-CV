import json
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing import Manager


def year_count():  # 统计论文的出版年份有多少种不同的值
    with open('train_pub.json', 'r', encoding="UTF-8") as rpub:
        pub_data = json.load(rpub)

    year_set = []
    for pub in pub_data:
        if pub_data[pub]["year"] not in year_set:
            year_set.append(pub_data[pub]["year"])
    print(len(year_set), year_set)


def init(l):
    global lock
    lock = l





def word_add(i, part_my_pub_data, return_dict):
    current_pub_num=0
    l=len(part_my_pub_data)
    global_word_set = []
    for pub_id in part_my_pub_data:
        vec_word = part_my_pub_data[pub_id]
        for word in vec_word:
            if word not in global_word_set:
                global_word_set.append(word)
        print("\r", "Loading ...{}%".format(100 * current_pub_num / l)+str(i), end="", flush=True)
        current_pub_num=current_pub_num+1
    return_dict[str(i)] = global_word_set
    print("process",i,"done")

    return 1


def get_part_my_pub_data(my_pub_data, i):
    part_my_pub_data = {}
    start = i * 10351
    if i == 11:
        end = 124216
    else:
        end = (i + 1) * 10351 - 1
    num = 0
    for key in my_pub_data:
        if num >= start and num <= end:
            part_my_pub_data[key] = my_pub_data[key]
        num = num + 1
    return part_my_pub_data


if __name__ == '__main__':
    with open('pub_word_vector.json', 'r', encoding="UTF-8") as pf:
        my_pub_data = json.load(pf)
    print("有{}篇论文的特征需要处理".format(len(my_pub_data)))
    pub_ids = list(my_pub_data.keys())
    print(int(len(pub_ids) / 12))

    global_word_set = []
    process_list = []
    manager = Manager()
    return_dict = manager.dict()
    for i in range(12):  # 开启12个子进程执行word_add函数
        part_my_pub_data = get_part_my_pub_data(my_pub_data, i)  # 切分一个子字典
        p = Process(target=word_add, args=(i, part_my_pub_data, return_dict,))  # 实例化进程对象，让该进程操作一个子字典
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    
    with open("global_word_set.txt","w",encoding="UTF-8") as wt:
        for key in return_dict:
            print(key, len(return_dict[key]))
            for word in return_dict[key]:
                wt.write(word+"\n")
        print(len(return_dict))
    # # for pub in my_pub_data:
    # #     vec_word = my_pub_data[pub]
    # #     for word in vec_word:
    # #         # print(word)
    # #         if word not in global_word_set:
    # #             global_word_set.append(word)
    # #     current_pub_num = current_pub_num + 1
    #
    # sorted(global_word_set, key=lambda x: len(x))
    # print(len(global_word_set), global_word_set)
    #
    # # with open("global_word_set.txt", "w", encoding="UTF-8") as wt:
    # #     for word in global_word_set:
    # #         wt.write(word + "\n")
