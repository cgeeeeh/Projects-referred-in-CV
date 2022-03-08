import json
import re
import numpy as np
import re
from nltk.corpus import stopwords

"""将论文数据特征化"""

"""辅助函数"""


# 对author_name 进行清洗
def clean_name(name, flag=0):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    # x = [k.strip() for k in name.lower().strip().replace("-", "").replace("_", ' ').split()]
    full_name = ' '.join(x)
    name_part = full_name.split()
    if (len(name_part) >= 1):
        return full_name
    else:
        return None


def dic_keynum(d):  # 获取字典长度
    return len(d.keys())


def name_over5(author_data):  # 从作者数据中挑选所有“同名作者数大于等于5个”的名字（用于训练）
    name_train = set()
    # 筛选训练集，只取同名作者数大于等于5个的名字作为训练集。
    # for key in author_data:
    #     print(author_data[key])
    for name in author_data:
        persons = author_data[name]
        if (dic_keynum(persons) > 5):
            name_train.add((name))
    return name_train


""""P9a1gcvg": {
        "authors": [
            {
                "name": "Fenghe Qiu",
                "org": "Institute of Pharmacology and Toxicology"
            },
        ],
        "title": "Rapid determination of central nervous drugs in plasma by solid-phase extraction and GC-FID and GC-MS",
        "abstract": "Objective: To establish a simultaneous determination method of central nervous drugs including barbitals, benzodiazepines, phenothiozines and tricyclic antidepressants in human plasma. Methods: Drugs in plasma were extracted and purified by using X-5 resin solid phase extraction columns, followed by identification and quantitation using capillary GC-FID and GC-MS. Results: More than 20 drugs were simultaneously extracted from human plasma, and effectively separated in GC and TIC spectra. The correlation coefficient of standard curves was larger than 0.99, and relative standard differences (RSD) were less than 10% for most of the drugs. Under neutral extraction conditions, the lowest detection limits of barbitals were in the range of 2 ~ 5 \u03bcg/ml, in optimized conditions, they were reduced to 0.3 ~ 0.5 \u03bcg/ml. Conclusion: X-5 resin solid-phase extraction is fit for the simultaneous extraction and purification of large number of drugs in plasma, therefore it is satisfactory for rapid determination of overdose drugs clinically, when combined with capillary GC and GC-MS.",
        "keywords": [
            "central nervous drugs",
            "GC-FID",
            "GC-MS",
            "solid-phase extraction",
            "X-5 resin"
        ],
        "venue": "Chinese Pharmaceutical Journal",
        "year": 1996,
        "id": "P9a1gcvg"
    },"""


def proc_string(s, stop_words):  # 对文本信息的处理
    s = s.strip().lower()
    s = re.sub('[^a-z]', ' ', s)  # 替换所有非英文的单词
    s = [k.strip() for k in s.split()]
    for st in stop_words:  # 移除所有的停用词
        while st in s:
            s.remove(st)
    return [k if len(k) >= 3 else None for k in s]


def document_to_vec(pub_data, doc_id):  # 根据论文id找到论文，并将其所有的单词拼接成一个向量
    stop_words = stopwords.words('english')
    try:
        doc_dic = pub_data[doc_id]
    except:  # pub_train 中没有该论文id
        print(doc_id)
        return None
    vec_doc = []  # 存储所有的单词
    for key in doc_dic:
        if key == "authors":  # 如果key是author的话，对其调用clean name 函数
            if type(doc_dic[key]) == type([]):  # key author有可能是一个数组（对应一篇论文有多个作者的情况），也有可能是字典，对应一个作者的情况
                for author_dic in doc_dic[key]:  # 处理所有的作者姓名和机构信息
                    cleaned_name = clean_name(author_dic['name'])
                    # print(cleaned_name)
                    vec_doc.append(cleaned_name)
                    try:  # 处理org信息
                        org_info = proc_string(author_dic['org'], stop_words)
                        # print(org_info)
                        vec_doc.extend(org_info)
                    except:  # 有的论文只有一个空壳子，里面除了名字信息什么都没有
                        pass
        # if None in vec_doc:
        #     print("author  ",vec_doc)
        #     exit()
        if key in ("title", "abstract"):  # 处理论文的标题与摘要信息
            if type(doc_dic[key]) == type("a"):  # 判断论文的title与abstract是否是string变量
                try:
                    text_info = proc_string(doc_dic[key], stop_words)
                    # print(text_info)
                    vec_doc.extend(text_info)
                except:
                    print("tezhenghua title abstract error ", doc_id)
                    exit()
        # if None in vec_doc:
        #     print(key,"  ",vec_doc)
        #     exit()
        if key == "venue":  # 处理论文的期刊信息
            if type(doc_dic[key]) == type("a"):
                try:
                    venue_info = proc_string(doc_dic[key], stop_words)
                    # print(venue_info)
                    vec_doc.extend(venue_info)
                except:
                    # print("tezhenghua venue error ", doc_id)
                    exit()
        # if None in vec_doc:
        #     print(key,"  ",vec_doc)
        #     exit()
        if key == "keywords":  # 处理论文关键字
            if type(doc_dic[key]) == type([]):
                try:
                    for key_w in doc_dic[key]:
                        keyw_info = proc_string(key_w, stop_words)
                        vec_doc.extend(keyw_info)
                        # print(keyw_info)
                except:
                    print("tezhenghua keyword error ", doc_id)
                    exit()
        # if None in vec_doc:
        #     print(key,"  ",vec_doc)
        #     exit()
        if key == "year":
            if type(doc_dic[key]) != type("a"):
                try:
                    year_info = str(doc_dic[key])
                    vec_doc.append(year_info)
                    # print(year_info)
                except:
                    print("tezhenghua year error ", doc_id)
                    exit()
            else:
                vec_doc.append(doc_dic[key])
                # print(doc_dic[key])
        # if None in vec_doc:
        #     print(key,"  ",vec_doc)
        #     exit()
    return vec_doc


def get_train_pub_num(name_train, author_data):  # 获得训练集中论文的总数量
    count = 0
    for name_real in name_train:  # 对训练集中的所有作者名字，找到所有“该作者”名下的论文，对他们进行特征化
        for name_id in author_data[name_real]:
            for pub_id in author_data[name_real][name_id]:
                count = count + 1
    return count


def move_null_list_None(vec_doc):  # 除去单词化向量中的空字符串和list型的内容
    while "" in vec_doc:
        vec_doc.remove("")
    while None in vec_doc:
        vec_doc.remove(None)
    for e in vec_doc:
        if type(e) == type([]):
            vec_doc.remove(e)
            vec_doc.extend(e)


def check_vec(vec_doc):
    for e in vec_doc:
        if e == "":
            print(e, "isnull")
            return 0
        if type(e) != type("a"):
            print(e, "not string")
            print(vec_doc[vec_doc.index(e) - 2:vec_doc.index(e) + 4])
            return 0


if __name__ == '__main__':
    with open("OAG-v2-track1/train_author.json", "r", encoding="UTF-8") as f_author:
        author_data = json.load(f_author)
    with open("OAG-v2-track1/train_pub.json", "r", encoding="UTF-8") as f_pub:
        pub_data = json.load(f_pub)

    name_train = name_over5(author_data)
    # print(len(name_train), name_train)
    # print("a\n     ".strip())
    # print("a   b  c".split())
    # print("".strip())
    # a = ['a']
    # a.extend([])
    # print(a)
    pub_num = get_train_pub_num(name_train, author_data)
    print("一共有{}篇论文需要特征化".format(pub_num))
    current_veced_num = 0  # 记录当前已经特征化了多少篇论文
    repeated_pub_num = 0
    global_word_set = []  # 存储所有的字
    pub_word_vector = {}  # 字典，key是pub_id，值是vec_doc

    for name_real in name_train:  # 对训练集中的所有作者名字，找到所有“该作者”名下的论文，对他们进行特征化
        for name_id in author_data[name_real]:
            for pub_id in author_data[name_real][name_id]:
                vec_doc = document_to_vec(pub_data, pub_id)
                if vec_doc != None:  # 成功生成了vec_doc
                    move_null_list_None(vec_doc)  # 移除vec_doc中的非法元素（None，List，空字符串）
                    if check_vec(vec_doc) == 0:  # 对上述操作的效果检查通过
                        exit()
                    print("\r", "Loading... {}%".format(100 * current_veced_num / pub_num), end="", flush=True)
                    current_veced_num = current_veced_num + 1

                    # for word in vec_doc:#填充word字典
                    #     if word not in global_word_set:
                    #         global_word_set.append(word)

                    if pub_id not in pub_word_vector:  # 制作word_vec字典,考虑论文被重复特征化的现象
                        pub_word_vector[pub_id] = vec_doc
                    else:
                        repeated_pub_num = repeated_pub_num + 1

    print("重复出现了{}篇论文".format(repeated_pub_num))
    # sorted(global_word_set, key=lambda x: len(x))
    # print(len(global_word_set), global_word_set)

    # 将list与jason数据写入文件中
    with open("pub_word_vector.json", 'w', encoding="UTF-8") as wf:
        json.dump(pub_word_vector, wf, indent=4)

    # with open("global_word_set.txt",'w',encoding="UTF-8") as wt:
    #     for word in global_word_set:
    #         wt.write(word+"\n")
