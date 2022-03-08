import json
import numpy as np
import random
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import get_tmpfile
import math
import re
from sklearn.metrics import pairwise_distances


class process_data():  # 数据的与处理工作
    def __init__(self):
        pass

    def parse_authorname(self, name):
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

    def parse_org(self, org):
        org = org.lower().strip()
        org = re.sub('[^a-z]', ' ', org)
        org = [k.strip() for k in org.split()]
        stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
                    'is', 'are', 'can', 'university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science',
                    'international', 'technology', 'key', 'sciences', 'research', 'center', 'state', 'national',
                    'province', 'hospital', 'engineering', 'shanghai', 'guangzhou', 'nanjing', 'engineering']
        for word in stopword:
            while word in org:
                org.remove(word)
        for word in org:
            if len(word) <= 2:
                org.remove(word)
        return org

    def parse_venue_or_title(self, venue):  # 这套流程对title与venue都是用
        venue = venue.lower().strip()
        venue = re.sub('[^a-z]', ' ', venue)
        venue = [k.strip() for k in venue.split()]
        stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
                    'is', 'are', 'can', 'university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science',
                    'international', 'technology', 'key', 'sciences', 'research', 'center', 'state', 'national',
                    'province', 'hospital', 'engineering', 'shanghai', 'guangzhou', 'nanjing', 'engineering']
        for word in stopword:
            while word in venue:
                venue.remove(word)
        for word in venue:
            if len(word) <= 2:
                while word in venue:
                    venue.remove(word)
        return venue

    def process_pub(self, raw_pub_data, pub_id):  # 处理一个pub_id数据
        names = raw_pub_data[pub_id]["authors"]  # 拿出所有的作者数据
        name_datas = []
        org_datas = []

        for name_dict in names:
            name_data = self.parse_authorname(name_dict["name"])
            org_data = self.parse_org(name_dict["org"])
            if name_data not in name_datas:
                name_datas.append(name_data)
            for word in org_data:
                if word not in org_datas:
                    org_datas.append(word)
        venue_datas = self.parse_venue_or_title(raw_pub_data[pub_id]["venue"])
        title_datas = self.parse_venue_or_title(raw_pub_data[pub_id]["title"])
        # print(title_datas)
        return name_datas, org_datas, venue_datas, title_datas

    def process_all_pubids(self, raw_pub_data, file):  # 处理所有的数据
        pub_dict = {}
        for pub_id in raw_pub_data:
            name_datas, org_datas, venue_datas, title_datas = self.process_pub(raw_pub_data, pub_id)
            pub_dict[pub_id] = {}
            pub_dict[pub_id]["author"] = name_datas
            pub_dict[pub_id]["orgword"] = org_datas
            pub_dict[pub_id]["venue"] = venue_datas
            pub_dict[pub_id]["title"] = title_datas
        with open(file, "w", encoding="UTF_8") as wf:
            json.dump(pub_dict, wf, indent=4)

    def all_names(self, pub_dict):  # 找到所有的name
        names = []
        for pub_id in pub_dict:
            for name in pub_dict[pub_id]["author"]:
                if name not in names:
                    names.append(name)
        return names

    def all_orgwords(self, pub_dict):  # 找到所有的name
        orgwords = []
        for pub_id in pub_dict:
            for word in pub_dict[pub_id]["orgword"]:
                if word not in orgwords:
                    orgwords.append(word)
        return orgwords

    def name2pubids(self, pub_dict, name):  # 通过名字找到pubid
        pub_ids = []  # 存放所有的pubid，每个的author里都有name
        for pub_id in pub_dict:  # 遍历每一个pubid，看看name有没有在他的author俩民
            if name in pub_dict[pub_id]["author"]:
                pub_ids.append(pub_id)
        return pub_ids

    def orgword2pubids(self, pub_dict, orgword):  # 通过orgword找到pubid
        pub_ids = []  # 存放所有的pubid，每个的author里都有name
        for pub_id in pub_dict:  # 遍历每一个pubid，看看name有没有在他的author俩民
            if orgword in pub_dict[pub_id]["orgword"]:
                pub_ids.append(pub_id)
        return pub_ids

    def make_2pubids_dict(self, pub_dict, file_name, file_orgword):  # 制作字典 name2pubids
        name2pubids_dict = {}
        orgword2pubids_dict = {}
        names = self.all_names(pub_dict)
        orgwords = self.all_orgwords(pub_dict)

        for name in names:
            pub_ids = self.name2pubids(pub_dict, name)
            name2pubids_dict[name] = pub_ids
        for orgword in orgwords:
            pub_ids = self.orgword2pubids(pub_dict, orgword)
            orgword2pubids_dict[orgword] = pub_ids

        with open(file_name, "w", encoding="UTF_8") as wf:
            json.dump(name2pubids_dict, wf, indent=4)
        with open(file_orgword, "w", encoding="UTF_8") as wf:
            json.dump(orgword2pubids_dict, wf, indent=4)


class path_maker():  # 生成pub 路径
    def __init__(self, walk_epoch, walk_length):
        self.walk_epoch = walk_epoch  # 要走几遍
        self.walk_length = walk_length  # 要走多长

    def one_step_walk(self, real_name, pub_id, pubs_data, coauthor_data, orgword_data):  # 只游走一步
        final_path = ""  # 要返回的字符串，pubid空格pubid空格
        # print(pub_id)
        pub0_authors = pubs_data[pub_id]["author"]  # 起点论文的作者群
        pub0_orgwords = pubs_data[pub_id]["orgword"]  # 七点论文的机构单词群
        next_pubid = None

        l_authors = len(pub0_authors)  # 作者群的数量
        if l_authors >= 2:  # 作者群要有至少两个
            target_author_name = pub0_authors[random.randrange(l_authors)]  # 找到下一个作者名字（author游走路线）
            while target_author_name == real_name:  # 不能找realname啊这是犯规的
                target_author_name = pub0_authors[random.randrange(l_authors)]  # 找到下一个作者名字（author游走路线）

            try:
                pub_id_candidate = coauthor_data[target_author_name]  # 不一定能找到到
                l_choice = len(pub_id_candidate)  # 有多少可以被选择的论文id
                if l_choice >= 2:  # 至少要有两个可供选择的id
                    pubid_chosen = pub_id_candidate[random.randrange(l_choice)]  # 选择一篇论文id
                    while pubid_chosen == pub_id:  # 不能选回去
                        pubid_chosen = pub_id_candidate[random.randrange(l_choice)]  # 选择一篇论文id
                    final_path = final_path + pubid_chosen + " "
                    next_pubid = pubid_chosen
            except:
                pass

        l_orgwords = len(pub0_orgwords)  # 机构单词群的数量
        # 机构单词可以只有一个
        if l_orgwords:
            target_orgword = pub0_orgwords[random.randrange(l_orgwords)]  # 找到下一个作者名字（author游走路线）
            try:
                pub_id_candidate = orgword_data[target_orgword]  # 不一定能找到到
                l_choice = len(pub_id_candidate)  # 有多少可以被选择的论文id
                if l_choice >= 2:  # 至少要有两个可供选择的id
                    pubid_chosen = pub_id_candidate[random.randrange(l_choice)]  # 选择一篇论文id
                    while pubid_chosen == pub_id:  # 不能选回去
                        pubid_chosen = pub_id_candidate[random.randrange(l_choice)]  # 选择一篇论文id
                    final_path = final_path + pubid_chosen + " "
                    next_pubid = pubid_chosen
            except:
                pass

        return final_path, next_pubid

    def one_whole_walk(self, real_name, pub0_id, pubs_data, coauthor_data, orgword_data):  # 完成一整个游走的链条
        whole_walk = pub0_id + " "  # 一开始的时候要带上他
        next_pubid = pub0_id
        for i in range(self.walk_length):  # 要走多长?
            add_str, next_pubid = self.one_step_walk(real_name, next_pubid, pubs_data, coauthor_data,
                                                     orgword_data)  # 走一步!
            if add_str == "" and next_pubid is None:  # 停止游走了啊别nm走了
                return whole_walk
            whole_walk = whole_walk + add_str
        return whole_walk + "\n"

    def realname_whole_walk(self, author_data, real_name, pubs_data, coauthor_data,
                            orgword_data):  # 对一个realname下的所有论文疯狂游走
        final_paths = []  # 存储一个realname下所有的path
        for pub_id in author_data[real_name]:  # 对于每一个realname名下的论文id
            pub_id_path = self.one_whole_walk(real_name, pub_id, pubs_data, coauthor_data, orgword_data)
            #
            final_paths.append(pub_id_path)
        return final_paths

    def whole_walk(self, file_path, author_data, pubs_data, coauthor_data,
                   orgword_data):  # 对所有的realname进行游走，还要有走好几遍
        final_paths = []
        for epoch in range(self.walk_epoch):  # 要走多少遍啊
            for real_name in author_data:  # 遍历所有的真名
                add_path = self.realname_whole_walk(author_data, real_name, pubs_data, coauthor_data, orgword_data)
                final_paths.extend(add_path)

        with open(file_path, "w", encoding="UTF_8") as wt:
            for line in final_paths:
                wt.write(line)
        return


class path2vec():  # 将所有的path转化为向量的形式
    def __init__(self, file_path):
        self.file_path = file_path

    def train_data(self, ):  # 训练数据啊
        input_data = LineSentence(self.file_path)
        path = get_tmpfile("word2vec.model")  # 创建临时文件
        model = Word2Vec(input_data, size=100, negative=5, min_count=2, window=5)
        model.save("word2vec.model")


class transform_and_write():  # 将转化后的pub数据写入文件里面
    def __init__(self, model):
        self.model = model

    def tran_one_cluster(self, pubs_data, file):  # 转化一个路径为向量
        pub_dict = {}  # key 是pubid value是转化后的数据
        for pub_id in pubs_data:  # 遍历每一个pubid
            pub_dict[pub_id] = self.model[pub_id].tolist()
        with open(file, "w", encoding="UTF_8") as wf:
            json.dump(pub_dict, wf, indent=4)


class similarity_calculator():
    def __init__(self):
        pass

    def cos_sim(self, vector_a, vector_b):
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def cal(self, pub_id1, pub_id2, pub_dict):  # 计算两个pubid的相似度
        data1 = np.array([pub_dict[pub_id1]])
        data2 = np.array([pub_dict[pub_id2]])
        return pairwise_distances(data1, data2, metric="cosine")[0][0]

    def make_cos_matrix_real_name(self, author_data, real_name, pub_dict):
        matrix = []
        for i, pub_id1 in enumerate(author_data[real_name]):
            row = []
            for j, pub_id2 in enumerate(author_data[real_name]):
                if pub_id1 == pub_id2:  # 万群一样的两篇论文
                    row.append(1)
                else:
                    row.append(self.cal(pub_id1, pub_id2, pub_dict))
            matrix.append(row)
        return matrix


class make_final_cluster_dict():  # 在分类结果出来后（pred）制作最终的分类字典dict
    def __init__(self):
        pass

    def make_first_dict(self, pred, real_name, author_data):  # 将分类的结果初步做一个字典出来
        first_dict = {}  # key值是-1，0，1，2，3。。。。。。。value值是pubid列表
        for pub_index, pub_id in enumerate(author_data[real_name]):
            label = pred[pub_index]  # label是该论文所属于的粗
            if str(label) not in first_dict:  # 加入进去
                first_dict[str(label)] = [pub_id]
            else:
                first_dict[str(label)].append(pub_id)
        return first_dict

    def tanimoto(self, p, q):
        c = [v for v in p if v in q]
        if len(c) != 0:
            return float(len(c) / (len(p) + len(q) - len(c)))
        else:
            return 0

    def extract_similarity(self, pubid1, pubid2, pub_dict):  # 挖掘两篇论文的相似度
        similarity = 0
        dict1 = pub_dict[pubid1]
        dict2 = pub_dict[pubid2]
        # 先对名字进行比较
        author1 = dict1["author"]
        author2 = dict2["author"]
        similarity = similarity + 1.5 * (self.tanimoto(author1, author2))
        # 再对venue进行比较
        venue1 = dict1["venue"]
        venue2 = dict2["venue"]
        similarity = similarity + self.tanimoto(venue1, venue2)
        # 再对org进行比较
        org1 = dict1["orgword"]
        org2 = dict2["orgword"]
        similarity = similarity + self.tanimoto(venue1, venue2)
        # 再对title及逆行比较
        title1 = dict1["title"]
        title2 = dict2["title"]
        similarity = similarity + self.tanimoto(title1, title2) / 3.0
        return similarity

    def find_home(self, alone_pub_id, first_dict, pub_dict):  # 为离群点找到他们的家
        maxest_similarity = 0
        maxest_key = "-1"
        for key in first_dict:  # 遍历每一个簇
            if key != "-1":  # 不能是离群点这个粗
                pub_ids = first_dict[key]
                max_similarity = 0
                for pub_id in pub_ids:  # 遍历每一个论文，看看最高的相似度是多少
                    similarity = self.extract_similarity(alone_pub_id, pub_id, pub_dict)
                    if similarity > max_similarity:  # 看看谁比较大，选出这个簇中的最佼佼者
                        max_similarity = similarity
                if max_similarity > maxest_similarity:  # 干掉了当前的最优者
                    maxest_similarity = max_similarity
                    maxest_key = key
        if maxest_similarity > 1.5:
            return maxest_key
        else:
            return "-1"

    def make_final_dict_real_name(self, pred, real_name, author_data, pub_dict):  # 制作最终的字典
        first_dict = self.make_first_dict(pred, real_name, author_data)
        print(first_dict)
        for pub_id in first_dict["-1"]:  # 考虑所有的离群点
            new_key = self.find_home(pub_id, first_dict, pub_dict)
            print(new_key, end=" ")
            if new_key != "-1":  # 找到了新东家
                first_dict["-1"].remove(pub_id)  # 从原有的粗中去除
                first_dict[new_key].append(pub_id)  # 放到新的地方去

        final_list_one_name = []
        for key in first_dict:
            if key != "-1":
                final_list_one_name.append(first_dict[key])
            else:
                for pub_id in first_dict[key]:
                    final_list_one_name.append([pub_id])
        print("")
        return final_list_one_name
