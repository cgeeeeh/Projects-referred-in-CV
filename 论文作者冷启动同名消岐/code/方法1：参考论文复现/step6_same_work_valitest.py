import step1_pre_pro
import step5_word2vec
import json


def same_work_vali_gen_wordlist():
    with open("OAG-v2-track1/valid/sna_valid_author_raw.json", "r", encoding="UTF-8") as f_author:
        author_data_vali = json.load(f_author)
    with open("OAG-v2-track1/valid/sna_valid_pub.json", "r", encoding="UTF-8") as f_pub:
        pub_data_vali = json.load(f_pub)

    print("一共有{}篇论文需要特征化".format(len(pub_data_vali)))
    current_veced_num = 0  # 记录当前已经特征化了多少篇论文
    repeated_pub_num = 0
    pub_word_vector = {}  # 字典，key是pub_id，值是vec_doc

    for name_real in author_data_vali:  # 对训练集中的所有作者名字，找到所有“该作者”名下的论文，对他们进行特征化
        for pub_id in author_data_vali[name_real]:
            vec_doc = step1_pre_pro.document_to_vec(pub_data_vali, pub_id)
            if vec_doc != None:  # 成功生成了vec_doc
                step1_pre_pro.move_null_list_None(vec_doc)  # 移除vec_doc中的非法元素（None，List，空字符串）
                if step1_pre_pro.check_vec(vec_doc) == 0:  # 对上述操作的效果检查通过
                    exit()
                print("\r", "Loading... {}%".format(100 * current_veced_num / len(pub_data_vali)), end="", flush=True)
                current_veced_num = current_veced_num + 1

                if pub_id not in pub_word_vector:  # 制作word_vec字典,考虑论文被重复特征化的现象
                    pub_word_vector[pub_id] = vec_doc
                else:
                    repeated_pub_num = repeated_pub_num + 1

    print("重复出现了{}篇论文".format(repeated_pub_num))

    # 将list与jason数据写入文件中
    with open("pub_word_vector_vali.json", 'w', encoding="UTF-8") as wf:
        json.dump(pub_word_vector, wf, indent=4)


# same_work_vali_gen_wordlist()

def same_work_vali_write_line_txt():  # 将pubdata写到txt文件中，方便训练器读取，一行一句句子，每个单词用 隔开
    with open("pub_word_vector_vali.json", "r", encoding="UTF-8") as rf:
        my_pub_data = json.load(rf)
    with open("line_data_vali.txt", "w", encoding="UTF-8") as wt:
        count = 0
        for pub in my_pub_data:
            result = ""
            vec = my_pub_data[pub]
            for word in vec:
                result = result + word + " "
            result = result.strip() + "\n"
            wt.write(result)
            count = count + 1
#same_work_vali_write_line_txt()


