# coding: utf-8
import chardet
# 获取文件编码类型
def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


file_name =r'C:\Users\cgeeeeh\Desktop\金融技术与实践\Scrath_Model\Data\Pure_Data.txt'
encoding = get_encoding(file_name)
print(encoding)
