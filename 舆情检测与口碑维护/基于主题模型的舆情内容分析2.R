##清除工作环境
cat("\014")
rm(list = ls())
##加载程序包
library(ggplot2)   # 用于绘图
library(jiebaR)    # 用于分词
library(jiebaRD)    # 用于分词
library(stringr)    # 用于文本处理
library(dplyr)     # 用于文本处理
library(lda)      #用于lda建模
library(readxl)
library(plyr)
library(reshape2)
setwd("D:\\面试准备\\在线实习项目\\基于网络舆情数据的口碑分析与预测\\TASK5-主题模型深入分析")
wball = read_excel("小丑.xlsx", col_names = T)
names(wball) <-
  c('ID',
    'content',
    'posttime',
    'agree',
    'rank',
    'follower',
    'enrolltime',
    'filmseen')
head(wball)

#将content转换成字符型变量
wball$content = as.character(wball$content)
#去除content中末尾的字符型数据
wball$content = str_replace_all(wball$content, pattern = '<(.+?)>', '')
##定义分词器
cutter = worker(bylines = T,
                user = 'userwords.txt',
                stop_word = 'stopwords.txt')
words = cutter[wball$content]

##LDA建模
vocab = unique(unlist(words))
#词频序号信息
get.terms = function(x) {
  #为每个词找到在此表中对应的位置
  index = match(x, vocab)
  index = index[!is.na(index)]
  #为每个位置的词赋予1的频数
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
# document每一项为一份文本内部的词频序号信息
documents <- lapply(words, get.terms)
#设置随机种子
#set.seed(123)
#设置主题数目
K <- 3
ite <- 1000 #迭代次数
result <-
  lda.collapsed.gibbs.sampler(documents, K, vocab, ite, 0.1, 0.1, compute.log.likelihood =
                                TRUE)
# 查看每个主题内部的高频词
top.words <- top.topic.words(result$topics, 10, by.score = TRUE)
top.words

# 计算每一条文本内部的各主题比例
topic.proportions <-
  t(result$document_sums + 0.1) / colSums(result$document_sums + 0.1)
# 这里加alpha是为了把主题词频次转化为文本主题概率
head(round(topic.proportions,4))

#计算每个主题的概率平均值
Topic_p <- apply(topic.proportions, 2, mean)
tp = data_frame(Topic = c('演员表现', '电影情节', '社会内涵'), Topic_p)                              #把Topic_p变为数据框的格式
tp$Topic_p <-
  round(tp$Topic_p, 3) #对频率保留4位小数
head(tp$Topic_p)
windowsFonts(myFont = windowsFont("楷体"))                #绑定字体

#使用ggplot绘图
ggplot(tp, aes(x = Topic, y = Topic_p, fill = Topic)) +      #绘制各主题占比的玫瑰图
  geom_bar(stat = "identity", alpha = 0.7) +
  coord_polar() +
  theme_bw() +
  labs(x = "", y = "", title = "") +
  geom_text(aes(
    y = Topic_p / 2 + max(Topic_p) / 4,
    label = Topic_p,
    color = Topic
  ), size = 10) +
  # 加上数字
  theme(axis.text.y = element_blank()) +                  #去掉左上角的刻度标签
  theme(axis.text.x = element_text(
    size = 20,
    family = "myFont",
    face = "bold"
  )) +
  theme(axis.ticks = element_blank()) +                   #去掉左上角的刻度线
  theme(panel.border = element_blank()) +                 #去掉外层边框
  theme(legend.position = "none") +                       #去掉图例
  theme(title = element_text(
    vjust = -56,
    face = "bold",
    family = "myFont"
  )) +
  scale_fill_manual(values = c("#F38181", "#FCE38A", "#EAFFD0", '#95E1D3'))

#开始绘制星级交叉分析图
wball$rank = factor(wball$rank, levels = c("很差", "较差", "还行", "推荐", "力荐"))
#将所有数据按照评分分组
#第一步是将数据拼接到dataframe中
document.sums = data.frame(t(result$document_sums))
document.sums$rank = wball$rank
colnames(document.sums) = c('演员表现', '电影情节', '社会内涵', '星级')
temp = ddply(document.sums, .(星级), function(x) {
  colSums(x[, 1:3])
})
row.names(temp) = temp$星级
temp = t(temp[,-1])
ratio = data.frame(matrix(NA, 3, 5))
row.names(ratio) = c("演员表现", "电影情节", "社会内涵")
colnames(ratio) = c("很差", "较差", "还行", "推荐", "力荐")
for (i in 1:5) {
  ratio[, i] = temp[, i] / colSums(temp)[i]
}
ratio = melt(
  ratio,
  measure.vars = c("很差", "较差", "还行", "推荐", "力荐"),
  variable.name = "星级",
)
ratio$主题  = rep(c("演员表现", "电影情节", "社会内涵"), 5)

#因子化
ratio$星级 = factor(ratio$星级, levels = c("很差", "较差", "还行", "推荐", "力荐"))
ratio$主题 = factor(ratio$主题, levels = c("演员表现", "电影情节", "社会内涵"))

##绘制堆叠柱状图
ggplot(ratio, aes(x = 星级, y =value, fill = 主题)) +
  # 横轴为时段，按照主题百分比排列
  geom_bar(stat = 'identity', position = 'fill') +
  # 添加坐标轴标签
  labs(x = "评分", y = "文本平均主题概率", fill = "主题") +
  # 设置柱子颜色
  # 调整画板主题及字体大小
  theme(
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 20),
    panel.background = element_rect(fill = "transparent"),
    panel.border = element_rect(fill = 'transparent', color = 'transparent'),
    axis.line = element_line(color = "black"),
    legend.title=element_text(size=20),
    legend.text=element_text(size=20)
  ) +
  scale_fill_manual(values = c("#F38181", "#FCE38A", "#EAFFD0", '#95E1D3'))

