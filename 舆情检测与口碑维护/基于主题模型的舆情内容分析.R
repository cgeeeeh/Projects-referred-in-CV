##清除工作环境
cat("\014")
rm(list = ls())
setwd("D:\\面试准备\\在线实习项目\\基于网络舆情数据的口碑分析与预测\\TASK4 基于主题模型的舆情内容分析")
##加载程序包
library(ggsci)
library(readxl)
library(ggplot2)   # 用于绘图
library(jiebaR)    # 用于分词
library(jiebaRD)    # 用于分词
library(stringr)    # 用于文本处理
library(dplyr)     # 用于文本处理
library(lda)      #用于lda建模
library(LDAvis)   #用于LDA结果展示
wball = read_excel('小丑_中评.xlsx', col_names = T)
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

word_topic <-
  data.frame('topic' = c('演员表现', '电影情节', '社会内涵'),
             'num' = result[["topic_sums"]])
mylevel = factor(c('社会内涵', '演员表现', '电影情节'),
                 levels = c('社会内涵', '演员表现', '电影情节'))
word_topic$topic = factor(word_topic$topic, levels = mylevel)
word_topic = word_topic[order(word_topic$num),]
#制作饼图上要出现的比例
label_value <-
  paste('(', round(word_topic$num / sum(word_topic$num) * 100, 1), '%)', sep = '')
label <- paste(word_topic$topic, label_value, sep = '')
#计算标签的纵值
y_pos = word_topic$num / 2 + c(0, cumsum(word_topic$num)[-length(word_topic$num)])

ggplot(word_topic, aes(
  x = '',
  y = num,
  fill = topic,
  alpha = 0.8
)) +
  geom_bar(stat = 'identity', position = 'stack') +
  # 这里coord_polar是做极坐标变换,这样就能变成饼图了
  coord_polar(theta = 'y') +
  theme_void() +
  theme(legend.position = "none",axis.title.y = element_text(size=20)) +
  # # 这里修改配色方案
  scale_color_npg() + scale_fill_npg() +
  #scale_fill_manual(values = wes_palette("Royal2")) +
  geom_text(aes(
    #y = y_pos,
    y = c(y_pos[1], y_pos[2], y_pos[3]),
    x = 1,
    label = label,
    
  ),size = 9)


##使用LDA进行可视化
alpha <- 0.10
eta <- 0.02
theta <-
  t(apply(result$document_sums + alpha, 2, function(x)
  {
    x / sum(x)
  }))  #文档—主题分布矩阵
phi <-
  t(apply(t(result$topics) + eta, 2, function(x)
    x / sum(x)))  #主题-词语分布矩阵
term.table <- table(unlist(words))
#这里有两步，unlist用于统计每个词的词频；table把结果变成一个交叉表式的factor
term.table <- sort(term.table, decreasing = TRUE) #按照词频降序排列
term.frequency <- as.integer(term.table)   #词频
doc.length <-
  sapply(documents, function(x)
    sum(x[2, ])) #每篇文章的长度，即有多少个词
json <- createJSON(
  phi = phi,
  theta = theta,
  doc.length = doc.length,
  vocab = vocab,
  term.frequency = term.frequency
)
#json为作图需要数据，下面用servis生产html文件，通过out.dir设置保存位置
serVis(json, out.dir = './AAA')
writeLines(
  iconv(readLines("./AAA/lda.json"), from = "GBK", to = "UTF-8"),
  file("./AAA/lda.json", encoding = "UTF-8")
)
