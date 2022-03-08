##清除工作环境
cat("\014")
rm(list = ls())
##加载程序包
library(readxl)
library(ggplot2)
library(jiebaR)
library(wordcloud2)
##设置工作路径
setwd("D:\\meeting\\在线实习项目\\基于网络舆情数据的口碑分析与预测\\TASK3-舆情热度分析")
##读入文件
douban = read_excel("小丑_差评.xlsx", col_names = T)
colnames(douban) =
  c(
    "ID",
    "short",
    "commenttime",
    "votes",
    "commend",
    "followed",
    "enrolldate",
    "filmseen"
  )
##转换时间格式，筛选某段时间内的数据
douban$commenttime = as.Date(douban$commenttime)
douban.use = douban[douban$commenttime >= as.Date('2019/10/01') &
                      douban$commenttime <= as.Date('2020/3/01'),]
##用rgb函数控制颜色
ggplot(douban.use, aes(x = commenttime)) +
  geom_density(fill = rgb(178, 200, 187, maxColorValue = 255)) +
  theme_classic() +
  scale_x_date(breaks = "2 week", limits = as.Date(c('2019/10/01', '2020/3/01'))) +
  theme(axis.text.x = element_text(
    angle = 45,
    hjust = 0.85,
    vjust = 0.9
  )) +
  labs(x = '', y = "豆瓣短评密度")+
  theme(axis.title.y = element_text(size=20))

##绘制词云
##将热度较大的时间段的数据跳出来
douban.hot = douban.use[douban.use$commenttime >= as.Date('2019/10/01') &
                          douban.use$commenttime <= as.Date('2020/3/01'),]
##生成分词器，并且安排其分词任务
cutter = worker(bylines = T,
                user = "userwords.txt",
                stop_word = "stopwords.txt")
douban.comment = cutter[douban.hot$short]
##将分词的结果统计词频，生成table 并排序
douban.table = sort(table(unlist(douban.comment)), decreasing = T)[1:70]
douban.table = data.frame(douban.table)
douban.wc = wordcloud2(
  douban.table,
  size = 0.5,
  shape = 'circle',
  color = 'random-light',
  backgroundColor = 'white',
  fontFamily = '微软雅黑'
)
douban.wc
