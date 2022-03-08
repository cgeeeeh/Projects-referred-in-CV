

##清除工作环境
cat("/014")
rm(list = ls())
setwd("D:\\面试准备\\在线实习项目\\基于网络舆情数据的口碑分析与预测\\TASK6-情感倾向分析")

##
library(rjson)
library(openxlsx)
library("httr")

###鉴权
url2 = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=CNpigYy8iGizr4beUy1MxOoi&client_secret=59Es8DjPlywNlEB4Lb92sUmEOrjWrrYh&"
fromJSON(file = url2)
#access token：24.c986089319e71f632e6b638f8e4fc6e0.2592000.1647332778.282335-25603357



##数据读取
wball = read.xlsx("小丑.xlsx")
##转化为时间格式
wball$commenttime = as.Date(wball$commenttime, format = "%Y/%m/%d")
##删除空缺值
wball = wball[!is.na(wball$short),]
##提取出两列
wball2 = data.frame(wball$commenttime, wball$short)
colnames(wball2) = c("commenttime", "short")
##数据清洗
library(stringr)
wball2$short = str_replace_all(wball2$short, pattern = '#(.+?)#', '')
wball2$short = str_replace_all(wball2$short, pattern = '<(.+?)>', '')
wball2$short = str_trim(wball2$short)
wball2 = wball2[!wball2$short == "", ]
wball2$short = str_sub(wball2$short, 1L, 256L)
##构建URL
emotion1 = numeric(nrow(wball2))
headers = c('Content-Type' = 'application/json')
for (i in 1:nrow(wball2)) {
  print(i)
  payload <- list('text' = wball2$short[i])
  #print(payload)
  url <-
    "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token=24.c986089319e71f632e6b638f8e4fc6e0.2592000.1647332778.282335-25603357&charset=UTF-8"
  raw2 <-
    POST(url,
         add_headers(.headers = headers),
         body = payload,
         encode = c('json'))
  tmp <- fromJSON(rawToChar(content(raw2, "raw")))
  #print(tmp[3]$items[[1]][1])
  emotion1[i] <- tmp[3]$items[[1]][1]
  Sys.sleep(0.5)
}

wball2$emotion = unlist(emotion1)
write.csv(wball2, "emotion.csv", row.names = F)


##读取情感分数数据
cat("/014")
rm(list = ls())
emotion = read.csv("emotion.csv")
emotion$commenttime = as.Date(emotion$commenttime)
time_long = unique(emotion$commenttime)
time_scale=seq.Date(from=as.Date("2019/8/31",format="%Y/%m/%d"),by="5 days",length.out = 40)
##计算每天的平均分数
#输入下标i，计算时间段i，i+1内数据的平均值
mean.of.period <- function(i) {
  scores = emotion[emotion$commenttime>=time_scale[i] & emotion$commenttime<time_scale[i+1],3]
  return (mean(scores))
}
score.of.period = numeric(length(time_scale))
for (i in 1:length(time_scale)) {
  score.of.period[i]=mean.of.period(i)
}
dat0=data.frame("day"=time_scale,
                "score"=score.of.period)
dat0=dat0[-40,]


##绘制时间序列图
library(ggplot2)
ggplot(dat0,aes(x=day,y=score))+geom_line()+theme_classic()+
  scale_x_date(date_breaks = "1 month")+
  theme(axis.text.x = element_text(angle=45,hjust=1),
        axis.title.y =element_text(size=20),
        axis.title.x =element_text(size=20))+
  xlab("时间指标")+ylab("正向情感分数")
