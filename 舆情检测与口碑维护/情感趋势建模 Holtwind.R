##清除工作环境
cat("\014")
rm(list = ls())
setwd("D:\\面试准备\\在线实习项目\\基于网络舆情数据的口碑分析与预测\\TASK7-情感趋势建模与预测")
dbemotion = read.csv("emotion.csv")
dbemotion$commenttime = as.Date(dbemotion$commenttime, format = "%Y/%m/%d")

library(dplyr)
time_dat = group_by(dbemotion, commenttime)

douban = summarise(time_dat, mean(emotion))
names(douban) = c("Date", "meanemo")

library(zoo)
douban$meanemo = rollmean(douban$meanemo, k = 5, 'center')
douban = douban[!is.na(douban$meanemo),]

library(ggplot2)
ggplot(douban, aes(x = Date, y = meanemo)) +
  scale_x_date(breaks = "1 month") +
  geom_line() + geom_point(size = 1) +
  theme_classic() + labs(x = "", y = "豆瓣短评日均正向情感得分") +
  theme(axis.text.x = element_text(
    angle = 45,
    hjust = 0.85,
    vjust = 0.9
  ))


library(tseries)
library(forecast)
SSE1 = c()
for (i in 2:30) {
  db_ts = ts(douban$meanemo, frequency = i)
  hotfit = HoltWinters(db_ts, seasonal = "additive")
  SSE1 = c(SSE1, hotfit[["SSE"]])
}
N = which(SSE1 == min(SSE1)) + 1
##找到了N，开始建模
db_ts = ts(douban$meanemo, frequency = N)
holtfit = HoltWinters(db_ts, seasonal = "additive")
para = c(holtfit[['alpha']], holtfit[['beta']], holtfit[['gamma']])
para

## 接下来基于Holt-winters模型进行预测
holtforecast <- forecast(holtfit, h = 14)
## 储存预测结果
l=length(douban$meanemo)
holtdata <-
  data.frame(index = seq(1, l),
             value = douban$meanemo,
             type = '样本情感得分') %>%
  rbind(data.frame(
    index = seq(l+1, (l + 14)),
    value = as.numeric(holtforecast[["mean"]]),
    type = 'Holt-winters模型预测结果'
  ))
print(holtdata)
## 绘制出预测的时间序列图
ggplot(data = holtdata) +
  geom_line(mapping = aes(x = index, y = value, color = type)) +
  geom_point(mapping = aes(x = index, y = value, color = type)) + theme_light() +
  xlim(0, 250) + ylim(0, 1) + labs(x = '时间单位', y = '日平均情感得分', color = '数据类别') +
  theme(legend.position = "bottom",
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20))
