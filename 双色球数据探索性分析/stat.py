#!/usr/bin/python3

import fileinput
import sys
import os
import pandas as pd

blue = {}
red = {}
for line in fileinput.input(sys.argv[1]):
    line = line.strip()
    infor = line.split('\t')
    for item in infor[1:7]:
        if item in blue:
            blue[item] = blue[item] + 1
        else:
            blue[item] = 1
    if infor[-1] in red:
        red[infor[-1]] = red[infor[-1]] + 1
    if infor[-1] not in red:
        red[infor[-1]] = 1
fileinput.close()

blue_data = []
blue_labels= []

for key in sorted(blue.keys()):
    blue_labels.append(key)
    blue_data.append(blue[key])
blue_1 = pd.DataFrame({'Number':blue_labels,'Frequence':blue_data})
blue_1.to_csv('blue_data.csv',index=False)

red_data = []
red_labels = []

for key1 in sorted(red.keys()):
    red_labels.append(key1)
    red_data.append(red[key1])
red_1 = pd.DataFrame({'Number':red_labels,'Frequence':red_data})
red_1.to_csv('red_data.csv',index=False)
r = open('distribution.r','w')
r_script = '''data = read.csv(\"%s\",header=T,stringsAsFactors=TRUE)
png('blue_number_distribution.png',width=1200,height=600)
dis<-barplot(data$Frequence,names.arg=data$Number,ylim=c(0,500),width=0.75,xlim=c(1,35),xlab='blue number',ylab='each blue number frequence',main='blue number distribution')
text(dis,data$Frequence,labels=data$Frequence,cex=1,pos=1)
abline(h=0)
dev.off()

data1=read.csv(\"%s\",header=T,stringsAsFactors=TRUE)
png('red_number_distribution.png',width=600,height=600)
dis_red <- barplot(data1$Frequence,names.arg=data1$Number,ylim=c(0,200),xlim=c(0,17),width=0.75,xlab='red number',ylab='each red number frequence',main='red number distribution')
text(dis_red,data1$Frequence,labels=data1$Frequence,cex=1,pos=1)
abline(h=0)
dev.off()
'''%('blue_data.csv','red_data.csv')
r.write(r_script+'\n')
r.close()
try:
    os.system('/usr/bin/Rscript distribution.r')
except Exception as e:
    print(e)
