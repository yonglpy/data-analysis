 ##-*- coding: utf-8 -*-。

import re
import urllib.request
import xlrd
import xlwt
import openpyxl
from time import sleep
import random
import datetime


my_header=["Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6 GTB5","Mozilla/5.0 (Windows NT 6.0) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.120 Safari/535.2",
           "Mozilla/5.0 (Windows NT 6.0; rv:14.0) Gecko/20100101 Firefox/14.0.1","Mozilla/5.0 (Windows NT 6.0; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.75 Safari/535.7 ",
           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36 SE 2.X MetaSr1.0"]

headers=("User-Agent","Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36 SE 2.X MetaSr1.0")
    
def getlink(url):
    opener=urllib.request.build_opener()
    opener.addheaders=[headers]
    urllib.request.install_opener(opener)
    file=urllib.request.urlopen(url)
    data=str(file.read())
    pat='(http://[^\s)";]+\.(\w|/)*.shtml)'
    link=re.compile(pat).findall(data)
    link=list(set(link))
    return link
url="http://kaijiang.500.com/shtml/ssq/17088.shtml"
linklist=getlink(url)
print(len(linklist))
def getnumber(url1):
    try:
        out=[]
        proxy={"http":"122.114.31.177:808"}
        random_header=random.choice(my_header)
        headers=("User-Agent",random_header)
        proxy_support=urllib.request.ProxyHandler(proxy)
        opener=urllib.request.build_opener(proxy_support)
        opener.addheaders=[headers]
        urllib.request.install_opener(opener)
        file1=urllib.request.urlopen(url1)
        data1=str(file1.read())
        contentpat='<div class="ball_box01">(.*?)</div>'
        userpat='<li class="ball_(red|blue)">(.*?)</li>'
        numberlist=re.compile(contentpat,re.S).findall(data1)
        numberlist=str(numberlist)
        number=re.compile(userpat,re.S).findall(numberlist)
        for element in number:
            out.append(element[1])
        return out
    except urllib.error.URLError as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
        sleep(10)
    except Exception as e:
        print("exception:"+str(e))
        sleep(1)
wb=openpyxl.Workbook()
sheet=wb.active
sheet.title="历年双色球中奖号码"
nlist=[]
for step in range(1,240):
    start=(step-1)*10
    end=step*10
    if end >len(linklist):
        end=len(linklist)
    else:
        end=step*10
    for link in linklist[start:end]:
        numberList=getnumber(link[0])
        time=link[0].split('/')[-1]
        info="%s\t%s" %(time.split('.')[0],"\t".join(numberList))
        infolist=info.split('\t')
        nlist.append(infolist)
        print("%s\t%s" %(link[0],datetime.datetime.now()))
        sleep(20)
    sleep(600)

for i in range(0,len(nlist)):
    for j in range(0,len(nlist[i])):
        sheet.cell(row=i+1,column=j+1,value=int(nlist[i][j]))
    print(i)
wb.save("D:\双色球.xlsx")
