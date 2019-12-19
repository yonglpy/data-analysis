# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:47:05 2019

@author: lenovo
"""

import xlrd
import sys
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

workbook = xlrd.open_workbook(r'D:\lenovo\桌面1\RNA\RNA类相互作用网络.xlsx')
sheet_names = workbook.sheet_names()

lncRNA_sheet = workbook.sheet_by_name(sheet_names[0])
circRNA_sheet = workbook.sheet_by_name(sheet_names[1])
mRNA_sheet = workbook.sheet_by_name(sheet_names[2])

miRNA_input = []
lncRNA_input = []
circRNA_input = []
mRNA_input = []
if len(sys.argv) in [2,3,4,5]:
	for input_name in sys.argv:
		if input_name.startswith('miRNA'):
			miRNA_input = input_name.split(':')[1].split(';')
		if input_name.startswith('lncRNA'):
			lncRNA_input = input_name.split(':')[1].split(';')
		if input_name.startswith('circRNA'):
			circRNA_input = input_name.split(':')[1].split(';')
		if input_name.startswith('mRNA'):
			mRNA_input = input_name.split(':')[1].split(';')

lncRNA_miRNA = {}
circRNA_miRNA = {}
mRNA_miRNA = {}
miRNA2target = {}
lncRNA = []
circRNA = []
mRNA = []
miRNA = []
color = []
G = nx.Graph()

miRNA2target.setdefault('circRNA',{})
miRNA2target.setdefault('lncRNA',{})
miRNA2target.setdefault('mRNA',{})

for i in range(1,lncRNA_sheet.nrows):
	item5 = lncRNA_sheet.row_values(i)
	if item5[1] in lncRNA_miRNA:
		lncRNA_miRNA[item5[1]] = lncRNA_miRNA[item5[1]]+'\t'+item5[2]
	else:
		lncRNA_miRNA[item5[1]] = item5[2]
		lncRNA.append(item5[1])
	if item5[2]  not in miRNA:
		miRNA.append(item5[2])
	if item5[2] in miRNA2target['lncRNA']:
		miRNA2target['lncRNA'][item5[2]] = miRNA2target['lncRNA'][item5[2]] + '\t' + item5[1]
	else:
		miRNA2target['lncRNA'].setdefault(item5[2],item5[1])
for j in range(1,circRNA_sheet.nrows):
	infor = circRNA_sheet.row_values(j)
	if infor[0] in circRNA_miRNA:
		circRNA_miRNA[infor[0]] = circRNA_miRNA[infor[0]] +'\t'+infor[5]
	else:
		circRNA_miRNA[infor[0]] = infor[5]
		circRNA.append(infor[0])
	if infor[5] not in miRNA:
		miRNA.append(infor[5])
	if infor[5] in miRNA2target['circRNA']:
		miRNA2target['circRNA'][infor[5]] = miRNA2target['circRNA'][infor[5]] + '\t' + infor[0]
	else:
		miRNA2target['circRNA'].setdefault(infor[5],infor[0])
for n in range(1,mRNA_sheet.nrows):
	element = mRNA_sheet.row_values(n)
	if element[1] in mRNA_miRNA:
		mRNA_miRNA[element[1]] = mRNA_miRNA[element[1]] + '\t'+element[0]
	else:
		mRNA_miRNA[element[1]] = element[0]
		mRNA.append(element[1])
	if element[0] not in miRNA:
		miRNA.append(element[0])
	if element[0] in miRNA2target['mRNA']:
		miRNA2target['mRNA'][element[0]] = miRNA2target['mRNA'][element[0]] + '\t' + element[1]
	else:
		miRNA2target['mRNA'].setdefault(element[0],element[1])
all_element = miRNA+lncRNA+circRNA+mRNA
group = []
size = []
#print(miRNA2target['lncRNA']['hsa-mir-150'])
for m in range(len(miRNA)):
	group.append(1)
	size.append(5)
for i1 in range(len(lncRNA)):
	group.append(2)
	size.append(10)
for j1 in range(len(circRNA)):
	group.append(3)
	size.append(15)
for n5 in range(len(mRNA)):
	group.append(4)
	size.append(20)

Linknodes = pd.DataFrame({'name':all_element,'group':group,'size':size},columns=['name','group','size'])
Linknodes.to_csv('RNA_network_linknode.csv',index=False)
soure2target = {}

for p in range(len(all_element)):
	soure2target[all_element[p]] = p

source = []
target = []
value = []
item = ''
if miRNA_input:
	for item in miRNA_input:
		if item not in list(G.node):
			G.add_node(item)
			color.append('g')
		if item in miRNA2target['circRNA']:
			circRNA_miRNA_list = miRNA2target['circRNA'][item].split('\t')
			for circRNA_miRNA_list_item in circRNA_miRNA_list:
				if circRNA_miRNA_list_item not in list(G.node):
					G.add_node(circRNA_miRNA_list_item)
					color.append('b')
					G.add_edge(item,circRNA_miRNA_list_item)
					source.append(soure2target[item])
					target.append(soure2target[circRNA_miRNA_list_item])
					value.append(random.randrange(0,20))
				if circRNA_miRNA_list_item in list(G.node):
					if (item,circRNA_miRNA_list_item) not in list(G.edges()):
						G.add_edge(item,circRNA_miRNA_list_item)
						source.append(soure2target[item])
						target.append(soure2target[circRNA_miRNA_list_item])
						value.append(random.randrange(0,20))
		if item in miRNA2target['lncRNA']:
			lncRNA_miRNA_list = miRNA2target['lncRNA'][item].split('\t')
			for lncRNA_miRNA_list_item in lncRNA_miRNA_list:
				if lncRNA_miRNA_list_item not in list(G.node):
					G.add_node(lncRNA_miRNA_list_item)
					color.append('r')
					G.add_edge(item,lncRNA_miRNA_list_item)
					source.append(soure2target[item])
					target.append(soure2target[lncRNA_miRNA_list_item])
					value.append(random.randrange(0,20))
				if lncRNA_miRNA_list_item in list(G.node):
					if (item,lncRNA_miRNA_list_item) not in list(G.edges()):
						G.add_edge(item,lncRNA_miRNA_list_item)
						source.append(soure2target[item])
						target.append(soure2target[lncRNA_miRNA_list_item])
						value.append(random.randrange(0,20))
		if item in miRNA2target['mRNA']:
			mRNA_miRNA_list = miRNA2target['mRNA'][item].split('\t')
			for mRNA_miRNA_list_item in mRNA_miRNA_list:
				if mRNA_miRNA_list_item not in list(G.node):
					G.add_node(mRNA_miRNA_list_item)
					color.append('orange')
					G.add_edge(item,mRNA_miRNA_list_item)
					source.append(soure2target[item])
					target.append(soure2target[mRNA_miRNA_list_item])
					value.append(random.randrange(0,20))
				if mRNA_miRNA_list_item in list(G.node):
					if (item,mRNA_miRNA_list_item) not in list(G.edges()):
						G.add_node(item,mRNA_miRNA_list_item)
						source.append(soure2target[item])
						target.append(soure2target[mRNA_miRNA_list_item])
						value.append(random.randrange(0,20))
if lncRNA_input:
	for item1 in lncRNA_input:
		if item1 not in list(G.node):
			G.add_node(item1)
			color.append('r')
		for miRNA_name in lncRNA_miRNA[item1].split('\t'):
			if miRNA_name not in list(G.node):
				G.add_node(miRNA_name)
				color.append('g')
				G.add_edge(miRNA_name,item1)
				source.append(soure2target[miRNA_name])
				target.append(soure2target[item1])
				value.append(random.randrange(0,20))
			if miRNA_name in list(G.node):
				if (miRNA_name,item1) not in list(G.edges()):
					G.add_edge(miRNA_name,item1)
					source.append(soure2target[miRNA_name])
					target.append(soure2target[item1])
					value.append(random.randrange(0,20))
				if miRNA_name in miRNA2target['circRNA']:
					if miRNA_name not in list(G.node):
						G.add_node(miRNA_name)
						color.append('g')
				circRNA_miRNA_list1 = miRNA2target['circRNA'][miRNA_name].split('\t')
				for circRNA_miRNA_list1_item in circRNA_miRNA_list1:
					if circRNA_miRNA_list1_item not in list(G.node):
						G.add_node(circRNA_miRNA_list1_item)
						color.append('b')
						G.add_edge(miRNA_name,circRNA_miRNA_list1_item)
						source.append(soure2target[miRNA_name])
						target.append(soure2target[circRNA_miRNA_list1_item])
						value.append(random.randrange(0,20))
					if circRNA_miRNA_list1_item in list(G.node):
						if (miRNA_name,circRNA_miRNA_list1_item) not in list(G.edges()):
							G.add_edge(miRNA_name,circRNA_miRNA_list1_item)
							source.append(soure2target[miRNA_name])
							target.append(soure2target[circRNA_miRNA_list1_item])
							value.append(random.randrange(0,20))
			if miRNA_name in miRNA2target['mRNA']:
				if miRNA_name not in list(G.node):
					G.add_node(miRNA_name)
					color.append('g')
				mRNA_miRNA_list1 = miRNA2target['mRNA'][miRNA_name].split('\t')
				for mRNA_miRNA_list1_item in mRNA_miRNA_list1:
					if mRNA_miRNA_list1_item not in list(G.node):
						G.add_node(mRNA_miRNA_list1_item)
						color.append('orange')
						G.add_edge(miRNA_name,mRNA_miRNA_list1_item)
						source.append(soure2target[miRNA_name])
						target.append(soure2target[mRNA_miRNA_list1_item])
						value.append(random.randrange(0,20))
					if mRNA_miRNA_list1_item in list(G.node):
						if (miRNA_name,mRNA_miRNA_list1_item) not in list(G.edges()):
							G.add_edge(miRNA_name,mRNA_miRNA_list1_item)
							source.append(soure2target[miRNA_name])
							target.append(soure2target[mRNA_miRNA_list1_item])
							value.append(random.randrange(0,20))
if circRNA_input:
	for item2 in circRNA_input:
		if item2 not in list(G.node):
			G.add_node(item2)
			color.append('b')
		for miRNA_name1 in circRNA_miRNA[item2].split('\t'):
			if miRNA_name1 not in list(G.node):
				G.add_node(miRNA_name1)
				color.append('g')
				G.add_edge(miRNA_name1,item2)
				source.append(soure2target[miRNA_name1])
				target.append(soure2target[item2])
				value.append(random.randrange(0,20))
			if miRNA_name1 in list(G.node):
				if (miRNA_name1,item2) not in list(G.edges()):
					G.add_edge(miRNA_name1,item2)
					source.append(soure2target[miRNA_name1])
					target.append(soure2target[item2])
					value.append(random.randrange(0,20))
			if miRNA_name1 in miRNA2target['lncRNA']:
				if miRNA_name1 not in list(G.node):
					G.add_node(miRNA_name1)
					color.append('g')
				lncRNA_miRNA_list1 = miRNA2target['lncRNA'][miRNA_name1].split('\t')
				for lncRNA_miRNA_list1_item in lncRNA_miRNA_list1:
					if lncRNA_miRNA_list1_item not in list(G.node):
						G.add_node(lncRNA_miRNA_list1_item)
						color.append('r')
						G.add_edge(miRNA_name1,lncRNA_miRNA_list1_item)
						source.append(soure2target[miRNA_name1])
						target.append(soure2target[lncRNA_miRNA_list1_item])
						value.append(random.randrange(0,20))
					if lncRNA_miRNA_list1_item in list(G.node):
						if (miRNA_name1,lncRNA_miRNA_list1_item) not in list(G.edges()):
							G.add_edge(miRNA_name1,lncRNA_miRNA_list1_item)
							source.append(miRNA_name1)
							target.append(lncRNA_miRNA_list1_item)
							value.append(random.randrange(0,20))
			if miRNA_name1 in miRNA2target['mRNA']:
				if miRNA_name1 not in list(G.node):
					G.add_node(miRNA_name1)
					color.append('g')
				mRNA_miRNA_list2 = miRNA2target['mRNA'][miRNA_name1]
				for mRNA_miRNA_list2_item in mRNA_miRNA_list2:
					if mRNA_miRNA_list2_item not in list(G.node):
						G.add_node(mRNA_miRNA_list2_item)
						color.append('orange')
						G.add_edge(miRNA_name1,mRNA_miRNA_list2_item)
						source.append(soure2target[miRNA_name1])
						target.append(soure2target[mRNA_miRNA_list2_item])
						value.append(random.randrange(0,20))
					if mRNA_miRNA_list2_item in list(G.node):
						if (miRNA_name1,mRNA_miRNA_list2_item) not in list(G.edges()):
							G.add_edge(miRNA_name1,mRNA_miRNA_list2_item)
							source.append(soure2target[miRNA_name1])
							target.append(soure2target[mRNA_miRNA_list2_item])
							value.append(random.randrange(0,20))
if mRNA_input:
	for item3 in mRNA_input:
		if item3 not in list(G.node):
			G.add_node(item3)
			color.append('orange')
		for miRNA_name2 in mRNA_miRNA[item3].split('\t'):
			#print(miRNA_name2)
			if miRNA_name2 not in list(G.node):
				G.add_node(miRNA_name2)
				color.append('g')
				G.add_edge(item3,miRNA_name2)
				source.append(soure2target[item3])
				target.append(soure2target[miRNA_name2])
				value.append(random.randrange(0,20))
			if miRNA_name2 in list(G.node):
				if (item3,miRNA_name2) not in list(G.edges()):
					G.add_edge(item3,miRNA_name2)
					source.append(soure2target[item3])
					target.append(soure2target[miRNA_name2])
					value.append(random.randrange(0,20))
			if miRNA_name2 in miRNA2target['circRNA']:
				if miRNA_name2 not in list(G.node):
					G.add_node(miRNA_name2)
					color.append('g')
				circRNA_list = miRNA2target['circRNA'][miRNA_name2].split('\t')
				#print(circRNA_list)
				for circRNA_item in circRNA_list:
					if circRNA_item not in list(G.node):
						G.add_node(circRNA_item)
						color.append('b')
						G.add_edge(miRNA_name2,circRNA_item)
						source.append(miRNA_name2)
						target.append(circRNA_item)
						value.append(random.randrange(0,20))
					if circRNA_item in list(G.node):
						if (miRNA_name2,circRNA_item) not in list(G.edges()):
							G.add_edge(miRNA_name2,circRNA_item)
							source.append(miRNA_name2)
							target.append(circRNA_item)
							value.append(random.randrange(0,20))
			if miRNA_name2 in miRNA2target['lncRNA']:
				if miRNA_name2 not in list(G.node):
					G.add_node(miRNA_name2)
					color.append('g')
				lncRNA_list = miRNA2target['lncRNA'][miRNA_name2].split('\t')
				#print(lncRNA_list)
				for lncRNA_item in lncRNA_list:
					if lncRNA_item not in list(G.node):
						G.add_node(lncRNA_item)
						color.append('r')
						G.add_edge(miRNA_name2,lncRNA_item)
						source.append(miRNA_name2)
						target.append(lncRNA_item)
						value.append(random.randrange(0,20))
					if lncRNA_item in list(G.edges()):
						if (miRNA_name2,lncRNA_item) in list(G.edges()):
							G.add_edge(miRNA_name2,lncRNA_item)
							source.append(miRNA_name2)
							target.append(lncRNA_item)
							value.append(random.randrange(0,20))
links = pd.DataFrame({'source':source,'target':target,'value':value})
links.to_csv('RNA_network_links.csv',index=False)
pos = nx.spring_layout(G)
#print(pos)
nx.draw(G,pos,with_labels=True,node_color=color,node_size=200,width=0.1,aex=(-1.1,1.1,-1.1,1.1),font_size=8)
#nx.draw(G,pos,with_labels=True,node_size=200,width=0.1)
plt.axis([-1.2,1,-1.2,1.2])
plt.text(-1.15,0.9,'lncRNA',color='red',bbox=dict(boxstyle='round,pad=0.5', fc='red', lw=1 ,ec=	'k',alpha=0.5))
plt.text(-1.15,0.75,'circRNA',color='blue',bbox=dict(boxstyle='round,pad=0.5', fc='blue', lw=1,ec='k',alpha=0.5))
plt.text(-1.15,0.6,'mRNA',color='orange',bbox=dict(boxstyle='round,pad=0.5', fc='orange',lw=1,ec='k',alpha=0.5))
plt.text(-1.15,0.45,'miRNA',color='green',bbox=dict(boxstyle='round,pad=0.5', fc='green', ec='k',lw=1 ,alpha=0.5))
plt.show()