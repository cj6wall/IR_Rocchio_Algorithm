import glob
import pandas as pd
import numpy as np
import os
import math
import operator
from sklearn.metrics.pairwise import cosine_similarity
from progressbar import *

alpha = 0.3
beta = 0.9
TOPK = 2

TF_arrD = np.load("TF_arrD.npy")
TF_arrQ = np.load("TF_arrQ.npy")
tfidfD = np.load("tfidfD.npy")
tfidfQ = np.load("tfidfQ.npy")
IDF_arr = np.load("IDF_arr.npy")
vsm_table = np.load("vsmTable.npy")

list1 = glob.glob(r"Document/*")
print("Total Document = ",len(list1))

widgets = ['Loading Document: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets = widgets)
for x in progress(range(len(list1))):
	f = open(list1[x], 'r')

list2 = glob.glob(r"Query/*")
print("Total Query = ",len(list2))

widgets = ['Loading Query: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets = widgets)
for y in progress(range(len(list2))):
	f = open(list2[y], 'r')

D = {}
E = {}
Topk_doc = []
TopK_tfidf = np.zeros((15884,1),float)
NewtfidfQ = np.zeros((15884,len(list2)),float)
# print(tfidfD)

widgets = ['Doing Rocchio Algorithm: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets = widgets)
for i in progress(range(len(list2))):
	Topk_doc = []
	for j in range(len(list1)):
		D.update({list1[j]:vsm_table[i,j]})
		E.update({list1[j]:j})
	sorted_D = sorted(D.items(),key=operator.itemgetter(1),reverse = True)
	for k in range(TOPK):
		pick = sorted_D[k][0]
		Topk_doc.append(E.get(pick))
		# print(Topk_doc)
	for x in range(TOPK):	
		for y in range(15884):
			TopK_tfidf[y,0] += tfidfD[y,Topk_doc[x]]
	TopK_tfidf /= x
	TopK_tfidf *= beta
	tfidfQ *= alpha
	for m in range(15884):
		NewtfidfQ[m,i] = TopK_tfidf[m,0] + tfidfQ[m,i]

# print(NewtfidfQ)

#-----------------------------------------------------------

#D_tfidf,Q_tfidf
PointTable = np.zeros((len(list2),len(list1)),float)
# print(tfidfD.shape)
# print(NewtfidfQ.shape)

widgets = ['Doing cosine_similarity: ',Percentage(), ' ', Bar('#'),' ', Timer()]
progress = ProgressBar(widgets = widgets)
for x in progress(range(len(list2))): #Q有800個
	for y in range(len(list1)): #D有2265個
		PointTable[x,y] = cosine_similarity(tfidfD[:,y].reshape(1, -1),NewtfidfQ[:,x].reshape(1, -1))

#-----------------------------------------------------------

print("write ranking!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


result = open("Ranking.txt", 'w')
result.write("Query,RetrievedDocuments\n")

for i in range(len(list2)):
	D = {}
	result.write(list2[i].replace("Query/",""))
	result.write(",")
	for j in range(len(list1)):
		list1[j] = list1[j].replace("Document/","")
		D.update({list1[j]:PointTable[i,j]})
	D = sorted(D.items(),key=operator.itemgetter(1),reverse = True)
	for k in range(100):
		result.write(D[k][0] + " ")
	result.write("\n")

