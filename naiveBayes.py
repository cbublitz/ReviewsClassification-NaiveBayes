"""
Naive Bayes Classifer for movie reviews text classification (positive or negative comment)
Machine learning - 2016/01 - UFRGS
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import math
import datetime
from random import shuffle

vocabulary = []
vocabularyPOS = []
vocabularyNEG = []
textAllPOS = []
textAllNEG = []
num_total_docs = 48000
num_total_class = 24000
pPOS = 0.0
pNEG = 0.0
docs = [[0 for x in range(2)] for x in range(num_total_docs)]
folderPos = "/home/carlos/Desktop/TrabalhoML/IMDB/pos/"
folderNeg = "/home/carlos/Desktop/TrabalhoML/IMDB/neg/"
ndocsPOS = 0
ndocsNEG = 0
probPOS = {}
probNEG = {}



def readFile(file):
	arquivo = open(file, 'r')
	return arquivo.read()

def createVocabulary():

	#Class POS
	for file in os.listdir(folderPos):
		text = readFile(folderPos + file)
		add2Vocabulary(text)

	#Class NEG
	for file in os.listdir(folderNeg):
		text = readFile(folderNeg + file)
		add2Vocabulary(text)
		
		
def add2Vocabulary(text):

	text = text.lower();
	text = removeCaracters(text);
	words = text.split(" ")

	for word in words:
		word = word.replace(" ", "")
		#Se a string não estiver vazia e a palavra ainda não estiver no vocabulary
		if word and len(word) > 1 and (word not in vocabulary):
			 vocabulary.append(word)


def removeCaracters(text):

	#Remover pontuacoes
	text = text.replace("."," ")
	text = text.replace(",","")
	text = text.replace(":","")
	text = text.replace(";","")
	text = text.replace("?","")
	text = text.replace("!","")
	text = text.replace(")","")
	text = text.replace("(","")
	text = text.replace("*","")

	#Remover tags html

	text = text.replace("<br"," ")
	text = text.replace("/><br"," ")
	text = text.replace("/>"," ")
	text = text.replace("<","")
	text = text.replace(">","")
	text = text.replace("/"," ")
	text = text.replace(r'\r','')

	#Remover artigos e pronomes
	text = text.replace(" to "," ")
	text = text.replace(" a "," ")
	text = text.replace(" as "," ")
	text = text.replace(" the "," ")
	text = text.replace(" of "," ")
	text = text.replace(" on "," ")
	text = text.replace(" in "," ")
	text = text.replace(" an "," ")
	text = text.replace(" its "," ")
	text = text.replace(" for "," ")
	text = text.replace(" i "," ")
	text = text.replace(" and "," ")
	text = text.replace(" this "," ")
	text = text.replace(" that "," ")
	text = text.replace(" these "," ")
	text = text.replace(" it "," ")
	text = text.replace(" he "," ")
	text = text.replace(" she "," ")
	text = text.replace(" his "," ")

	return text

def printVocabulary(data):
	for word in data:
		print word


def printProbability(data):
	for word in data:
		print word + " - " + str(data[word])


def add2VocabularyPOS(text):

	text = text.lower();
	text = removeCaracters(text);
	words = text.split(" ")

	for word in words:
		word = word.replace(" ", "")
		#Se a string não estiver vazia e a palavra ainda não estiver no vocabulary
		if word and len(word) > 1:
			textAllPOS.append(word)
			if word not in vocabularyPOS:
				vocabularyPOS.append(word)


def add2VocabularyNEG(text):

	text = text.lower();
	text = removeCaracters(text);
	words = text.split(" ")

	for word in words:
		word = word.replace(" ", "")
		#Se a string não estiver vazia e a palavra ainda não estiver no vocabulary
		if word and len(word) > 1:
			textAllNEG.append(word)
			if word not in vocabularyNEG:
				vocabularyNEG.append(word)


def learnNaiveBayes(trainingData):

	nPOS = 0
	nNEG = 0
	ndocsPOS = 0
	ndocsNEG = 0

	del vocabularyPOS[:]
	del vocabularyNEG[:]
	del textAllPOS[:]
	del textAllNEG[:]

	#Criar vocabulario da classe POS - textj
	i = 0
	while i <= len(trainingData)-1:

		#Verifica se é da classe POS (1) ou NEG (0)
		if trainingData[i][1] == 1:
			text = readFile(folderPos + str(trainingData[i][0]) + ".txt")
			add2VocabularyPOS(text)
			ndocsPOS = ndocsPOS + 1
		else:
			text = readFile(folderNeg + str(trainingData[i][0]) + ".txt")
			add2VocabularyNEG(text)
			ndocsNEG = ndocsNEG + 1

		i = i + 1

	#Probabilidades a priori POS e NEG
	global pPOS
	global pNEG
	pPOS = ndocsPOS/float(len(trainingData))
	pNEG = ndocsNEG/float(len(trainingData))

	#Numero de palavras distintas das classes POS e NEG
	nPOS = len(vocabularyPOS)
	nNEG = len(vocabularyNEG)

	#Tamanho vocabulario
	nVocabulary = len(vocabulary)

	probPOS.clear()
	probNEG.clear()
	
	#Calcula probabilidades palavras POS
	for wk in vocabularyPOS:
		if wk in vocabulary:
			nk = textAllPOS.count(wk)
			probPOS[wk] = (nk+1)/float(nPOS+nVocabulary)
	
	#Calcula probabilidades palavras NEG
	for wk in vocabularyNEG:
		if wk in vocabulary:
			nk = textAllNEG.count(wk)
			probNEG[wk] = (nk+1)/float(nNEG+nVocabulary)
			

def classifyNaiveBayes(testingData):

	nVP = 0.0
	nVN = 0.0
	nFP = 0.0
	nFN = 0.0
	vPOS = 0.0
	vNEG = 0.0
	i = 0
	while i <= len(testingData)-1:

		vPOS = 0.0
		vNEG = 0.0
		#Verifica se é da classe POS (1) ou NEG (0) para ler da pasta correta
		if testingData[i][1] == 1:
		
			text = readFile(folderPos + str(testingData[i][0]) + ".txt")
			#Calcula probabilidade classe POS
			text = removeCaracters(text);
			words = text.split(" ")

		else:

			text = readFile(folderNeg + str(testingData[i][0]) + ".txt")
			#Calcula probabilidade classe NEG
			text = removeCaracters(text);
			words = text.split(" ")


		for word in words:
			word = word.replace(" ", "")
			#Se a string não estiver vazia e se a palavra estiver no vocabulary POS
			if word and (word in vocabularyPOS):
				vPOS = vPOS + (math.log(pPOS) + math.log(probPOS[word]))

			#Se a string não estiver vazia e se a palavra estiver no vocabulary NEG
			if word and (word in vocabularyNEG):
				vNEG = vNEG + (math.log(pNEG) + math.log(probNEG[word]))


		if (vPOS < vNEG) and (testingData[i][1] == 1):
			nVP = nVP + 1

		if (vPOS < vNEG) and (testingData[i][1] == 0):
			nFP = nFP + 1

		if (vNEG < vPOS) and (testingData[i][1] == 0):
			nVN = nVN + 1

		if (vNEG < vPOS) and (testingData[i][1] == 1):
			nFN = nFN + 1

		i = i + 1

	arq.write("\n")
	arq.write("1 - Verdadeiro Positivo (VP): " + str(nVP) + "\n")
	arq.write("2 - Verdadeiro Negativo (VN): " + str(nVN) + "\n")
	arq.write("3 - Falso Positivo (FP): " + str(nFP) + "\n")
	arq.write("4 - Falso Negativo (FN): " + str(nFN) + "\n")
	arq.write("5 - Acuracia " + str((nVP+nVN)/len(testingData)) + "\n")
	arq.write("6 - Precisao " + str(nVP/(nVP+nFP)) + "\n")
	arq.write("7 - Recall " + str(nVP/(nVP+nFN)) + "\n")
	arq.write("8 - Medida-F " + str( (2*(nVP/(nVP+nFP))*(nVP/(nVP+nFN)))/((nVP/(nVP+nFP))+(nVP/(nVP+nFN)))) + "\n")

def getProbability(word, vetor):

	z = 0 
	prob = 0
	end = 0
	while z <= (len(vetor)-1) and (end == 0):
		if vetor[z][0] == word:
			prob = vetor[z][1]
			end = 1
		z = z + 1
	
	return prob

def createArrayDocs():
	
	# Primeira metade preenche com os arquivos Pos e depois com os NEG (0-24000 = 1 | 24001-48000 = 0)
	# POS = 1 NEG = 0
	# docs[num_doc][classe_doc]
	num = 0
	i = 0
	while i < (num_total_docs/2):
		docs[i][0] = (num+6)
		if (i%2 == 0):
			docs[i][1] = 1
		else:
			docs[i][1] = 0
		i = i + 1
		num = num + 1

	num=0
	while i <= num_total_docs-1:
		docs[i][0] = (num+6)
		if (i%2 == 0):
			docs[i][1] = 0
		else:
			docs[i][1] = 1
		i = i + 1
		num = num + 1


def cross_validation():

	#Aplicar o 10-fold
	num_folds = 10
	subset_size = len(docs)/num_folds
	for i in range(num_folds):

		#Conjunto de docs de teste
	    testing_this_round = docs[i*subset_size:][:subset_size]

	    #Conjunto de docs de treinamento
	    training_this_round = docs[:i*subset_size] + docs[(i+1)*subset_size:]

	    #Treinamento Naive Bayes
	    learnNaiveBayes(training_this_round)

	    #Classificacao
	    classifyNaiveBayes(testing_this_round)

	    arq.write("FOLD " + str(i) + "\n")
		
	    print "FOLD " + str(i)

#*************************
#* MAIN
#*************************

#Arquivo de log
arq = open("log.txt", "w")

arq.write("INICIO: " + str(datetime.datetime.now()) + "\n")
arq.write("\n")
print "INICIO: " + str(datetime.datetime.now())

createArrayDocs()
createVocabulary()

arq.write("CROSS VALIDATION: " + str(datetime.datetime.now()) + "\n")
arq.write("\n")
print "INICIO CROSS VALIDATION: " + str(datetime.datetime.now())

cross_validation()

arq.write("\n")
arq.write("FIM: " + str(datetime.datetime.now()) + "\n")
print "FIM: " + str(datetime.datetime.now())

arq.close()
