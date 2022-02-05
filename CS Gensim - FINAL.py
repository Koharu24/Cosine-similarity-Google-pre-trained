#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This code run only with Gensim 3.8.3 so we remove 4.0 and then we install gensim 3.8.3

#!pip uninstall gensim  

#!pip install gensim==3.8.3


# In[1]:


#import

from gensim import models
import gensim
import numpy as np
from statistics import mean
import nltk
from nltk.corpus import stopwords
import re


# In[2]:


#path we are going to use 

chosenidioms= "C:/Users/Giulia Santoro/Desktop/cl/esame/chosen_idioms.txt"
fulldataset= "C:/Users/Giulia Santoro/Desktop/cl/esame/dataset completo.txt"
lphrases= "C:/Users/Giulia Santoro/Desktop/cl/esame/literal_phrases.txt"
nlphrases= "C:/Users/Giulia Santoro/Desktop/cl/esame/non_literal_phrases.txt"


# In[3]:


#we restrict the google vectors passing our dataset to this function

def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    
#import google vectors 

model = models.KeyedVectors.load_word2vec_format(
    'C:/Users/Giulia Santoro/Desktop/cl/esame/GoogleNews-vectors-negative300.bin', binary=True)


# In[5]:


# we create my own personal vectors starting from my cleaned and tokenized dataset

corpus = open(fulldataset, encoding="utf8") 

astring = corpus.read()

tokens = nltk.word_tokenize(astring)

words = [word.lower() for word in tokens if word.isalpha()] #eliminate punctuation

#we create our personal stopword set to add to the one nltk provide

ourset={"the","neither", "shall", "no", "yes", "that", "a","in","may","would","can","could","rather","got"} 
stop_words = set(stopwords.words('english'))|ourset

vocab = list(set([word for word in words if not word in stop_words])) #we create the vocab to pass to the function restrict

#print(vocab)

restricted = restrict_w2v(model, vocab)


# In[7]:


#this function create a list of list of words (every sentence is a list of tokenized words and is stored in a bigger list)

def listcreator(filename, stop_words={}):#stopword is empty to give the opportunity to decide to pass our stopword set or not
    if len(stop_words) == 0:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    #print(stop_words)
    
    with open(filename, "r") as handlefile:
        flines=handlefile.readlines()
        group= []
        filtered= []
   
        for line in flines:
            group.append(nltk.word_tokenize(line))

        for line in group:
            line = [each_string.lower() for each_string in line]
            ## the double loop is needed to make to remove steps, punct and stopwords
            for word in line:
                for word in line:
                    if not word.isalpha():
                        line.remove(word)
                    if word in stop_words:
                        line.remove(word)
            #we discard the square brackets
            if line != []:
                filtered.append(line)
       
        return filtered


# In[8]:


#we apply the function to our split dataset (nonliteral e literal)

literal_list = listcreator(lphrases, stop_words)
nonliteral_list = listcreator(nlphrases, stop_words)

#print("Literal tokenized list of list:\n")
#print(literal_list)
#print("\nNon-literal tokenized list of list:\n")
#print(nonliteral_list)


# In[9]:


#we calculate the cosine similarity between the words in every sentence. We could have done it putting all the value in a matrix
#with all the words and then retrieve the values we needed but it was a waste of computational power, since we want to do the 
#cosine similarity of the words in a given line and not of all the words. 

#we also could have calculated the cosine similarity between all the words in a sentence and put it in a matrix but we did not want
#to calculate the value repetitions typical of a matrix (ex: of the line I like fish you calculate I and like but also like e I).
#it was no problem however as the cosine similarity should have be averaged after this step

#we opted for scanning the list of list with indexes, and then calculate the cosine similarity incrementally 
#and avoiding repetitions

def cossimll(mylist):

    cosinell= []
    cosinel= []
    for sentence in mylist:
        myarray= np.array(sentence)
        #print("\n***** New sentence to analize: ******\n" + str(myarray))
        x=1
        for i in range(len(sentence)):
            #print("  <------->")
            #print("Word from compare:->  " + myarray[i])
            sourcetocompare = myarray[i]
            j=x
            for k in range(len(myarray)-j):
                #print("Word  to  compare:->  " + sentence[j])
                desttocompare = sentence[j]
            
                print("source:-> " + sourcetocompare + "     dest:-> " + desttocompare +"\n")
                try:
                    cosine_similarity = model.similarity(myarray[i], sentence[j])
                    #print("cosine similarity between "+ sourcetocompare + " and " + desttocompare + " is:")
                    #print(cosine_similarity)
                    #print("\n")
                    cosinel.append(cosine_similarity)
                except:
                    pass
                j=j+1
            x=x+1
        cosinell.append(cosinel)
        cosinel=[]    
        
    return cosinell      

literalcossim= cossimll(literal_list)
nliteralcossim= cossimll(nonliteral_list)

#print("\nCosine similarity for each word of each literal sentence:\n")
#print(literalcossim)
#print("\nCosine similarity for each word of each non-literal sentence:\n")
#print(nliteralcossim)


# In[10]:


#we do the average of every sentence in each list

def listaverage(list_):

    listresults= []
    for l in list_:
        listresults.append(mean(l))
    
    return listresults   

literalaverage=listaverage(literalcossim)
nliteralaverage= listaverage(nliteralcossim)


#print("Average of the words' cosine similarity of each literal sentence:\n")
#print(literalaverage)
#print("\nAverage of the words' cosine similarity of each non-literal sentence:\n")
#print(nliteralaverage)


# In[11]:


#We check if all the sentences are in the same number. We need this to do the next step: handle the list of averages knowing how
#many sentences cointain a specific idiom

def countidioms(idiomf, file):
    count= 0
    listcount=[]
    with open(idiomf, "r") as idiomf_:
        strings= idiomf_.readlines()
        idiomlist= [idiom.strip() for idiom in strings]

        with open(file, "r") as file_:
            strings2=file_.readlines()
            for idiom in idiomlist:
                for string in strings2:
                    if re.search(idiom, string, re.IGNORECASE):
                        count=count+1
                        
                listcount.append(count)
                count=0            
    return listcount

lcount= countidioms(chosenidioms, lphrases)
nlcount= countidioms(chosenidioms, nlphrases)

#print("count for literal sentences for each idiom:\n")
#print(lcount)
#print("\ncount for non-literal sentences for each idiom:\n")
#print(nlcount)


# In[12]:


#we create a function that does the average of the average results of each sentence with a specific idiom

def averageofav (count, mylist):
    meanlist=[]
    j=0
    k=0
    for i in count:
        k=k+i
        meanlist.append(mean(mylist[j:k]))
        j=j+i
    return meanlist

lvaluelist= averageofav(lcount, literalaverage)
nlvaluelist= averageofav(nlcount, nliteralaverage)

#print("Average of literal sentences for each idiom (20):\n")
#print(lvaluelist)
#print("\nAverage of non-literal sentences for each idiom (20):\n")
#print(nlvaluelist)


# In[13]:


def createdict(idioms, valuelist):
    
    with open(idioms, "r") as filesource:
        strings= filesource.readlines()
    
        idiomlist= [idiom.strip() for idiom in strings]
        #print(idiomlist) 
    
        dict_= dict(zip(idiomlist,valuelist))
        
        return dict_
    
ldict= createdict(chosenidioms, lvaluelist)
nldict= createdict(chosenidioms, nlvaluelist)

#print("Dictionary of idioms as keys and average of literal sentences for each idiom as value:\n")
#print(ldict)
#print("\nDictionary of idioms as keys and average of non-literal sentences for each idiom as value:\n")
#print(nldict)


# In[18]:


#as we start from the idea that literal sentences should be more congruent than the non-literal counterparts aka nearer to 1 
#we compare the values of the two dicts and we put the result in two counters

with open(chosenidioms, "r") as filesource:
    strings= filesource.readlines()
    
    idiomlist= [idiom.strip() for idiom in strings]
    #print(idiomlist) 
    
    neilist=[]
    expected= 0
    notexpected= 0 
    for idiom in idiomlist:
        try:
            if ldict[idiom] > nldict[idiom]:
                expected+= 1
            
            else:
                notexpected+=1
                nei= idiom
                neilist.append(nei)
    
        except: 
            pass
        
print("Expected results: literal sentences are more congruent than non-literal:\n")
print(expected)
print("\nNot expected results: non-literal sentences are more congruent than literal:\n")
print(notexpected)
print("\nThese are the idioms which non-literal sentences were more congruent than their literal counterpart:\n")
print(neilist)

