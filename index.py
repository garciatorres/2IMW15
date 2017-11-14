# Libraries ***********************************************************
# *********************************************************************
import os, re, sys, lucene

from subprocess import *
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.store import SimpleFSDirectory

import pandas as pd
import numpy as np
import csv as csv_api

# Init Lucene *********************************************************
# *********************************************************************
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

# Main variables ******************************************************
# *********************************************************************
rumor_folder = "/rumors"
index_folder = rumor_folder+"_index"

# Functions ***********************************************************
# *********************************************************************
def indexDirectory(dir):
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isfile(path):
            indexFile(dir, name)

def indexFile(dir, filename):
    
    path = os.path.join(dir, filename)
    
    print "  Cluster: ", filename

    if filename.endswith('.csv'):
        # reads the cluster file into a dataframe
        tweetV = pd.read_csv(open(path,'rU'), delimiter="\t", engine='python')
    else: pass
    
    #gets label of cluster
    label = np.array(tweetV["Cluster_Lab"].head(1))
    #gets score of document
    lentv = len(tweetV["T"]) 
    score = tweetV["T"].sum()
    score = (lentv+score)*100/(2*lentv)
    #gets keywords
    

    doc = Document()
    doc.add(Field("label", label[0], TextField.TYPE_STORED))
    doc.add(Field("score", str(score), StringField.TYPE_STORED))
#   doc.add(Field("section", section, StringField.TYPE_STORED))
    doc.add(Field("name", label[0].strip(), TextField.TYPE_STORED))
#   doc.add(Field("synopsis", synopsis.strip(), TextField.TYPE_STORED))
#   doc.add(Field("keywords", ' '.join((command, name, synopsis, description)),TextField.TYPE_NOT_STORED))
    doc.add(Field("filename", os.path.abspath(path), StringField.TYPE_STORED))
    print

    writer.addDocument(doc)

# Program *************************************************************
# *********************************************************************		

# create index directory
index_dir = os.getcwd() + index_folder
if not os.path.exists(index_dir):
    os.makedirs(index_dir)
# rumour directory (should be already created)
rumor_dir = os.getcwd() + rumor_folder
if not os.path.exists(rumor_dir):
    os.makedirs(rumor_dir)	
# get index storage
directory = SimpleFSDirectory(Paths.get(index_dir))
# get the analyzer
analyzer = StandardAnalyzer()
analyzer = LimitTokenCountAnalyzer(analyzer,10000)
config = IndexWriterConfig(analyzer)
# get the index writer
writer = IndexWriter(directory, config)
# get the rumour directory
rumor_path = rumor_dir.split(os.pathsep)

for dir in rumor_path:
    print 
    print "Crawling on folder...", dir
    print
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isdir(path):
            indexDirectory(path)
print
# finalize execution
writer.commit()
writer.close()
