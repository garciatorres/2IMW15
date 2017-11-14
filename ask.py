#!/usr/bin/env python

import sys, os, lucene

from time import sleep
from string import Template
from datetime import datetime
from getopt import getopt, GetoptError

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory

# Main variables ******************************************************
# *********************************************************************
rumor_folder = "/rumors"
index_folder = rumor_folder+"_index"
question = "ENDTER: What is the rumour?:"

# Ask function ********************************************************
# *********************************************************************
def ask(searcher, analyzer):
    while True:
        print
        command = raw_input(question)
        print ""
        if command == '':
            return

        print
        print "Searching for:", command, "..."
        sleep(3) # Time in seconds.

        parser = QueryParser("label", analyzer)
        parser.setDefaultOperator(QueryParser.Operator.OR)
        #query = parser.parse(' '.join(args))        
        query = parser.parse(command)
        start = datetime.now()
        scoreDocs = searcher.search(query, 50).scoreDocs
        duration = datetime.now() - start
#       if stats:
        print >>sys.stderr, "Found %d RUMOURS(s) (in %s) that matched query '%s':" %(len(scoreDocs), duration, query)
        print

        k = 1
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            field = doc.getFields()
            for field in doc.getFields():
                if field.name() == 'label':
                    print str(k),'.'
                    print field.stringValue()
                    k = k+1
                if field.name() == 'score':
                    if float(field.stringValue()) > 50: result = 'TRUE'
                    else: result = 'FALSE'
                    print 'Truth Finder Score: ', field.stringValue(), '% (',result,')'
                    print

# Program *************************************************************
# *********************************************************************
if __name__ == '__main__':

    lucene.initVM(vmargs=['-Djava.awt.headless=true']) # Init Lucene

    def usage():
        print sys.argv[0], "[--format=<format string>] [--index=<index dir>] [--stats] <query...>"

    try:
        options, args = getopt(sys.argv[1:], '', ['format=', 'index=', 'stats'])
    except GetoptError:
        usage()
        sys.exit(2)

    format = "#name"	
    indexDir = os.getcwd() + index_folder
    stats = False
    for o, a in options:
        if o == "--format":
            format = a
        elif o == "--index":
            indexDir = a
        elif o == "--stats":
            stats = True


    class CustomTemplate(Template):
        delimiter = '#'
    template = CustomTemplate(format)

    fsDir = SimpleFSDirectory(Paths.get(indexDir))
    searcher = IndexSearcher(DirectoryReader.open(fsDir))
    analyzer = StandardAnalyzer()
    ask(searcher, analyzer)
    del searcher
