from __future__ import print_function
import xml.etree.ElementTree as ET
import load_save_pkl
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import os
import time
import json
from nltk.parse.stanford import StanfordDependencyParser
from collections import defaultdict
import re
import glob
import itertools

entity_ann_path = 'D:/thesis_stuff/code/master_thesis/entity_annotations/'
ann_files_re = os.path.join(entity_ann_path, '*')
ann_files = sorted(glob.glob(ann_files_re))
repo_path = 'D:/thesis_stuff/code/master_thesis/'
per_list = []
org_list = []
post_list = []
def read_tsv(input_path):
    rows=[]
    for i, line in enumerate(open(input_path, 'rb')):
        row =line.split("\t")
        if len(row) < 6:
            continue
        # print(row)
        if row[3] == "O":
            continue
        if row[3] == "PERSON":
            per_list.append(row[4])
        if row[3] == "ORGANIZATION":
            org_list.append(row[4])
        if row[3] == "POST":
            post_list.append(row[4])
    return rows

print(ann_files)
for file in ann_files:
    print("Processing %s" %file)
    read_tsv(file)
    #read_tsv("D:/thesis_stuff/code/master_thesis/entity_annotations/930105-0101.annotated.tsv")


per_list = list(set(per_list))
org_list = list(set(org_list))
post_list = list(set(post_list))

load_save_pkl.save_as_pkl(per_list, os.path.join(repo_path,"outputs/per_entity_list.pkl"))
load_save_pkl.save_as_pkl(org_list, os.path.join(repo_path,"outputs/org_entity_list.pkl"))
load_save_pkl.save_as_pkl(post_list, os.path.join(repo_path,"outputs/post_entity_list.pkl"))
print(per_list)
print(org_list)
print(post_list)

print(len(per_list))
print(len(org_list))
print(len(post_list))
exit(0)
