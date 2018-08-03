#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import fileinput
import sys
from collections import defaultdict
import load_save_pkl
import os

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

# per_org_candidates_list= load_save_pkl.load_pickle_file("outputs/per_org_candidates_list.pkl")
repo_path = 'D:/thesis_stuff/code/master_thesis/'
per_entity_list = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/per_entity_list.pkl"))
org_entity_list = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/org_entity_list.pkl"))
post_entity_list = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/post_entity_list.pkl"))
# print(per_entity_list)
per_entity_list=  [x.lower() for x in per_entity_list]
org_entity_list=  [x.lower() for x in org_entity_list]
post_entity_list=  [x.lower() for x in post_entity_list]
# print(per_entity_list)

load_save_pkl.save_as_pkl(per_entity_list, os.path.join(repo_path,"outputs/per_entity_list.pkl"))
load_save_pkl.save_as_pkl(org_entity_list, os.path.join(repo_path,"outputs/org_entity_list.pkl"))
load_save_pkl.save_as_pkl(post_entity_list, os.path.join(repo_path,"outputs/post_entity_list.pkl"))

exit(0)
freebase_data_path = '/home/ads/ads/rajaram_s/thesis_stuff/software/freebase-easy-latest/'

def collect_relationships():
    #######################
    # weak entity linking #
    #######################
    # load freebase entities to memory
    # select only a few entities, based on the relationships

    per_org_relationships = defaultdict(list)
    per_post_relationships = defaultdict(list)
    count = 0

    data = './facts_0_0.7.txt'
    print("the freebase file is %s" %data)
    numbered = re.compile('#[0-9]+$')
    for line in fileinput.input(data):
        if count % 5000 == 0:
            f_status = open("status.txt", "a")
            f_status.write(str(count)+'\t'+str(len(per_org_relationships))+str(len(per_post_relationships))+'\n')
            # for r in relationships.keys():
            #     for e in relationships[r]:
            #         f_status.write(r[0]+'\t'+e+'\t'+r[1]+'\n')
            f_status.close()

        e1, r, e2, point = line.split('\t')
        if e1.startswith('/') or e2.startswith('/'):
            continue
        if e1.startswith('m/') or e2.startswith('m/'):
            continue
        if re.search(numbered, e1) or re.search(numbered, e2):
            continue
        # lots of irrelevant stuff in contained_by
        if re.match(r'^[0-9]+$', e1) or re.match(r'^[0-9]+$', e2):
            continue
        if e1.startswith('DVD Region') or e2.startswith('DVD Region'):
            continue
        if e1.startswith('US Census'):
            continue
        if e1.startswith('Area code') or e2.startswith('Area code'):
            continue
        else:
            if "(" in e1:
                e1 = re.sub(r"\(.*\)", "", e1).strip()
            if "(" in e2:
                e2 = re.sub(r"\(.*\)", "", e2).strip()

            if e1 in per_entity_list and e2 in org_entity_list:
                per_org_relationships[(e1, e2)].append(r)
            if e1 in org_entity_list and e2 in per_entity_list:
                per_org_relationships[(e2, e1)].append(r)
            if e1 in per_entity_list and e2 in post_entity_list:
                per_post_relationships[(e1, e2)].append(r)
            if e1 in post_entity_list and e2 in per_entity_list:
                per_post_relationships[(e2, e1)].append(r)
        count += 1
    fileinput.close()

    print "Writing collected relationships to disk"
    f1_entities = open("per_org_relationships.txt", "w")
    for r in per_org_relationships.keys():
        for e in per_org_relationships[r]:
            f1_entities.write(r[0]+'\t'+e+'\t'+r[1]+'\n')
    f1_entities.close()
    f2_entities = open("per_post_relationships.txt", "w")
    for r in per_post_relationships.keys():
        for e in per_post_relationships[r]:
            f2_entities.write(r[0]+'\t'+e+'\t'+r[1]+'\n')
    f2_entities.close()
    return per_org_relationships, per_post_relationships


def main():
    per_org_relationships, per_post_relationships = collect_relationships()

    print len(per_org_relationships)
    print len(per_post_relationships)
    load_save_pkl.save_as_pkl(per_org_relationships, os.path.join(repo_path,"outputs/per_org_relationships.pkl"))
    load_save_pkl.save_as_pkl(per_post_relationships, os.path.join(repo_path,"outputs/per_post_relationships.pkl"))
if __name__ == "__main__":
    main()
