import pickle
import load_save_pkl
freebase_file_location = "D:/thesis_stuff/code/master_thesis"
from collections import defaultdict
import os
repo_path = 'D:/thesis_stuff/code/master_thesis/'

# file1 = load_save_pkl.load_pickle_file("master_thesis_0_0.7/outputs/per_post_relationships.pkl")
# file2 = load_save_pkl.load_pickle_file("master_thesis_0.7_1/outputs/per_post_relationships.pkl")
# file3 = load_save_pkl.load_pickle_file("master_thesis_1_1.5/outputs/per_post_relationships.pkl")
# file4 = load_save_pkl.load_pickle_file("master_thesis_1.5_2/outputs/per_post_relationships.pkl")
# file5 = load_save_pkl.load_pickle_file("master_thesis_2_2.5/outputs/per_post_relationships.pkl")
# file6 = load_save_pkl.load_pickle_file("master_thesis_2.5_3/outputs/per_post_relationships.pkl")
# file7 = load_save_pkl.load_pickle_file("master_thesis_3_3.6/outputs/per_post_relationships.pkl")
#
# file_list = [file1,file2,file3,file4,file5,file6,file7]
# freebase_per_post = defaultdict(list)
# for file in file_list:
#     print(file)
#     print(len(file))
#     for k, v in file.iteritems():
#
#         freebase_per_post[k].extend(v)
#
# print freebase_per_post
# print(len(freebase_per_post))

# load_save_pkl.save_as_pkl(freebase_per_post, "outputs/freebase_per_post.pkl")

freebase_per_org = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/freebase_per_org.pkl"))
freebase_per_post = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/freebase_per_post.pkl"))

# for k,v in freebase_per_org.iteritems():
#     print k,v

# print freebase_per_post
freebase_per_org_dict = defaultdict(list)
for k,v in freebase_per_org.iteritems():
    per = k[0]
    org = k[1]
    if v[0] not in ['Place of publication', 'Color', 'Place of birth', 'Place of death', 'Release of', 'Recording', 'Contained by', 'Neighborhood of', 'Parent Company', 'Place', 'is-a', 'Program']:
        freebase_per_org_dict[per].append(org)

print(freebase_per_org_dict)
print(len(freebase_per_org_dict))

freebase_per_post_dict = defaultdict(list)
for k,v in freebase_per_post.iteritems():
    per = k[0]
    post = k[1]
    if v[0] not in ['Category']:
        freebase_per_post_dict[per].append(post)

print(freebase_per_post_dict)
print(len(freebase_per_post_dict))

load_save_pkl.save_as_pkl(freebase_per_org_dict, os.path.join(repo_path,"outputs/consolidated_freebase_per_org_dict.pkl"))
load_save_pkl.save_as_pkl(freebase_per_post_dict, os.path.join(repo_path,"outputs/consolidated_freebase_per_post_dict.pkl"))


per_alias_dict = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/per_alias_dict.pkl"))
org_alias_dict = load_save_pkl.load_pickle_file(os.path.join(repo_path,"outputs/org_alias_dict.pkl"))
print(len(per_alias_dict))
print(per_alias_dict)
print(org_alias_dict)
print(len(org_alias_dict))

print("New code")
# create freebase_per_org_list_tuple
freebase_per_org_list_tuple = []
for freebase_per,freebase_org in freebase_per_org_dict.iteritems():
    # print(freebase_per)
    # print(freebase_per.split()[-1])
    # print(freebase_org)
    per_names_list = set()
    org_names_list = set()
    per_names_list.add(freebase_per)
    per_names_list.add(freebase_per.split()[-1])
    org_names_list.update(freebase_org)

    for k,v in per_alias_dict.iteritems():
        v = [x.lower() for x in v]
        v.append(k.lower())
        if any(i in per_names_list for i in v):
            per_names_list.update(v)
            for k1, v1 in org_alias_dict.iteritems():
                v1 = [x1.lower() for x1 in v1]
                v1.append(k1.lower())
                if any(i in org_names_list for i in v1):
                    org_names_list.update(v1)
    # print(per_names_list)
    # print(org_names_list)
    # exit(0)
    # per_names_list = list(set(per_names_list))
    # org_names_list = list(set(org_names_list))
    freebase_per_org_list_tuple.append([list(per_names_list), list(org_names_list)])
print(freebase_per_org_list_tuple)
print(len(freebase_per_org_list_tuple))
    # exit(0)


freebase_per_post_list_tuple = []
for freebase_per,freebase_post in freebase_per_post_dict.iteritems():
    per_names_list = set()
    per_names_list.add(freebase_per)
    per_names_list.add(freebase_per.split()[-1])

    for k,v in per_alias_dict.iteritems():
        v = [x.lower() for x in v]
        v.append(k.lower())
        if any(i in per_names_list for i in v):
            per_names_list.update(v)
    freebase_per_post_list_tuple.append([list(per_names_list), list(freebase_post)])
print(freebase_per_post_list_tuple)
print(len(freebase_per_post_list_tuple))


load_save_pkl.save_as_pkl(freebase_per_org_list_tuple, os.path.join(repo_path,"outputs/consolidated_freebase_per_org_list_tuple.pkl"))
load_save_pkl.save_as_pkl(freebase_per_post_list_tuple, os.path.join(repo_path,"outputs/consolidated_freebase_per_post_list_tuple.pkl"))