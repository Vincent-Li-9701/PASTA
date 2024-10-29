#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scene_synthesis.datasets.partnet import PartNetDataset
import os
import matplotlib.pyplot as plt


root_dir = "/afs/cs.stanford.edu/u/hansonlu/remote/partnethiergeo/chair_hier"
object_list = "val"
data_features = ['object', 'name']

def get_all_objects(root_dir, object_list, data_features):
    with open(os.path.join(root_dir, object_list + ".txt"), 'r') as f:
        object_names = [item.rstrip() for item in f.readlines()]
    return object_names



ten_parts = open(os.path.join(root_dir, "{}_10.txt".format(object_list)), "w")
less_than_10_parts = open(os.path.join(root_dir,"{}_less_than_10.txt".format(object_list)), "w")
more_than_10_parts = open(os.path.join(root_dir,"{}_more_than_10.txt".format(object_list)), "w")

for idx in get_all_objects(root_dir, object_list, data_features):
    obj = PartNetDataset.load_object(os.path.join(root_dir, idx +'.json'), \
                        load_geo=True)
    part_boxes, part_geos, part_angles, part_quats = obj.graph(leafs_only=True)

    if len(part_boxes) == 10:
        ten_parts.write(idx + "\n")
    if len(part_boxes) <= 10:
        less_than_10_parts.write(idx + "\n")
    else:
        more_than_10_parts.write(idx + "\n")

ten_parts.close()
less_than_10_parts.close()
more_than_10_parts.close()


