import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
from scipy.io import savemat
import random
import pickle

#prob
# df = pd.read_excel('./att/prob_cwru.xlsx', sheet_name='Sheet1')
# att = df.iloc[0:5, 0:5].values
df = pd.read_excel('../[Step3]prob_fusion/results/cwru/fusion_resultsR5.xlsx', sheet_name='prob')
att = df.iloc[1:6, 14:19].values
att = att.astype(float)

##allclasses_name
f = open('data/ours/classes_cwru.txt', 'r')
all_classes = f.readlines()
classes_name = []
for line in all_classes:
    idx, name = [i.strip() for i in line.split(' ')]
    classes_name.append(name)
allclasses_names = np.empty((len(classes_name), 1), dtype=np.object)
for i, class_name in enumerate(classes_name):
    allclasses_names[i, 0] = class_name

#testclasses ##第一次选取零样本学习的测试集类别时，随机选取，后续为了与其他对比方法统一，则注释掉这段代码
# selected_classes = random.sample(all_classes, k=2)
# test_class_name = []
# test_class_idx = []
# for line in selected_classes:
#     idx, name = [i.strip() for i in line.split(' ')]
#     print(f"编号: {idx}, 类别名称: {name}")
#     test_class_name.append(name)
#     test_class_idx.append(idx)
# with open('data/ours/testclasses_cwru.txt', 'w') as f:
#     for name in test_class_name:
#         f.write(f'{name}\n')
# with open('data/ours/testclasses_idx_cwru.txt', 'w') as f:
#     for idx in test_class_idx:
#         f.write(f'{idx}\n')

#test_unseen_loc
mat = loadmat('data/ours/res101_cwruR5.mat')
all_labels = np.squeeze(mat['labels'])
test_unseen_loc=[]
with open('data/ours/testclasses_idx_cwru.txt') as fp:
    test_class_idxs = fp.readlines()
    test_class_idxs = [label.strip() for label in test_class_idxs]
for test_idx in test_class_idxs:
    for idx, label in enumerate(all_labels, start=1):
        label = str(label)
        if test_idx == label:
            test_unseen_loc.append(idx)
test_unseen_loc = np.array(test_unseen_loc, dtype=int).reshape(-1, 1)

#trainval_loc
def paichu(part,all):
    for i in part:
        if i in all:
            all.remove(i)
    return all
all_idxs= np.unique(all_labels)
train_labels=[]
seen_loc=[]
test_class_idxs = [int(idx) for idx in test_class_idxs]
train_idxs = paichu(test_class_idxs,list(all_idxs))
for train_idx in train_idxs:
    for label, idx in enumerate(all_labels, start=1):
        if train_idx == idx:
            seen_loc.append(label)

traindata_len = int(len(seen_loc) * 0.85)
trainval_loc = random.sample(seen_loc, traindata_len)
testseen_loc = paichu(trainval_loc,seen_loc)
trainval_loc = np.array(trainval_loc)
testseen_loc = np.array(testseen_loc)
test_seen_loc = np.array(testseen_loc, dtype=int).reshape(-1, 1)
trainval_loc = np.array(trainval_loc, dtype=int).reshape(-1, 1)
savemat('data/ours/att_splits_cwruR5.mat', {'att': att, 'allclasses_names': allclasses_names, 'test_unseen_loc':test_unseen_loc, 'test_seen_loc':test_seen_loc, 'trainval_loc':trainval_loc})

##当换数据之后，维持train_loc\test_loc仍然不变
# mat1=loadmat('./data/att_splits_cwru.mat')
# mat2=loadmat('./att_chopper/unseen_3/att_splits_unseen_3.mat')
# test_seen_loc2=mat2['test_seen_loc']
# test_unseen_loc2=mat2['test_unseen_loc']
# trainval_loc2=mat2['trainval_loc']
# mat1['test_seen_loc']=test_seen_loc2
# mat1['test_unseen_loc']=test_unseen_loc2
# mat1['trainval_loc']=trainval_loc2
# savemat('./comparison_method/ori_softatt_splits_chp.mat', mat1)
