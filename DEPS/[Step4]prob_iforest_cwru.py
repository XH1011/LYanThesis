from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
import pandas as pd

# for seeds in range (1,200):
#     print('seeds',seeds)

np.random.seed(96)
dir='../[Step1]diffusion-encoded/DCAE_cwru/results/en_'

# 读取正常数据
path_train_0 = dir+'x0_train.pkl'
with open(path_train_0,'rb') as f0:
        X_train_0 = pickle.load(f0)
        # print(X_train_0)

# # 标准化
des = X_train_0.std(axis=0)
media = X_train_0.mean(axis=0)
X_train_0 = (X_train_0 - media) / des

# ac = []
clf = IsolationForest(n_estimators=100,
                 max_samples="auto",
                 contamination="legacy",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=None,
                 behaviour='old',
                 random_state=None,
                 verbose=0)
clf.fit(X_train_0)

# 测试阶段
faults=['x0','x1', 'x2','x3','x4']
accum_percent = []
X_test_all=[]
for fault in faults:
    path_test = dir + fault + '_test.pkl'
    with open(path_test, 'rb') as f:
        X_test = pickle.load(f)
    X_test = (X_test - media) / des
    X_test_all.append(X_test)
X_test_all = np.concatenate(X_test_all, axis=0)
score = clf.decision_function(X_test_all)
min_score = np.min(score)
max_score = np.max(score)
prob = (score - min_score) / (max_score - min_score)
ab_prob = 1 - prob

labels = np.squeeze(np.concatenate((np.zeros((300, 1), dtype=int), np.full((1200, 1), -1, dtype=int))))
pre_labels = np.where(prob > ab_prob, 0, -1)

n_error = 0
for i in range (len(labels)):
    if labels[i] != pre_labels[i]:
        n_error += 1
accuracy =1- (n_error/len(labels))
print('accuracy',accuracy)
TP = np.sum((labels == 0) & (pre_labels == 0))
FP = np.sum((labels == -1) & (pre_labels == 0))
FN = np.sum((labels == 0) & (pre_labels == -1))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = 2 * (precision * recall) / (precision + recall)

data = {'Normal_prob': prob, 'Abnormal_prob': ab_prob}
df = pd.DataFrame(data)
accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})
F1_df = pd.DataFrame({'F1_score': [F1_score]})
recall_df = pd.DataFrame({'recall': [recall]})
precision_df = pd.DataFrame({'precision': [precision]})
labels_df = pd.DataFrame({'labels': labels})
pre_labels_df = pd.DataFrame({'pre_labels': pre_labels})
df = pd.concat([df, accuracy_df, F1_df,recall_df,precision_df, labels_df, pre_labels_df], axis=1)
writer = pd.ExcelWriter('./results/cwru/Prob_iforest.xlsx' )
df.to_excel(writer, 'sheet', float_format='%.4f')
writer.save()


