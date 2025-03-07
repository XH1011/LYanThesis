from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
import pandas as pd
from collections import Counter

def calculate_purity(labels, pre_labels):
    # 获取所有唯一的簇标签
    unique_clusters = np.unique(pre_labels)
    total_samples = len(labels)
    matched_samples = 0

    # 遍历每个簇
    for cluster in unique_clusters:
        # 获取当前簇的样本索引
        cluster_indices = np.where(pre_labels == cluster)[0]
        # 找到当前簇中真实标签出现次数最多的类别
        true_labels_in_cluster = labels[cluster_indices]
        majority_label_count = np.max(np.bincount(true_labels_in_cluster))
        matched_samples += majority_label_count

    # 计算 Purity
    purity = matched_samples / total_samples
    return purity



np.set_printoptions(threshold=np.inf)
print('revision:',revision)

file_name=f'../[Step2]clustering/cwru/diffusion-encoded-data/en_test{revision}.pkl'
with open(file_name, "rb") as f:
     labels = np.squeeze(pickle.load(f)[1])
# labels = np.squeeze(np.concatenate((np.zeros((300, 1),dtype= int), np.full((1800, 1), -1,dtype=int))))
#iforest
df = pd.read_excel(f'../[Step3]prob_fusion/results/cwru/fusion_results{revision}.xlsx', sheet_name='prob')
prob_n_ifo = df.iloc[1:, 0].values # 如果 prob 是第一列
prob_ab_ifo = df.iloc[1:, 1] .values # 如果 ab_prob 是第二列
# accuracy_ifo = df['Accuracy'].iloc[0]
# print('异常识别：', accuracy_ifo)


#clustering
df = pd.read_excel(f'../[Step3]prob_fusion/results/cwru/fusion_results{revision}.xlsx', sheet_name='prob')
prob_C0_cls = df.iloc[1:, 2].values # 如果 prob 是第一列
prob_C1_cls = df.iloc[1:, 3].values
prob_C2_cls = df.iloc[1:, 4].values
prob_C3_cls = df.iloc[1:, 5].values
prob_C4_cls = df.iloc[1:, 6].values

for w_ifo in range(0,11):
    results = []
    w_ifo = w_ifo/10
    # print('w_ifo:',w_ifo)
    w_cls = (10-w_ifo*10)/10
    prob_n = w_ifo * prob_n_ifo + w_cls * prob_C0_cls
    prob_f = 1-prob_n
    prob_C0 = prob_n
    prob_C1 = np.zeros(len(labels))
    prob_C2 = np.zeros(len(labels))
    prob_C3 = np.zeros(len(labels))
    prob_C4 = np.zeros(len(labels))
    pre_labels = np.squeeze(np.zeros(labels.shape, dtype=int))
    for i in range(len(labels)):
        prob_C1[i]= (prob_C1_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
        prob_C2[i]= (prob_C2_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
        prob_C3[i]= (prob_C3_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
        prob_C4[i]= (prob_C4_cls[i]/(1-prob_C0_cls[i]))*prob_f[i]
        pre_labels[i] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}[np.argmax([prob_C0[i], prob_C1[i], prob_C2[i],prob_C3[i],prob_C4[i]])]
    n_error=0
    for i in range(len(labels)):
        if labels[i] != pre_labels[i]:
            n_error+=1
    accuracy = 1- (n_error/len(labels))
    # Calculate PUR
    cluster_label_counts = {}
    for true_label, pred_label in zip(labels, pre_labels):
        if pred_label not in cluster_label_counts:
            cluster_label_counts[pred_label] = Counter()
        cluster_label_counts[pred_label][true_label] += 1

    purity = 0.0
    for cluster, counts in cluster_label_counts.items():
        max_count = max(counts.values())
        purity += max_count / sum(counts.values())

    print(accuracy)

    results.append({'w_ifo': w_ifo,'w_cls':w_cls, 'accuracy': accuracy,'pre_labels':pre_labels})
    df_results = pd.DataFrame(results)
    df_results.to_excel('./results/cwru/fusion_results.xlsx', index=False)



