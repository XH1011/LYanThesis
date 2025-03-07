import pickle

import hdf5storage
import scipy.io as sio
import os, traceback
import pandas as pd

from AEnet_13_sc import ConvAE
from AEutils_sc import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VI+SIBLE_DEVICES"] = "0"

d = 16
# d = 15
alpha = 6
# alpha = 5
# ro = 0.02
ro = 0.01

np.set_printoptions(threshold=np.inf)

#diffusion_encoded(cwru)
data = hdf5storage.loadmat('D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/diffusion-encoded-data/en_test.mat')
Img = data['data']
Label = data['gnd']
has_nan = np.isnan(Img).any()
print(has_nan)
Img = np.reshape(Img, (Img.shape[0], 128))
order = list(range(len(Img)))

n_input = [128, 1]
kernel_size = [3]
n_hidden = [15]
# n_hidden = [20, 10, 5]
# kernel_size = [5, 5, 5]
batch_size = 5 * 300
model_path = 'D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/model/model.ckpt'
restore_path = 'D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/model/model.ckpt'
logs_path = 'D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/logs/logs'

num_class = 5 # how many class we sample
num_sa = 300

batch_size_test = num_sa * num_class

iter_ft = 0
ft_times = 30
display_step = 300

fine_step = -1

reg1 = 1e-4
# reg02 = 100
# reg03 = 1e-5

mm = 0
mreg = [0, 0, 0, 0]
mlr2 = 0
startfrom = [0, 0, 0]

# def test_face(Img, Label, CAE, num_class):
def test_face(Img, Label, CAE, num_class,learning_rate):
    acc_ = []
    for i in range(0, 1):
        face_10_subjs = np.array(Img[num_sa * i:num_sa * (i + num_class), :])
        face_10_subjs = face_10_subjs.astype(float)
        label_10_subjs = np.array(Label[num_sa * i:num_sa * (i + num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs)

        CAE.initlization()
        CAE.save_model()
        # CAE.restore()
        COLD = None
        lastr = 1.0
        losslist = []
        for iter_ft in range(ft_times):
            # print('iter_ft',iter_ft)
            CAE.restore()
            # cost, C, dd, dt = CAE.partial_fit(face_10_subjs, mode='fine')
            cost, C, dd, dt = CAE.partial_fit(face_10_subjs, learning_rate, mode='fine')  #
            CAE.save_model()
            losslist.append(cost[-1])
            if iter_ft % display_step == 0 and iter_ft > 10:
                print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
                print(cost)
                for posti in range(2):
                    display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)

            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(C - COLD, ord='fro')
                r = normcd / normc
                # print(epoch,r)
                if r < 1.0e-6 and lastr < 1.0e-6:
                    print("early stop")
                    print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
                    print(cost)
                    for posti in range(2):
                        display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)
                    break
                lastr = r
            COLD = C

        # print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
        # print(cost)

        # drawC(C)
        # print(C)
        for posti in range(1):
            acc, L, y_pre,index_cluster, index_original,y_pre_o,NMI,PUR = display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)
            acc_.append(acc)
            probs = probas(L, y_pre, order, index_cluster, index_original)
            acc_probs = cal_acc(label_10_subjs, probs)
            # data = {
            #     'reg2': [reg2] * probs.shape[0],
            #     'reg3': [reg3] * probs.shape[0],
            #     'reg4': [reg4] * probs.shape[0],
            #     'acc': [acc_probs] * probs.shape[0],
            #     'NMI': [NMI] * probs.shape[0],
            #     'PUR': [PUR] * probs.shape[0],
            #     'y_pre': [y_pre_o]* probs.shape[0]
            # }
            #
            # # 向数据字典中添加每个类别的概率列
            # for i in range(probs.shape[1]):
            #     data[f'prob_C{i}'] = probs[:, i]
            # df = pd.DataFrame(data)
            # file_path = "D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/results/"
            # file_name = f"results.xlsx"
            # file_full_path = file_path + file_name
            # df.to_excel(file_full_path, index=False)

        acc_.append(acc)

    # for sd in [12,16]:
    # 	for sa in [6,8]:
    # 		for sr in [0.02,0.03,0.04]:
    # 			print(sd, sa, sr)
    # 			display(C, coil20_all_subjs, sd, sa, sr, num_class, label_all_subjs)

    acc_ = np.array(acc_)
    # print(acc_)
    mm = np.max(acc_)

    # print("%d subjects:" % num_class)
    # print("Max: %.4f%%" % ((1 - mm) * 100))
    # print(acc_)
    lossnp = np.asarray(losslist)
    # np.savetxt("loss-l2.csv", lossnp, delimiter=',')
    return (1 - mm),L,y_pre,y_pre_o,

all_subjects = [5]
# for reg2 in [0.1,1,10]:
#     for reg3 in [0.1,1,10]:
#         for reg4 in [0.1,1,10]:
for reg2 in [0.1]:
    for reg3 in [0.1]:
        for reg4 in [0.1]:
            for lr2 in [1e-4]:
                # for ft_times in range(1,31,1):
                try:
                    print("reg:", reg2, reg3, reg4, lr2)
                    # print("reg:", reg2, reg3, reg4)
                    # print('ft_times',ft_times)
                    avg = []
                    med = []
                    iter_loop = 0
                    while iter_loop < len(all_subjects):
                        num_class = all_subjects[iter_loop]
                        batch_size = num_class * num_sa

                        tf.reset_default_graph()
                        CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2,
                                     re_constant3=reg3, re_constant4=reg4, ds=num_class, \
                                     kernel_size=kernel_size, batch_size=batch_size, model_path=model_path,
                                     restore_path=restore_path, logs_path=logs_path)
                        avg_i,L,y_pre,y_pre_o = test_face(Img, Label, CAE, num_class, lr2)
                        # df = pd.DataFrame(y_pre_o, columns=['y_pre(Chopper)'])
                        # df.to_excel('D:/Desktop/paper2/images/experiment/1.xlsx', sheet_name='1',index=False)
                        # avg_i = test_face(Img, Label, CAE, num_class)
                        avg.append(avg_i)
                        iter_loop = iter_loop + 1
                        visualizem(Img, Label, CAE, 'D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/results/clusterin')
                    iter_loop = 0

                    if 1 - avg[0] > mm:
                        drawC(L, 'D:/Desktop/毕业论文/codes/[Step2]clustering/cwru/results/L-L2.png')
                        mreg = [reg2, reg3, reg4, lr2]
                        # mreg = [reg2, reg3, reg4]
                        mm = 1 - avg[0]
                    # print("max:", mreg, mm)

                except:
                    print("error in ", reg2, reg3, lr2)
                    # print("error in ", reg2, reg3)
                    traceback.print_exc()
                finally:
                    try:
                        CAE.sess.close()
                    except:
                        ''
