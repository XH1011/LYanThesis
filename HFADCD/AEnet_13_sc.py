import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,re_constant4=1.0,
                 batch_size=200, reg=None,ds = None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs',rawImg=None):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        usereg = 2
        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0]* n_input[1]])
        self.learning_rate = tf.placeholder(tf.float32, [])
        t_bs = tf.shape(self.x)[0]
        weights = self._initialize_weights()

        if denoise == False:#不去噪
            x_input = self.x
            # latent, shape = self.encoder(x_input, weights)
        else:#去噪：添加随机噪声到原始数据x上
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)

        # hid_dim =latent.shape[1].value * latent.shape[2].value #潜变量的维度 高*宽*通道数
        # z = tf.reshape(latent, [t_bs, hid_dim]) #将latent reshape 批量大小*隐变量维度
        # classifier  module
        if ds is not None: #ds:num class类别数
            #pslb0 = tf.layers.dense(z, 4*ds, kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.sigmoid,name='ss_d0')

            pslb = tf.layers.dense(x_input,ds,kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.softmax,name = 'ss_d')
            #创建全连接层  将输入z经过全连接层进行变换，得到一个具有ds维度的输出向量pslb
            cluster_assignment = tf.argmax(pslb, -1)
            eq = tf.to_float(tf.equal(cluster_assignment,tf.transpose(cluster_assignment)))#包含了每个样本所属类别的索引的向量
            # 计算两个样本是否属于同一个类别,不属于同一个类别，对应位置的元素为 1.0,否则为 0.0.  将cluster_assignment与转置后的cluster_assignment对比
        # ze = z
        Coef = weights['Coef']
        # z_ce = tf.matmul(Coef, ze) #矩阵相乘  z_ce 是一个包含了每个样本在每个聚类中加权和的张量
        # z_ce 是通过自表示系数和基向量进行加权组合得到的编码器输出的特征表示
        # if ds is not None:
        #     z_c = tf.layers.dense(z_ce, hid_dim,activation=tf.nn.relu, name='demb')

        # z_c = z_ce
        self.Coef = Coef

        # latent_c = tf.reshape(z_c, tf.shape(latent))
        # self.z = ze

        # self.x_r = self.decoder(latent_c, weights, shape) #重构输入 有Coef
        # self.x_r2 = self.decoder(latent, weights, shape) #重构输入 无Coef
        # l_2 reconstruction loss
        # self.reconst_cost = tf.reduce_sum(tf.square(tf.subtract(self.x_r, self.x))) #重构误差 有Coef
        # self.reconst_cost_pre = tf.reduce_sum(tf.square(tf.subtract(self.x_r2, self.x))) #无Coef的重构误差
        # tf.summary.scalar("recons_loss", self.reconst_cost)

        if usereg == 2:#正则化损失
            self.reg_losses = tf.reduce_sum(tf.square(self.Coef))+tf.trace(tf.square(self.Coef))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))+tf.trace(tf.abs(self.Coef))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        # self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.square(tf.subtract(z_ce, ze)))
        #自表示的损失 z_ce与ze之间的差异 重构特征 z_ce 与原始特征 ze 的差值

        # tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        x_flattten = tf.reshape(x_input, [t_bs, -1]) #原始输入
        # x_flattten2 = tf.reshape(self.x_r, [t_bs, -1]) #重构输入 有coef
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(XZ, x_flattten)))
        #自表示的损失 原始输入数据与原始输入数据与Coef相乘后两者间的差异

        normL = True
        #graph(C)
        absC = tf.abs(Coef) #对Coef元素取绝对值
        C = (absC + tf.transpose(
            absC)) * 0.5  # * (tf.ones([Coef.shape[0].value,Coef.shape[0].value])-tf.eye(Coef.shape[0].value))
        C = C + tf.eye(Coef.shape[0].value) #加上了单位矩阵，确保矩阵 C 对角线上的元素为 1

        self.cc=C- tf.eye(Coef.shape[0].value) #矩阵 C 减去单位矩阵
        # DD = tf.diag(tf.sqrt(1.0/tf.reduce_sum(C, axis=1)))
        # C = tf.matmul(DD,C)
        # C = tf.matmul(C,DD)
        # D = tf.eye(Coef.shape[0].value)
        # L = D - C
        if normL == True:#根据需求选择是否对拉普拉斯矩阵进行归一化操作，True进行归一化，False不进行
            D = tf.diag(tf.sqrt((1.0 / tf.reduce_sum(C, axis=1))))
            I = tf.eye(D.shape[0].value)
            L = I - tf.matmul(tf.matmul(D, C), D)
            D = I
        else:
            D = tf.diag(tf.reduce_sum(C, axis=1))
            L = D - C
        # self.reg_losses += 1.0*tf.reduce_sum(tf.square(tf.reduce_sum(Coef,axis=1)-tf.ones_like(tf.reduce_sum(Coef,axis=1))))
        # XLX = tf.matmul(tf.matmul(tf.transpose(x_flattten), L), x_flattten)
        # XLX2 = tf.matmul(tf.matmul(tf.transpose(x_flattten), L), x_flattten2)
        # YLY = tf.matmul(tf.matmul(tf.transpose(z), L), z)
        # XX = x_flattten - x_flattten2 #原始输入与有Coef的重构输入之间的差
        # XXDXX = tf.matmul(tf.matmul(tf.transpose(XX),D),XX)
        #由两部分组成的损失函数，第一部分表示x_flattten(原始输入)和x_flattten2(重构输入Coef)之间的差异，第二部分表示了x_flattten和x_flattten2
        # 在经过Laplacian矩阵变换后的差异  平衡重构和图结构约束的作用
        # self.tracelossx = tf.reduce_sum(tf.square(XX)) + 2.0 * tf.trace(XLX2)  # /self.batch_size
        # self.d = tf.reduce_sum(C, axis=1)

        self.d = cluster_assignment #聚类分配的结果
        # self.l = tf.trace(XLX2) #矩阵XLX2的迹
        regass = tf.to_float(tf.reduce_sum(pslb,axis=0)) #pslb按列进行求和，表示正则化项中的聚类权重

        onesl=np.ones(batch_size) #大小为batch_size的全1数组
        zerosl=np.zeros(batch_size) #大小为batch_size的全0数组
        #thershold
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        #它根据pslb沿着轴1的最大值是否大于0.8的条件，在onesl和zerosl之间进行选择。如果最大值大于0.8，则选择onesl对应的元素，否则选择zerosl对应的元素
        cluster_assignment1=tf.one_hot(cluster_assignment,ds) #将其转换为one-hot编码表示 ds指定了one-hot的维度
        self.w_weight=weight_label
        self.labelloss=tf.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)
        #softmax 交叉熵损失的值，表示模型的预测结果和真实标签之间的差异

        self.graphloss = tf.reduce_sum(tf.nn.relu((1-eq) * C)+tf.nn.relu(eq * (0.001-C)))+ tf.reduce_sum(tf.square(regass))
        #图损失的表达式



        #self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        #self.loss2 = (self.reconst_cost+self.tracelossx + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  + re_constant4 * self.selfexpress_losses2)
        self.loss3 = ( re_constant2 * self.selfexpress_losses2  + re_constant3 * self.labelloss+re_constant4 * self.graphloss)
         #重构损失 跟踪损失 自表达损失 标签损失 图约束损失
            # self.reconst_cost + reg_constant1 * self.reg_losses + re_constant3 * self.selfexpress_losses2 + re_constant2 * self.selfexpress_losses
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)
        self.optimizer = self.optimizer2#tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer #AdamOptimizer
        # self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
        #     self.reconst_cost_pre)  #无Coef的重构误差
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if v.name.startswith("Coef")])
        # self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef") or v.name.startswith("ss"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef")or v.name.startswith("ss"))]
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden) #隐藏层数量
        all_weights['Coef'] = tf.Variable(
            # 1 * tf.eye(self.batch_size, dtype=tf.float32), name='Coef')
            1.0e-5 * (tf.ones([self.batch_size, self.batch_size], dtype=tf.float32)), name='Coef')
            # 创建一个大小为batch_size x batch_size 的矩阵，并将其每个元素初始化为 1.0e-5
        # all_weights['enc_w0'] = tf.get_variable("enc_w0",
        #                                         shape=[self.kernel_size[0],  1, self.n_hidden[0]],
        #                                         initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        # # 权重，name = 'enc_w0' 高度 宽度 当前层输入通道数 输出通道数(dec_w0、dec_b0呢？)
        # all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))  #偏置项 , name = 'enc_b0'
        #
        # iter_i = 1  #encoder
        # while iter_i < n_layers:
        #     enc_name_wi = 'enc_w' + str(iter_i)
        #     all_weights[enc_name_wi] = tf.get_variable(enc_name_wi,
        #                                                shape=[self.kernel_size[iter_i],
        #                                                       self.n_hidden[iter_i - 1], \
        #                                                       self.n_hidden[iter_i]],
        #                                                initializer=layers.xavier_initializer_conv2d(),
        #                                                regularizer=self.reg)
        #     enc_name_bi = 'enc_b' + str(iter_i)
        #     all_weights[enc_name_bi] = tf.Variable(
        #         tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))  # , name = enc_name_bi
        #     iter_i = iter_i + 1
        #
        # iter_i = 1  #decoder
        # while iter_i < n_layers:
        #     dec_name_wi = 'dec_w' + str(iter_i - 1)
        #     all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i],
        #                                                                    self.n_hidden[n_layers - iter_i - 1],
        #                                                                    self.n_hidden[n_layers - iter_i]],
        #                                                initializer=layers.xavier_initializer_conv2d(),
        #                                                regularizer=self.reg)
        #     dec_name_bi = 'dec_b' + str(iter_i - 1)
        #     all_weights[dec_name_bi] = tf.Variable(
        #         tf.zeros([self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))  # , name = dec_name_bi
        #     iter_i = iter_i + 1
        #
        # dec_name_wi = 'dec_w' + str(iter_i - 1)
        # all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], 1,
        #                                                                self.n_hidden[0]],
        #                                            initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        # dec_name_bi = 'dec_b' + str(iter_i - 1)
        # all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype=tf.float32))  # , name = dec_name_bi

        return all_weights

    # # Building the encoder
    # def encoder(self, x, weights):
    #     shapes = []
    #     shapes.append(x.get_shape().as_list())
    #     layeri = tf.nn.bias_add(tf.nn.conv1d(x, weights['enc_w0'], stride=2, padding='SAME'),
    #                             weights['enc_b0']) #x 输入张量 卷积核张量 步长 填充方式
    #     layeri = tf.nn.relu(layeri) #对layeri进行激活，负值设置为0，正值不变
    #     shapes.append(layeri.get_shape().as_list())
    #
    #     n_layers = len(self.n_hidden)
    #     iter_i = 1
    #     while iter_i < n_layers:
    #         layeri = tf.nn.bias_add(
    #             tf.nn.conv1d(layeri, weights['enc_w' + str(iter_i)], stride=2, padding='SAME'),
    #             weights['enc_b' + str(iter_i)])
    #         layeri = tf.nn.relu(layeri)
    #         shapes.append(layeri.get_shape().as_list()) #shapes列表存储了每一层的输出形状
    #         iter_i = iter_i + 1
    #
    #     layer3 = layeri #layer3存储编码器最后一层的结果
    #     return layer3, shapes
    #
    # # Building the decoder
    # def decoder(self, z, weights, shapes):
    #     n_layers = len(self.n_hidden)
    #     layer3 = z #z = tf.reshape(latent, [t_bs, hid_dim])
    #     iter_i = 0
    #     while iter_i < n_layers:
    #         # if iter_i == n_layers-1:
    #         #    strides_i = [1,2,2,1]
    #         # else:
    #         #    strides_i = [1,1,1,1]
    #         shape_de = shapes[n_layers - iter_i - 1]
    #         layer3 = tf.add(tf.nn.conv1d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack(
    #             [tf.shape(self.x)[0], shape_de[1], shape_de[2]]), strides=2, padding='SAME'),
    #                         weights['dec_b' + str(iter_i)])
    #         layer3 = tf.nn.relu(layer3)
    #         iter_i = iter_i + 1
    #     return layer3 #layer3存储解码器最后一层的结果

    def partial_fit(self, X, lr, mode=0):  #
        cost0, summary, _, Coef, d, dt = self.sess.run((self.selfexpress_losses2, self.merged_summary_op,
                                                        self.optimizer, self.Coef, self.w_weight, self.d),
                                                       feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0], Coef, d, dt



    # def partial_pre(self, X, lr, mode=0):  #
    #     cost0, _, = self.sess.run((self.reconst_cost_pre,self.optimizer_pre),
    #     feed_dict={self.x: X, self.learning_rate: lr})  #
    #     self.iter = self.iter + 1
    #     return [cost0]

    def initlization(self):#初始化
        tf.reset_default_graph()
        self.sess.run(self.init)

    # def reconstruct(self, X): #将输入数据X提供给模型，模型根据输入数据进行重构，并返回重构后的结果(有Coef)
    #     return self.sess.run(self.x_r, feed_dict={self.x: X})

    # def transform(self, X): #将输入数据X提供给模型，模型根据输入数据进行编码转换，并返回编码后的结果
    #     return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self): #保存模型的参数和变量
        save_path = self.saver.save(self.sess, self.model_path)
        # print("model saved in file: %s" % save_path)

    def restore(self): #从指定的文件中恢复模型的参数和变量
        self.saver.restore(self.sess, self.restore_path)
        # print("model restored")
    # def restore(self):
    #     import os
    #     if os.path.isfile(self.restore_path + '.index'):
    #         self.saver.restore(self.sess, self.restore_path)
    #         print("Model restored")
    #     else:
    #         print("Model file not found at:", self.restore_path)
