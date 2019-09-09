import numpy as np
import tensorflow as tf
from math import ceil


class Batch(object):
    def __init__(self, x_s, y_s, d_s, x_t_tune, y_t_tune, d_t_tune, x_ts, batch_size):
        self.x_s = x_s
        self.y_s = y_s
        self.d_s = d_s
        self.x_t_tune = x_t_tune
        self.y_t_tune = y_t_tune
        self.d_t_tune = d_t_tune
        self.x_ts = x_ts
        self.batch_size = batch_size
        # self.batch_num = (len(x_s) + len(x_t_tune)) // batch_size  # 向上取整

    # def generate(self, shuffle=False):
    #     """
    #     各个域的标注数据完全打散
    #     """
    #     d_s = np.tile(np.array([0., 1]), (len(self.x_s), 1))  # 源域的域标签
    #     d_t = np.tile(np.array([1., 0]), (len(self.x_t_tune), 1))  # 目标域标注样本的域标签
    #     x = np.vstack([self.x_s, self.x_t_tune])
    #     y = np.vstack([self.y_s, self.y_t_tune])
    #     d = np.vstack([d_s, d_t])
    #
    #     if shuffle:
    #         shuffled_data = np.random.permutation(list(zip(x, y, d)))
    #     else:
    #         shuffled_data = list(zip(x, y, d))
    #     sample_nums = len(x)
    #     for start in range(self.batch_num):
    #         yield shuffled_data[start * self.batch_size: min((start + 1) * self.batch_size, sample_nums)]

    # def generate(self, domain='source', shuffle=False):
    #
    #     if domain == 'source':
    #         x = self.x_s
    #         y = self.y_s
    #         d = np.tile(np.array([0., 1]), (len(self.x_s), 1))  # 源域的域标签
    #     else:
    #         x = self.x_t_tune
    #         y = self.y_t_tune
    #         d = np.tile(np.array([1., 0]), (len(self.x_t_tune), 1))  # 目标域标注样本的域标签
    #
    #     if shuffle:
    #         shuffled_data = np.random.permutation(list(zip(x, y, d)))
    #     else:
    #         shuffled_data = list(zip(x, y, d))
    #
    #     sample_nums = len(x)
    #     batch_num = int(sample_nums / self.batch_size)
    #     for start in range(batch_num):
    #         yield shuffled_data[start * self.batch_size: min((start + 1) * self.batch_size, sample_nums)]

    def generate(self, shuffle=False):
        x = np.vstack([self.x_s, self.x_t_tune])
        y = np.vstack([self.y_s, self.y_t_tune])
        d = np.vstack([self.d_s, self.d_t_tune])

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(x, y, d)))
        else:
            shuffled_data = list(zip(x, y, d))
        sample_nums = len(x)
        batch_num = int(sample_nums / self.batch_size)
        for start in range(batch_num):
            yield shuffled_data[start * self.batch_size: min((start + 1) * self.batch_size, sample_nums)]


class AdaptiveModel(object):
    def __init__(self, flags):
        self.FLAGS = flags
        self.gamma_r = 1.0
        self.gamma_c = 0.5
        self.gamma_d = 0.1
        self.gamma_diff = 1.0  # 正交损失太小了啊！

    def build_model(self):
        # tf graph input
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])
        self.D = tf.placeholder(dtype=tf.float32, shape=[None, len(self.FLAGS.target_domains)+1])
        self.test_X = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.test_Y = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])

        # fd network
        self.shared_encode_mlp = MLP('shared_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_s],
                                     [tf.nn.sigmoid])
        self.private_encode_mlp = MLP('private_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_p],
                                      [tf.nn.sigmoid])
        self.shared_decode_mlp = MLP('shared_decode_mlp',
                                     [self.FLAGS.n_hidden_s, (self.FLAGS.n_hidden_s + self.FLAGS.n_input) // 2,
                                      self.FLAGS.n_input],
                                     [tf.nn.tanh, tf.nn.relu])
        self.domain_clf_mlp = MLP('domain_output_mlp', [self.FLAGS.n_hidden_s, len(self.FLAGS.target_domains)+1],
                                  [identity])

        emb_s = self.shared_encode_mlp.apply(self.X)  # 编码共享表示
        emb_p = self.private_encode_mlp.apply(self.X)  # 编码私有表示，暂时多个域共享私有编码器
        self.total_theta = (self.shared_encode_mlp.parameters +
                            self.private_encode_mlp.parameters +
                            self.shared_decode_mlp.parameters +
                            self.domain_clf_mlp.parameters
                            )
        l2_norm = get_l2_norm(self.total_theta)

        # optimizing the parameters of the decoder F
        # 也可以分别解码，然后相加，计算重建损失
        decoding = self.shared_decode_mlp.apply(emb_s + emb_p)  # 共享表示解码器
        self.theta_f = self.shared_decode_mlp.parameters
        self.R_loss = self.gamma_r * tf.reduce_mean(tf.square(decoding - self.X)) + l2_norm  # 重建损失只用来优化解码器
        self.R_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.R_loss,
                                                                                     var_list=self.theta_f)

        # optimizing the parameters of the domain classifier D
        pred_s = self.domain_clf_mlp.apply(emb_s)  # 共享表示领域判别
        pred_p = self.domain_clf_mlp.apply(emb_p)  # 私有表示领域判别，暂时对所有私有表示用一个领域判别器来解码
        loss_s = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_s))
        loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_p))
        self.D_loss = self.gamma_d * (loss_s + loss_p) + l2_norm
        self.theta_d = self.domain_clf_mlp.parameters
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.D_loss,
                                                                                     var_list=self.theta_d)

        # optimizing the parameters of the label classifier C

        def calculate_proto(emb_s, emb_p, Y):
            """
            计算共享编码和私有编码的原型表示，需要已知样本标签
            """
            keep_pos = tf.squeeze(tf.matmul(Y, tf.constant([[1], [0.]])))
            keep_neg = tf.squeeze(tf.matmul(Y, tf.constant([[0.], [1]])))
            pos_idx = tf.reshape(tf.where(tf.equal(keep_pos, tf.constant(1.))), (-1,))
            neg_idx = tf.reshape(tf.where(tf.equal(keep_neg, tf.constant(1.))), (-1,))
            emb_pos_s = tf.gather(emb_s, pos_idx, axis=0)
            emb_neg_s = tf.gather(emb_s, neg_idx, axis=0)
            emb_pos_p = tf.gather(emb_p, pos_idx, axis=0)
            emb_neg_p = tf.gather(emb_p, neg_idx, axis=0)
            proto_pos_s = tf.reshape(tf.reduce_mean(emb_pos_s, axis=0), (1, -1))
            proto_neg_s = tf.reshape(tf.reduce_mean(emb_neg_s, axis=0), (1, -1))
            proto_pos_p = tf.reshape(tf.reduce_mean(emb_pos_p, axis=0), (1, -1))
            proto_neg_p = tf.reshape(tf.reduce_mean(emb_neg_p, axis=0), (1, -1))
            # 返回正类样本共享原型，负类样本共享原型，正类样本私有原型，负类样本私有原型
            return proto_pos_s, proto_neg_s, proto_pos_p, proto_neg_p

        proto_pos_s, proto_neg_s, proto_pos_p, proto_neg_p = calculate_proto(emb_s, emb_p, self.Y)
        # 暂时源域和目标域的损失计算方式相同
        dists = euclidean_distance(emb_s, tf.concat([proto_pos_s, proto_neg_s], axis=0)) #+ \
            #euclidean_distance(emb_p, tf.concat([proto_pos_p, proto_neg_p], axis=0))
        log_p_y = tf.nn.log_softmax(-dists, axis=1)
        self.C_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.Y, log_p_y), axis=-1)) + l2_norm
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(log_p_y, axis=-1), tf.argmax(self.Y, axis=-1)), tf.float32))
        emb_test_s = self.shared_encode_mlp.apply(self.test_X)
        # emb_test_p = self.private_encode_mlp.apply(self.test_X)
        dists_test = euclidean_distance(emb_test_s, tf.concat([proto_pos_s, proto_neg_s], axis=0)) #+ \
            #euclidean_distance(emb_test_p, tf.concat([proto_pos_p, proto_neg_p], axis=0))
        self.prob = tf.nn.softmax(-dists_test, axis=1)
        self.acc_test = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.prob, axis=-1), tf.argmax(self.test_Y, axis=-1)), tf.float32))

        """
        keep_pos = tf.squeeze(tf.matmul(self.Y, tf.constant([[1], [0.]])))
        keep_neg = tf.squeeze(tf.matmul(self.Y, tf.constant([[0.], [1]])))
        pos_idx = tf.reshape(tf.where(tf.equal(keep_pos, tf.constant(1.))), (-1,))
        neg_idx = tf.reshape(tf.where(tf.equal(keep_neg, tf.constant(1.))), (-1,))
        emb_pos_s = tf.gather(encoding_s, pos_idx, axis=0)  # 正类样本嵌入表示
        emb_neg_s = tf.gather(encoding_s, neg_idx, axis=0)  # 负类样本嵌入表示
        proto_pos = tf.reshape(tf.reduce_mean(emb_pos_s, axis=0), (1, -1))  # 正类原型
        proto_neg = tf.reshape(tf.reduce_mean(emb_neg_s, axis=0), (1, -1))  # 负类原型
        dists = euclidean_distance(encoding_s, tf.concat([proto_pos, proto_neg], axis=0))  # 50*2
        log_p_y = tf.nn.log_softmax(-dists, axis=1)
        self.C_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.Y, log_p_y), axis=-1)) + l2_norm
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(log_p_y, axis=-1), tf.argmax(self.Y, axis=-1)), tf.float32))
        emb_test = self.shared_encode_mlp.apply(self.test_X)
        dists_t = euclidean_distance(emb_test, tf.concat([proto_pos, proto_neg], axis=0))
        self.probs = tf.nn.softmax(-dists_t, axis=1)
        """

        """
        # 源域和目标域数据混合输入
        keep_s = tf.squeeze(tf.matmul(self.D, tf.constant([[0], [1.]])))  # 源域样本稀疏索引
        keep_t = tf.squeeze(tf.matmul(self.D, tf.constant([[1.], [0]])))  # 目标域样本稀疏索引
        s_idx = tf.reshape(tf.where(tf.equal(keep_s, tf.constant(1.))), (-1,))  # 源域样本索引
        t_idx = tf.reshape(tf.where(tf.equal(keep_t, tf.constant(1.))), (-1,))  # 目标域样本索引
        encoding_s_s = tf.gather(encoding_s, s_idx, axis=0)  # 源域共享编码
        encoding_t_s = tf.gather(encoding_s, t_idx, axis=0)  # 目标域共享编码
        encoding_s_p = tf.gather(encoding_p, s_idx, axis=0)  # 源域私有编码
        encoding_t_p = tf.gather(encoding_p, t_idx, axis=0)  # 目标域私有编码
        Y_s = tf.gather(self.Y, s_idx, axis=0)  # 源域标签
        Y_t = tf.gather(self.Y, t_idx, axis=0)  # 目标域标签
        source_logits = self.source_clf_mlp.apply(encoding_s_s + encoding_s_p)
        target_logits = self.source_clf_mlp.apply(encoding_t_s + encoding_t_p)
        # shared_s_logits = self.shared_clf_mlp.apply(encoding_s_s)
        # shared_t_logits = self.shared_clf_mlp.apply(encoding_t_s)
        # private_s_logits = self.source_clf_mlp.apply(encoding_s_p)
        # private_t_logits = self.target_clf_mlp.apply(encoding_t_p)
        logits = tf.cond(tf.equal(tf.reduce_sum(keep_s), tf.reduce_sum(self.D)),
                         lambda: source_logits,
                         lambda: tf.cond(tf.equal(tf.reduce_sum(keep_t), tf.reduce_sum(self.D)),
                                         lambda: target_logits,
                                         lambda: tf.concat([source_logits,
                                                            target_logits], axis=0)))
        self.probs = tf.nn.softmax(logits)  # 只在选择伪标签的时候用到，所以输入只有一个域的数据
        self.C_loss = self.gamma_c * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat([Y_s, Y_t], axis=0), logits=logits)) + l2_norm
        self.C_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(
            loss=self.C_loss,
            var_list=self.source_clf_mlp.parameters)
        correct_preds = tf.equal(tf.argmax(tf.concat([Y_s, Y_t], axis=0), axis=1), tf.argmax(logits, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        """

        """
        所有域共享一个任务分类器的方案
        logits = self.task_output_mlp.apply(encoding_s + encoding_p)  # 暂时只设置一个任务分类器，两种编码表示相加，待会再试试级联
        self.logits = logits
        self.probs = tf.nn.softmax(logits)
        self.C_loss = self.gamma_c * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)) +l2_norm
        self.theta_c = self.task_output_mlp.parameters
        self.C_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_loss,
                                                                                     var_list=self.theta_c)
        correct_preds = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(logits, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        """

        """
        错误代码
        通过观察sigmoid和softmax求导公式发现，即使将原始输入x、编码后的嵌入表示、标签全部置为全0向量，
        梯度仍然不为0，所以即使损失为0也仍然会去更新参数！
        # source_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred)) \
        #               / (tf.reduce_sum(self.Y) + tf.constant(1e-8, dtype=tf.float32))
        # mask = 1. - tf.tile(tf.reshape(tf.reduce_sum(self.Y, axis=-1),
        #                                [-1, 1]),
        #                     [1, self.FLAGS.n_classes])  # 源域样例不计算信息熵，只计算交叉熵
        # target_loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(mask * pred), logits=pred))  # 用v2合适吗？
        # self.C_t_loss = self.gamma_c * tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=pred))
        # self.C_loss = self.gamma_c * (source_loss) + l2_norm  # 目标域任务标签为全0向量，对应交叉熵损失为0
        """
        """
        熵最小化实现方案
        # pred = self.task_output_mlp.apply(encoding_s)
        # self.pred = pred
        # self.prob = tf.nn.softmax(pred)
        # mean_pred = tf.tile(
        #     tf.reshape(tf.reduce_mean(pred, axis=0), [-1, 2]), [self.FLAGS.batch_size, 1])
        # self.C_s_loss = self.gamma_c * tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred)) + l2_norm
        # self.C_t_loss = self.gamma_c * (
        #     tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=pred)) -
        #     tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=mean_pred))) + l2_norm
        # self.theta_c = self.task_output_mlp.parameters
        # self.C_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_s_loss,
        #                                                                                var_list=self.theta_c)
        # self.C_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_t_loss,
        #                                                                                var_list=self.theta_c)
        # correct_prediction = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(pred, axis=1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """
        # optimizing the parameters of the private encoder Ep
        # self.Diff_loss = diff_loss(emb_s, emb_p, self.gamma_diff)
        # self.theta_p = self.private_encode_mlp.parameters
        # self.P_s_loss = self.R_loss + self.gamma_d * loss_p + self.Diff_loss + self.C_s_loss - l2_norm
        # self.P_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_s_loss,
        #                                                                                var_list=self.theta_p)
        # self.P_t_loss = self.R_loss + self.gamma_d * loss_p + self.Diff_loss + self.C_t_loss - l2_norm
        # self.P_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_t_loss,
        #                                                                                var_list=self.theta_p)
        self.Diff_loss = diff_loss(emb_s, emb_p, self.gamma_diff)
        self.theta_p = self.private_encode_mlp.parameters
        self.P_loss = self.R_loss + self.gamma_d * loss_p + self.Diff_loss #+ self.C_loss - l2_norm
        self.P_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_loss,
                                                                                     var_list=self.theta_p)

        # optimizing the parameters of the shared encoder Es
        # self.theta_s = self.shared_encode_mlp.parameters
        # self.S_s_loss = self.R_loss + self.Diff_loss + self.C_s_loss - self.gamma_d * loss_s - l2_norm
        # self.S_t_loss = self.R_loss + self.Diff_loss + self.C_t_loss - self.gamma_d * loss_s - l2_norm
        # self.S_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_s_loss,
        #                                                                                var_list=self.theta_s)
        # self.S_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_t_loss,
        #                                                                                var_list=self.theta_s)
        self.theta_s = self.shared_encode_mlp.parameters
        self.S_loss = self.R_loss - self.gamma_d * loss_s + self.Diff_loss + self.C_loss - l2_norm
        self.S_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_loss,
                                                                                     var_list=self.theta_s)


class MLP(object):
    def __init__(self, name, dims, activations):
        self.name = name
        self.dims = dims
        self.activations = activations
        self.weights = []
        self.biases = []
        self._initialize()

    @property
    def parameters(self):
        return self.weights + self.biases

    def _initialize(self):
        for i in range(len(self.dims) - 1):
            w = tf.Variable(xavier_init([self.dims[i], self.dims[i + 1]]), name=self.name + '_w_{0}'.format(i))
            b = tf.Variable(xavier_init([self.dims[i + 1]]), name=self.name + '_b_{0}'.format(i))
            self.weights.append(w)
            self.biases.append(b)

    def apply(self, x):
        out = x
        for a, w, b in zip(self.activations, self.weights, self.biases):
            out = a(tf.add(tf.matmul(out, w), b))
        return out


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # xavier初始化方法的标准差计算公式，置out_dim=0
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def identity(x):
    return x


def get_l2_norm(theta):
    l2_norm = 0.
    for tensor in theta:
        l2_norm += tf.reduce_sum(tf.abs(tensor))
    return 0.0001 * l2_norm


def euclidean_distance(a, b):
    """
    a.shape = N * D
    b.shape = M * D
    """
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    # 参考原型网络的开源
    return tf.reduce_sum(tf.square(a - b), axis=2)


def diff_loss(shared_feat, task_feat, gamma):
    """
    Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    """

    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * gamma
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
        loss_diff = tf.identity(cost)

    return loss_diff


if __name__ == '__main__':
    shared = tf.random_normal(shape=[50, 50])
    task = tf.random_normal(shape=[50, 50])
    loss = diff_loss(shared, task, 1.0)
    print(tf.Session().run(loss))

# import numpy as np
# import tensorflow as tf
#
#
# class Batch(object):
#     def __init__(self, source_x, source_y, target_x, batch_size):
#         self.x_s = source_x
#         self.y_s = source_y
#         self.x_t = target_x
#         self.batch_size = batch_size
#
#     def generate(self, domain='source', shuffle=False):
#         d_s = np.tile(np.array([0., 1]), (len(self.x_s), 1))  # 源域的域标签
#         d_t = np.tile(np.array([1., 0]), (len(self.x_t), 1))  # 目标域的域标签
#         y_t = np.tile(np.array([0., 0]), (len(self.x_t), 1))  # 目标域的分类任务标签，有问题，损失为0梯度不一定为0
#
#         if domain == 'source':
#             x = self.x_s
#             y = self.y_s
#             d = d_s
#         else:
#             x = self.x_t
#             y = y_t
#             d = d_t
#
#         if shuffle:
#             shuffled_data = np.random.permutation(list(zip(x, y, d)))
#         else:
#             shuffled_data = list(zip(x, y, d))
#
#         sample_nums = len(x)
#         batch_nums = int(sample_nums / self.batch_size)
#         for start in range(batch_nums):
#             yield shuffled_data[start * self.batch_size: min((start + 1) * self.batch_size, sample_nums)]
#
#
# class AdaptiveModel(object):
#     def __init__(self, flags):
#         self.FLAGS = flags
#         self.gamma_r = 1.0
#         self.gamma_c = 0.5
#         self.gamma_d = 0.1
#         self.gamma_diff = 0.01
#
#     def build_model(self):
#         # tf graph input
#         self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
#         self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])
#         self.D = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_domains])
#
#         # ff network
#         self.shared_encode_mlp = MLP('shared_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_s],
#                                      [tf.nn.sigmoid])
#         self.private_s_encode_mlp = MLP('private_s_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_p],
#                                         [tf.nn.sigmoid])
#         self.private_t_encode_mlp = MLP('private_t_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_p],
#                                         [tf.nn.sigmoid])
#         self.shared_decode_mlp = MLP('shared_decode_mlp',
#                                      [self.FLAGS.n_hidden_s, (self.FLAGS.n_hidden_s + self.FLAGS.n_input) // 2,
#                                       self.FLAGS.n_input],
#                                      [tf.nn.tanh, tf.nn.relu])
#         self.private_decode_mlp = MLP('private_decode_mlp',
#                                       [self.FLAGS.n_hidden_p, (self.FLAGS.n_hidden_p + self.FLAGS.n_input) // 2,
#                                        self.FLAGS.n_input],
#                                       [tf.nn.tanh, tf.nn.relu])
#         self.domain_output_mlp = MLP('domain_output_mlp', [self.FLAGS.n_hidden_s, self.FLAGS.n_domains],
#                                      [identity])
#         self.task_output_mlp = MLP('task_output_mlp', [self.FLAGS.n_hidden_s + self.FLAGS.n_hidden_p, self.FLAGS.n_classes],
#                                    [identity])
#
#         encoding_s = self.shared_encode_mlp.apply(self.X)  # 共享编码
#         encoding_p_s = self.private_s_encode_mlp.apply(self.X)  # 源域数据私有编码
#         encoding_p_t = self.private_t_encode_mlp.apply(self.X)  # 目标域数据私有编码
#         self.total_theta = (self.shared_encode_mlp.parameters +
#                             self.private_s_encode_mlp.parameters +
#                             self.private_t_encode_mlp.parameters +
#                             self.shared_decode_mlp.parameters +
#                             self.private_decode_mlp.parameters +
#                             self.domain_output_mlp.parameters +
#                             self.task_output_mlp.parameters)
#         l2_norm = get_l2_norm(self.total_theta)
#
#         # optimizing the parameters of the decoder F
#         decoding_s = self.shared_decode_mlp.apply(encoding_s)  # 共享表示解码器
#         decoding_p_s = self.private_decode_mlp.apply(encoding_p_s)  # 解码源域私有表示
#         decoding_p_t = self.private_decode_mlp.apply(encoding_p_t)  # 解码目标域私有表示
#         recon_s = decoding_s + decoding_p_s
#         recon_t = decoding_s + decoding_p_t
#         # 用一个私有解码器来解码所有领域的私有表示，合理吗？
#         self.theta_f = self.shared_decode_mlp.parameters + self.private_decode_mlp.parameters
#         self.R_loss_s = self.gamma_r * tf.reduce_mean(tf.square(recon_s - self.X)) + l2_norm  # 重建损失用来优化解码器
#         self.R_loss_t = self.gamma_r * tf.reduce_mean(tf.square(recon_t - self.X)) + l2_norm  # 目标域重建损失
#         self.R_solver_s = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.R_loss_s,
#                                                                                        var_list=self.theta_f)
#         self.R_solver_t = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.R_loss_t,
#                                                                                        var_list=self.theta_f)
#
#         # optimizing the parameters of the domain classifier D
#         pred_d = self.domain_output_mlp.apply(encoding_s)  # 共享表示领域判别
#         # pred_p = self.domain_output_mlp.apply(encoding_p)  # 私有表示领域判别，暂时不考虑
#         loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_d))
#         # loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D, logits=pred_p))
#         self.D_loss = self.gamma_d * (loss_d) + l2_norm
#         self.theta_d = self.domain_output_mlp.parameters
#         self.D_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.D_loss,
#                                                                                      var_list=self.theta_d)
#
#         # optimizing the parameters of the label classifier C
#         pred_s = self.task_output_mlp.apply(tf.concat([encoding_s, encoding_p_s], 1))  # 暂时采用相加策略，后面要测试级联的效果
#         pred_t = self.task_output_mlp.apply(tf.concat([encoding_s, encoding_p_t], 1))  # 目标域编码用于分类
#         self.pred_s = pred_s
#         self.prob = tf.nn.softmax(pred_s)
#
#         """
#         source_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred)) \
#                       / (tf.reduce_sum(self.Y) + tf.constant(1e-8, dtype=tf.float32))
#         mask = 1. - tf.tile(tf.reshape(tf.reduce_sum(self.Y, axis=-1),
#                                        [-1, 1]),
#                             [1, self.FLAGS.n_classes])  # 源域样例不计算信息熵，只计算交叉熵
#         target_loss = tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(mask * pred), logits=pred))  # 用v2合适吗？
#         self.C_t_loss = self.gamma_c * tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred), logits=pred))
#         self.C_loss = self.gamma_c * (source_loss) + l2_norm  # 目标域任务标签为全0向量，对应交叉熵损失为0
#         """
#
#         mean_pred = tf.tile(
#             tf.reshape(tf.reduce_mean(pred_s, axis=0), [-1, 2]), [self.FLAGS.batch_size, 1])
#         self.C_s_loss = self.gamma_c * tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_s)) + l2_norm
#         self.C_t_loss = self.gamma_c * (
#                 tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred_t), logits=pred_t)) -
#                 tf.reduce_mean(
#                     tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(pred_t),
#                                                                logits=mean_pred))) + l2_norm
#         self.theta_c = self.task_output_mlp.parameters
#         self.C_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_s_loss,
#                                                                                        var_list=self.theta_c)
#         self.C_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.C_t_loss,
#                                                                                        var_list=self.theta_c)
#         correct_pred_s = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(pred_s, axis=1))
#         correct_pred_t = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(pred_t, axis=1))
#         self.accuracy_s = tf.reduce_mean(tf.cast(correct_pred_s, tf.float32))
#         self.accuracy_t = tf.reduce_mean(tf.cast(correct_pred_t, tf.float32))
#
#         # optimizing the parameters of the private encoder Ep
#         Diff_loss_s = diff_loss(encoding_s, encoding_p_s, self.gamma_diff)  # 源域数据正交约束
#         Diff_loss_t = diff_loss(encoding_s, encoding_p_t, self.gamma_diff)  # 目标域数据正交约束
#         self.P_loss_s = self.R_loss_s + self.C_s_loss + Diff_loss_s  # 源域私有编码器的损失
#         self.P_loss_t = self.R_loss_t + Diff_loss_t  # 目标域私有编码器的损失
#         self.theta_p_s = self.private_s_encode_mlp.parameters
#         self.theta_p_t = self.private_t_encode_mlp.parameters
#         self.P_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_loss_s,
#                                                                                        var_list=self.theta_p_s)
#         self.P_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.P_loss_t,
#                                                                                        var_list=self.theta_p_t)
#
#         # optimizing the parameters of the shared encoder Es
#         self.theta_s = self.shared_encode_mlp.parameters
#         self.S_s_loss = self.C_s_loss + self.R_loss_s + Diff_loss_s - self.D_loss
#         self.S_t_loss = self.R_loss_t + Diff_loss_t - self.D_loss + l2_norm  # + self.C_t_loss - l2_norm
#         self.S_s_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_s_loss,
#                                                                                        var_list=self.theta_s)
#         self.S_t_solver = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_t_loss,
#                                                                                        var_list=self.theta_s)
#         # self.S_solver = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.S_loss,
#         #                                                                                 var_list=self.theta_s)
#
#
# class MLP(object):
#     def __init__(self, name, dims, activations):
#         self.name = name
#         self.dims = dims
#         self.activations = activations
#         self.weights = []
#         self.biases = []
#         self._initialize()
#
#     @property
#     def parameters(self):
#         return self.weights + self.biases
#
#     def _initialize(self):
#         for i in range(len(self.dims) - 1):
#             w = tf.Variable(xavier_init([self.dims[i], self.dims[i + 1]]), name=self.name + '_w_{0}'.format(i))
#             b = tf.Variable(xavier_init([self.dims[i + 1]]), name=self.name + '_b_{0}'.format(i))
#             self.weights.append(w)
#             self.biases.append(b)
#
#     def apply(self, x):
#         out = x
#         for a, w, b in zip(self.activations, self.weights, self.biases):
#             out = a(tf.add(tf.matmul(out, w), b))
#         return out
#
#
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # xavier初始化方法的标准差计算公式，置out_dim=0
#     return tf.random_normal(shape=size, stddev=xavier_stddev)
#
#
# def identity(x):
#     return x
#
#
# def get_l2_norm(theta):
#     l2_norm = 0.
#     for tensor in theta:
#         l2_norm += tf.reduce_sum(tf.abs(tensor))
#     return 0.0 * l2_norm
#
#
# def diff_loss(shared_feat, task_feat, gamma):
#     """
#     Constraints from https://github.com/tensorflow/models,
#     in directory research/domain_adaptation
#     """
#
#     task_feat -= tf.reduce_mean(task_feat, 0)
#     shared_feat -= tf.reduce_mean(shared_feat, 0)
#
#     task_feat = tf.nn.l2_normalize(task_feat, 1)
#     shared_feat = tf.nn.l2_normalize(shared_feat, 1)
#
#     correlation_matrix = tf.matmul(
#         task_feat, shared_feat, transpose_a=True)
#
#     cost = tf.reduce_mean(tf.square(correlation_matrix)) * gamma
#     cost = tf.where(cost > 0, cost, 0, name='value')
#
#     assert_op = tf.Assert(tf.is_finite(cost), [cost])
#     with tf.control_dependencies([assert_op]):
#         loss_diff = tf.identity(cost)
#
#     return loss_diff
