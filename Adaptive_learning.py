from multi_target_model import *
from dataset import *

flags = tf.flags
FLAGS = flags.FLAGS


class AdaptiveTrainer(object):
    def __init__(self, flags):
        self.FLAGS = flags
        self.model_save_to = "output/model/{0}_to_{1}.pkl".format(flags.source_domain, flags.target_domain)

    def train_whole_model(self, batch, x_valid, y_valid, x_test, y_test):
        """
        用源域数据及目标域标注(或伪标签)数据训练整个模型
        """
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                R_loss, D_loss, C_loss, Diff_loss, P_loss, S_loss = 0., 0., 0., 0., 0., 0.
                train_accuracy = 0.
                for b in batch.generate(shuffle=True):
                    x, y, d = zip(*b)
                    _, r_loss = self.sess.run([model.R_solver, model.R_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, d_loss = self.sess.run([model.D_solver, model.D_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, p_loss = self.sess.run([model.P_solver, model.P_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, s_loss, di_loss, accuracy, c_loss = self.sess.run(
                        [model.S_solver, model.S_loss, model.Diff_loss, model.acc, model.C_loss],
                        feed_dict={model.X: x, model.Y: y, model.D: d})
                    R_loss += r_loss
                    D_loss += d_loss
                    Diff_loss += di_loss
                    C_loss += c_loss
                    P_loss += p_loss
                    S_loss += s_loss
                    train_accuracy += accuracy
                batch_nums = (len(batch.x_s) + len(batch.x_t_tune)) / batch.batch_size
                print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, '
                      's_loss: {4}, diff_loss: {5}, train_acc: {6}'.format(
                    R_loss / batch_nums,
                    D_loss / batch_nums,
                    C_loss / batch_nums,
                    P_loss / batch_nums,
                    S_loss / batch_nums,
                    Diff_loss / batch_nums,
                    train_accuracy / batch_nums))

                if train_accuracy / batch_nums > 0.7:
                    valid_accuracy = model.acc_test.eval({model.X: batch.x_s, model.Y: batch.y_s,
                                                          model.test_X: x_valid, model.test_Y: y_valid},
                                                         session=self.sess)  # 这里应该用哪个域的训练数据生成原型呢？
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(sess=self.sess, save_path=self.model_save_to)
                    else:
                        wait_times += 1
                    if wait_times > self.FLAGS.tolerate_time:
                        print('best_result: {0}'.format(best_result))
                        break
                    print('valid_accuracy: {0}'.format(valid_accuracy))
            saver.restore(self.sess, self.model_save_to)
            test_accuracy = model.acc_test.eval({model.X: batch.x_s, model.Y: batch.y_s,
                                                 model.test_X: x_test, model.test_Y: y_test},
                                                session=self.sess)
            print('test_accuracy: {0}'.format(test_accuracy))
            return best_result, test_accuracy

    def get_predictions(self, x_s, y_s, x_ts):
        """
        对每个目标域的样本生成概率分布
        """
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)
        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, save_path=self.model_save_to)
            probs = []
            for i in range(len(x_ts)):
                prob, = self.sess.run([model.prob],
                                      feed_dict={model.X: x_s, model.Y: y_s, model.test_X: x_ts[i]})
                probs.append(prob)

        return probs

    def select_samples(self, u_xs, probs):
        x, y, d, unlabeled_xs = [], [], [], []
        for i in range(len(probs)):  # 对每个目标域的未标注样本和预测标签概率分布
            pos_idxes = set()
            neg_idxes = set()
            left_indexes = set(range(len(u_xs[i])))

            idxes = np.argsort(probs[i][:, 0])
            end_idx = (probs[i][:, 0][idxes] < 0.3).sum()  # 预测负例个数
            begin_idx = (probs[i][:, 0][idxes] > 0.7).sum()  # 预测正例个数
            end_idx = end_idx if end_idx > 0 else 1  # 每次至少选两个
            begin_idx = begin_idx if begin_idx > 0 else 1

            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            left_indexes = left_indexes.intersection(idxes[end_idx: -begin_idx])
            pos_idxes = np.array(list(pos_idxes))
            neg_idxes = np.array(list(neg_idxes))
            left_indexes = np.array(list(left_indexes))

            x_p = u_xs[i][pos_idxes]
            x_n = u_xs[i][neg_idxes]
            y_p = np.zeros(shape=(len(pos_idxes), 2), dtype='float32')
            y_p[:, 0] = 1.
            y_n = np.zeros(shape=(len(neg_idxes), 2), dtype='float32')
            y_n[:, 1] = 1.
            x.extend(np.concatenate([x_p, x_n], axis=0))
            y.extend(np.concatenate([y_p, y_n], axis=0))
            d.extend(np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x), 1)))
            unlabeled_x = u_xs[i][left_indexes] if left_indexes.size else np.array([], dtype='float32')
            unlabeled_xs.append(unlabeled_x)

            print('Pseudo label: {}'.format(len(x_p)+len(x_n)))
            print('Unlabeled samples: {}'.format(len(unlabeled_x)))

        return x, y, d, unlabeled_xs

    def train(self, batch, x_valid, y_valid, x_test, y_test):
        """
        self-training
        """
        best_result = 0.
        final_test_acc = 0.
        wait_times = 0

        new_batch = batch
        unlabeled_xs = batch.x_ts
        x_t, y_t, d_t = batch.x_t_tune, batch.y_t_tune, batch.d_t_tune
        min_len = min([len(x) for x in batch.x_ts])

        while min_len > 0:
            print('Self-training...')
            valid_acc, test_acc = self.train_whole_model(new_batch, x_valid, y_valid, x_test, y_test)
            probs = self.get_predictions(batch.x_s, batch.y_s, unlabeled_xs)  # 源域数据计算原型，作为目标域的分类器
            x_pseudo, y_pseudo, d_pseudo, unlabeled_xs = self.select_samples(unlabeled_xs, probs)
            x_t = np.concatenate([x_t, x_pseudo], axis=0)
            y_t = np.concatenate([y_t, y_pseudo], axis=0)
            d_t = np.concatenate([d_t, d_pseudo], axis=0)
            new_batch = Batch(batch.x_s, batch.y_s, batch.d_s, x_t, y_t, d_t, unlabeled_xs, batch.batch_size)
            min_len = min([len(x) for x in unlabeled_xs])

            if valid_acc > best_result:
                best_result = valid_acc
                final_test_acc = test_acc
                wait_times = 0
            else:
                wait_times += 1
            if wait_times > self.FLAGS.tolerate_time:
                print('best result: {}'.format(best_result))
                break
        print('Test accuracy: {}'.format(final_test_acc))

    # def train(self, batch, x_valid, y_valid, x_test, y_test):
    #     """
    #     不是严格的self-training
    #     """
    #     best_result = 0.
    #     final_test_acc = 0.
    #     wait_times = 0
    #     print('Pre-train model...')
    #     self.train_with_labeled_data(batch, x_valid, y_valid, x_test, y_test)
    #     probs = self.get_predictions(batch.x_t)
    #     x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
    #     while True:
    #         print('Self-training...')
    #         new_batch = Batch(batch.x_s, batch.y_s,  # 加入伪标签，生成新的train batch
    #                           np.concatenate([batch.x_t_tune, x_pseudo], axis=0),
    #                           np.concatenate([batch.y_t_tune, y_pseudo], axis=0),
    #                           batch.x_t, batch.batch_size)
    #         valid_acc, test_acc = self.train_with_labeled_data(new_batch, x_valid, y_valid, x_test, y_test)
    #         probs = self.get_predictions(batch.x_t)
    #         x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
    #         if valid_acc > best_result:
    #             best_result = valid_acc
    #             final_test_acc = test_acc
    #             wait_times = 0
    #         else:
    #             wait_times += 1
    #         if wait_times > self.FLAGS.tolerate_time:
    #             print('best result: {}'.format(best_result))
    #             break
    #     print('Test accuracy: {}'.format(final_test_acc))

    # def train(self, batch, x_valid, y_valid, x_test, y_test):
    #     wait_times = 0
    #     best_result = 0.
    #     self.graph = tf.Graph()
    #     tfConfig = tf.ConfigProto()
    #     tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    #     self.sess = tf.Session(graph=self.graph, config=tfConfig)
    #     model = AdaptiveModel(self.FLAGS)
    #
    #     with self.graph.as_default():
    #         model.build_model()
    #         saver = tf.train.Saver(var_list=model.total_theta)
    #         self.sess.run(tf.global_variables_initializer())
    #         # saver.restore(self.sess, self.model_load_from)
    #         while True:
    #             R_loss = 0.
    #             D_loss = 0.
    #             C_loss = 0.
    #             P_loss = 0.
    #             S_loss = 0.
    #             Diff_loss = 0.
    #             train_accuracy = 0.
    #             for b in batch.generate_pretrain_data(shuffle=True):
    #                 x, y, d = zip(*b)
    #                 _, r_loss = self.sess.run([model.R_solver, model.R_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, d_loss = self.sess.run([model.D_solver, model.D_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, c_loss = self.sess.run([model.C_solver, model.C_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, p_loss = self.sess.run([model.P_solver, model.P_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, s_loss, diff_loss, accuracy = self.sess.run(
    #                     [model.S_solver, model.S_loss, model.Diff_loss, model.accuracy],
    #                     feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 R_loss += r_loss
    #                 D_loss += d_loss
    #                 C_loss += c_loss
    #                 P_loss += p_loss
    #                 S_loss += s_loss
    #                 Diff_loss += diff_loss
    #                 train_accuracy += accuracy
    #             # for b in batch.generate(domain='target', shuffle=True):
    #             #     x, y, d = zip(*b)
    #             #     _, r_loss = self.sess.run([model.R_solver_t, model.R_loss_t],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, d_loss = self.sess.run([model.D_solver, model.D_loss],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, c_loss = self.sess.run([model.C_t_solver, model.C_t_loss],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, p_loss = self.sess.run([model.P_t_solver, model.P_loss_t],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, s_loss, = self.sess.run([model.S_t_solver, model.S_t_loss],
    #             #                                feed_dict={model.X: x, model.Y: y, model.D: d})
    #             batch_nums = len(batch.x_s) / batch.batch_size
    #             print(batch_nums)
    #             print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, s_loss: {4}, acc: {5}'.format(
    #                 R_loss / batch_nums,
    #                 D_loss / batch_nums,
    #                 C_loss / batch_nums,
    #                 P_loss / batch_nums,
    #                 S_loss / batch_nums,
    #                 Diff_loss / batch_nums,
    #                 train_accuracy / batch_nums
    #             ))
    #             # print('train_loss: {0}, train_accuracy: {1}'.format(train_loss / batch_nums, train_accuracy / batch_nums))
    #             if train_accuracy / batch_nums > 0.:
    #                 valid_accuracy = model.accuracy.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 # pred = model.pred.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 # encoding = model.encoding.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 if valid_accuracy > best_result:
    #                     best_result = valid_accuracy
    #                     wait_times = 0
    #                     print('Save model...')
    #                     saver.save(sess=self.sess, save_path=self.model_save_to)
    #                 else:
    #                     wait_times += 1
    #                 if wait_times > self.FLAGS.tolerate_time:
    #                     print('best_result: {0}'.format(best_result))
    #                     break
    #                 # print('pred: {0}'.format(pred))
    #                 # print('encoding: {0}'.format(encoding))
    #                 print('valid_accuracy: {0}'.format(valid_accuracy))
    #         saver.restore(self.sess, self.model_save_to)
    #         test_accuracy = model.accuracy.eval({model.X: x_test, model.Y: y_test}, session=self.sess)
    #         print('test_accuracy: {0}'.format(test_accuracy))
    #         return test_accuracy


def main(_):
    x, y, offset = load_amazon(5000, FLAGS.data_load_from)
    # 取出源域数据，和多个目标域数据
    x_s_tr, y_s_tr, x_t_trs, y_t_trs, x_s_tst, y_s_tst, x_t_tsts, y_t_tsts = split_data(
        FLAGS.source_domain, FLAGS.target_domains, x, y, offset, 2000)

    x_s = turn_tfidf(np.concatenate([x_s_tr, x_s_tst], axis=0))
    x_s_tr = np.copy(x_s[:len(x_s_tr)])
    d_s_tr = np.tile(np.eye(len(FLAGS.target_domains) + 1)[0], (len(x_s_tr), 1))  # 源域数据的域标签

    x_t_tune, y_t_tune, d_t_tune, x_ts = [], [], [], []
    for i in range(len(x_t_trs)):
        x_t = turn_tfidf(np.concatenate([x_t_trs[i], x_t_tsts[i]]))
        x_ts.append(x_t[:len(x_t_trs[i])])

        x_t_tst = np.copy(x_t[len(x_t_trs[i]):])
        y_t_tst = np.copy(y_t_tsts[i])
        d_t_tst = np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x_t_tst), 1))  # 第i个目标域的域标签
        x_t_tune.extend(x_t_tst[:50])
        y_t_tune.extend(y_t_tst[:50])
        d_t_tune.extend(d_t_tst[:50])

        x_t_tst = x_t_tst[50:]
        y_t_tst = y_t_tst[50:]

        if FLAGS.target_domains[i] == FLAGS.target_domain:
            x_t_valid = x_t_tst[:500]
            y_t_valid = y_t_tst[:500]
            x_t_tst = x_t_tst[500:]
            y_t_tst = y_t_tst[500:]

    batch = Batch(x_s_tr, y_s_tr, d_s_tr, x_t_tune, y_t_tune, d_t_tune, x_ts, FLAGS.batch_size)
    trainer = AdaptiveTrainer(FLAGS)
    trainer.train(batch, x_t_valid, y_t_valid, x_t_tst, y_t_tst)


flags.DEFINE_string("data_load_from", "data/amazon.mat", "data path")
flags.DEFINE_integer("source_domain", 0, "source domain id")
flags.DEFINE_integer("target_domain", 2, "target domain id")
flags.DEFINE_list("target_domains", [1, 2, 3], "target domain ids")
flags.DEFINE_integer("tolerate_time", 20, "stop training if it exceeds tolerate time")
flags.DEFINE_integer("n_input", 5000, "size of input data")
flags.DEFINE_integer("n_classes", 2, "size of output data")
flags.DEFINE_integer("n_hidden_s", 50, "size of shared encoder hidden layer")
flags.DEFINE_integer("n_hidden_p", 50, "size of private encoder hidden layer")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_float("lr", 1e-4, "learning rate")

if __name__ == "__main__":
    tf.app.run()
