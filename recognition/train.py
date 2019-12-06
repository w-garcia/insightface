from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os
import platform
import sys
import time

import tensorflow as tf
import yaml

from recognition.backbones.resnet_v1 import ResNet_v1_50
from recognition.data.generate_data import GenerateData
from recognition.losses.loss import arcface_loss, triplet_loss, center_loss, softmax_loss
from recognition.losses.private_loss import p_arcface_loss, p_triplet_loss, p_center_loss, p_softmax_loss
from recognition.models.models import MyModel
from recognition.predict import get_embeddings
from recognition.valid import Valid_Data

from privacy.tensorflow_privacy.privacy.optimizers.dp_optimizer import make_gaussian_optimizer_class
from privacy.tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from privacy.tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from privacy.tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# tf.enable_eager_execution(config=config)

tf.enable_eager_execution()

# log_cfg_path = '../logging.yaml'
# with open(log_cfg_path, 'r') as f:
#     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
# logging.config.dictConfig(dict_cfg)
# logger = logging.getLogger("mylogger")


class Trainer:
    def __init__(self, config):
        self.gd = GenerateData(config)

        self.train_data, cat_num = self.gd.get_train_data()
        valid_data = self.gd.get_val_data(config['valid_num'])
        self.model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=cat_num)
        self.epoch_num = config['epoch_num']
        self.m1 = config['logits_margin1']
        self.m2 = config['logits_margin2']
        self.m3 = config['logits_margin3']
        self.s = config['logits_scale']
        self.alpha = config['alpha']
        self.thresh = config['thresh']
        self.below_fpr = config['below_fpr']
        self.learning_rate = config['learning_rate']
        self.loss_type = config['loss_type']

        # DP params
        self.dp_enabled = config['dp_enabled']
        self.noise_multiplier = config['noise_multiplier']
        self.l2_norm_clip = config['l2_norm_clip']
        self.num_microbatches = config['num_microbatches']
        self.steps_per_epoch = len([x for x in self.train_data])
        self.batch_size = config['batch_size']
        self.train_count = self.steps_per_epoch * config['batch_size']
        print(f"Population size:\t{self.train_count}")
        self.target_delta = config['target_delta']

        # center loss init
        self.centers = None
        self.ct_loss_factor = config['center_loss_factor']
        self.ct_alpha = config['center_alpha']
        if self.loss_type == 'logit' and self.ct_loss_factor > 0:
            self.centers = tf.Variable(initial_value=tf.zeros((cat_num, config['embedding_size'])), trainable=False)

        optimizer = config['optimizer']
        if optimizer == 'ADADELTA':
            opt_cls = tf.keras.optimizers.Adadelta
        elif optimizer == 'ADAGRAD':
            opt_cls = tf.keras.optimizers.Adagrad
        elif optimizer == 'ADAM':
            opt_cls = tf.keras.optimizers.Adam
        elif optimizer == 'ADAMAX':
            opt_cls = tf.keras.optimizers.Adamax
        elif optimizer == 'FTRL':
            opt_cls = tf.keras.optimizers.Ftrl
        elif optimizer == 'NADAM':
            opt_cls = tf.keras.optimizers.Nadam
        elif optimizer == 'RMSPROP':
            opt_cls = tf.keras.optimizers.RMSprop
        elif optimizer == 'SGD':
            opt_cls = tf.keras.optimizers.SGD
        else:
            raise ValueError('Invalid optimization algorithm')
       
        if self.dp_enabled:
            # opt_cls = make_gaussian_optimizer_class(opt_cls)
            opt_cls = DPAdamGaussianOptimizer
            self.optimizer = opt_cls(
                    l2_norm_clip=self.l2_norm_clip, 
                    noise_multiplier=self.noise_multiplier, 
                    num_microbatches=self.num_microbatches, 
                    learning_rate=self.learning_rate)
        else:
            self.optimizer = opt_cls(self.learning_rate)

        ckpt_dir = os.path.expanduser(config['ckpt_dir'])
        

        if self.centers is None:
            self.ckpt = tf.train.Checkpoint(backbone=self.model.backbone, model=self.model, optimizer=self.optimizer)
        else:
            # save centers if use center loss
            self.ckpt = tf.train.Checkpoint(backbone=self.model.backbone, model=self.model, optimizer=self.optimizer,
                                            centers=self.centers)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=5, checkpoint_name='mymodel')

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.vd = Valid_Data(self.model, valid_data)

        summary_dir = os.path.expanduser(config['summary_dir'])
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(summary_dir, current_time, 'train')
        valid_log_dir = os.path.join(summary_dir, current_time, 'valid')
        # self.graph_log_dir = os.path.join(summary_dir, current_time, 'graph')

        if platform.system() == 'Windows':
            train_log_dir = train_log_dir.replace('/', '\\')
            valid_log_dir = valid_log_dir.replace('/', '\\')
            # self.graph_log_dir = self.graph_log_dir.replace('/', '\\')

        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.compat.v2.summary.create_file_writer(valid_log_dir)

        # self.graph_writer = tf.compat.v2.summary.create_file_writer(self.graph_log_dir)
        # tf.compat.v2.summary.trace_on(graph=True, profiler=True)
        # with graph_writer.as_default():
        #     tf.compat.v2.summary.trace_export(name="graph_trace", step=0, profiler_outdir=graph_log_dir)

    @tf.function
    def _train_step(self, img, label):
        with tf.GradientTape(persistent=True) as tape:
            prelogits, dense, norm_dense = self.model(img, training=True)

            # sm_loss = softmax_loss(dense, label)
            # norm_sm_loss = softmax_loss(norm_dense, label)
            # arc_loss = arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)
            
            if self.dp_enabled:
                # _loss = p_softmax_loss(dense, label)
                _loss = p_arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)

                if self.centers is not None:
                    ct_loss, self.centers = p_center_loss(prelogits, label, self.centers, self.ct_alpha)
                else:
                    ct_loss = [0] * int(_loss.shape[0])
                    ct_loss = tf.convert_to_tensor(ct_loss, dtype=float)

            else:
                # _loss = softmax_loss(dense, label)
                _loss = p_arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)

                if self.centers is not None:
                    ct_loss, self.centers = center_loss(prelogits, label, self.centers, self.ct_alpha)
                else:
                    ct_loss = 0           

            logit_loss = _loss
            loss = logit_loss + self.ct_loss_factor * ct_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # reduce for reporting only
        if self.dp_enabled:
            return tf.reduce_mean(loss), tf.reduce_mean(logit_loss), tf.reduce_mean(ct_loss)
        else:
            return loss, logit_loss, ct_loss

    @tf.function
    def _train_triplet_step(self, anchor, pos, neg):
        with tf.GradientTape(persistent=False) as tape:
            anchor_emb = get_embeddings(self.model, anchor)
            pos_emb = get_embeddings(self.model, pos)
            neg_emb = get_embeddings(self.model, neg)
            
            if self.dp_enabled:
                loss = p_triplet_loss(anchor_emb, pos_emb, neg_emb, self.alpha)
            else:
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, self.alpha)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # reduce for reporting only
        if self.dp_enabled:
            return tf.reduce_mean(loss)
        else:
            return loss

    def compute_epsilon(self, steps):
        if self.noise_multiplier == 0.0:
            return float('inf')
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = self.batch_size / self.train_count
        rdp = compute_rdp(q=sampling_probability,
                          noise_multiplier=self.noise_multiplier,
                          steps=steps,
                          orders=orders)
        # target delta: rule of thumb is to set it to be less than the inverse of the training data size (i.e., the population size)
        return get_privacy_spent(orders, rdp, target_delta=self.target_delta)[0]

    def train(self):
        for epoch in range(self.epoch_num):
            start = time.time()
            # triplet loss
            if self.loss_type == 'triplet':
                train_data, num_triplets = self.gd.get_train_triplets_data(self.model)
                print('triplets num is {}'.format(num_triplets))
                if num_triplets > 0:
                    for step, (anchor, pos, neg) in enumerate(train_data):
                        loss = self._train_triplet_step(anchor, pos, neg)
                        with self.train_summary_writer.as_default():
                            tf.compat.v2.summary.scalar('loss', loss, step=step)
                        print('epoch: {}, step: {}, loss = {}'.format(epoch, step, loss))
            elif self.loss_type == 'logit':
                # logit loss
                for step, (input_image, target) in enumerate(self.train_data):
                    loss, logit_loss, ct_loss = self._train_step(input_image, target)
                    with self.train_summary_writer.as_default():
                        tf.compat.v2.summary.scalar('loss', loss, step=step)
                        tf.compat.v2.summary.scalar('logit_loss', logit_loss, step=step)
                        tf.compat.v2.summary.scalar('center_loss', ct_loss, step=step)
                    print('epoch: {}, step: {}, loss = {}, logit_loss = {}, center_loss = {}'.format(epoch, step, loss,
                                                                                                     logit_loss,
                                                                                                     ct_loss))
            else:
                raise ValueError('Invalid loss type')

            # valid
            acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = self.vd.get_metric(self.thresh, self.below_fpr)

            with self.valid_summary_writer.as_default():
                tf.compat.v2.summary.scalar('acc', acc, step=epoch)
                tf.compat.v2.summary.scalar('p', p, step=epoch)
                tf.compat.v2.summary.scalar('r=tpr', r, step=epoch)
                tf.compat.v2.summary.scalar('fpr', fpr, step=epoch)
                tf.compat.v2.summary.scalar('acc_fpr', acc_fpr, step=epoch)
                tf.compat.v2.summary.scalar('p_fpr', p_fpr, step=epoch)
                tf.compat.v2.summary.scalar('r=tpr_fpr', r_fpr, step=epoch)
                tf.compat.v2.summary.scalar('thresh_fpr', thresh_fpr, step=epoch)
                if self.dp_enabled:
                    eps = self.compute_epsilon((epoch + 1) * self.steps_per_epoch)
                    tf.compat.v2.summary.scalar('eps', eps, step=epoch)

                    print('epoch: {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, fpr: {:.3f} \n'
                          'fix fpr <= {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, thresh: {:.3f}, eps: {:.3f}'
                          .format(epoch, acc, p, r, fpr, self.below_fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr, eps))
                else:
                    print('epoch: {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, fpr: {:.3f} \n'
                          'fix fpr <= {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, thresh: {:.3f}'
                          .format(epoch, acc, p, r, fpr, self.below_fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr))


            # ckpt
            # if epoch % 5 == 0:
            save_path = self.ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch, save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    t = Trainer(config)
    t.train()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()
