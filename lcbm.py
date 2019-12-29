#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import Counter

import dataset_util as ds_util
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score
import keras
from keras.layers import Dense, Activation, BatchNormalization
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag, Bernoulli
import time
from batch_generator import BatchGenerator
from evaluation import get_metrics, print_result
import os
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"

np.random.seed(0)
tf.set_random_seed(0)


def calculate_score(y_pred, y_true):
    acc_score = accuracy_score(y_pred, y_true)
    ham_loss = hamming_loss(y_pred, y_true)
    jac_score = jaccard_similarity_score(y_pred, y_true)
    return acc_score, ham_loss, jac_score


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def _most_common(lst):
    return Counter(lst).most_common()[0][0]


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, epochs=10, transfer_fct=tf.nn.leaky_relu,
                 learning_rate=0.0001, batch_size=128, sample_size=4, predict_sample_size=2000, categories=10):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.predict_sample_size = predict_sample_size
        self.epsilon = 1e-6
        self.epochs = epochs
        self.categories = categories

        self.x_view1 = tf.placeholder(tf.float32, [None, network_architecture["n_input_view_1"]])
        self.x_view2 = tf.placeholder(tf.float32, [None, network_architecture['n_input_view_2']])
        self.y_label = tf.placeholder(tf.float32, [None, network_architecture["n_label"]])
        self.tau = tf.placeholder(tf.float32, [], name="temperature")

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Define predicitor
        self._create_predictor()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def sample_gumble(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumble(tf.shape(logits))

        return tf.nn.softmax(y / temperature)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
          Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
          Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
          """
        y = self.gumbel_softmax_sample(logits=logits, temperature=temperature)
        if hard:
            k = tf.shape(logits)[-1]
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq, self.logits = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # c = gumble_softmax(logits)
        self.c = self.gumbel_softmax(self.logits, self.tau, hard=False)
        self.q_c = tf.nn.softmax(self.logits)
        self.log_q_c = tf.log(self.q_c + 1e-20)

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.input_reconstr_mean = \
            self._generator_network(self.network_weights["weights_gener"],
                                    self.network_weights["biases_gener"])
        # get latent varable x
        # gaussian_mixture = Mixture(
        #     cat=Categorical(probs=self.q_c),
        #     components=MultivariateNormalDiag(
        #         loc=self.z_mean,
        #         scale_diag=tf.sqrt(tf.exp(self.z_log_sigma_sq))# wrong should appear component’s multivariateNormal
        # ),name='sample_x from Gaussian Mixture model')
        # self.sample_x = tf.squeeze(gaussian_mixture.sample(1), 0)

        pass

    def _initialize_weights(self, n_input_view_1, n_input_view_2, n_label,
                            view_1_hidden_recog_1, view_1_hidden_recog_2, view_1_hidden_units,
                            view_2_hidden_recog_1, view_2_hidden_recog_2, view_2_hidden_units,
                            n_hidden_recog_1, n_hidden_recog_2, n_hidden_size,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_z, n_c, n_input, n_component):
        all_weights = dict()
        all_weights['weights_recog'] = {
            # view 1
            "view1_h1": tf.Variable(xavier_init(n_input_view_1, view_1_hidden_recog_1)),
            "view1_h2": tf.Variable(xavier_init(view_1_hidden_recog_1, view_1_hidden_units)),
            # "view1_h3": tf.Variable(xavier_init(view_1_hidden_recog_2, view_1_hidden_units)),
            # view 2
            "view2_h1": tf.Variable(xavier_init(n_input_view_2, view_2_hidden_recog_1)),
            "view2_h2": tf.Variable(xavier_init(view_2_hidden_recog_1, view_2_hidden_units)),
            # "view2_h3": tf.Variable(xavier_init(view_2_hidden_recog_2, view_2_hidden_units)),
            # get x
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'h3': tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_size)),
            'h4': tf.Variable(xavier_init(n_hidden_size, n_hidden_size)),

            "h4_1": tf.Variable(xavier_init(n_hidden_size, n_hidden_size)),
            "h4_2": tf.Variable(xavier_init(n_hidden_size, n_hidden_size // 2)),
            "out_logit": tf.Variable(xavier_init(n_hidden_size // 2, n_c)),

            'out_mean': tf.Variable(xavier_init(n_hidden_size, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_size, n_z))
        }
        all_weights['biases_recog'] = {
            "view1_b1": tf.Variable(tf.zeros([view_1_hidden_recog_1], dtype=tf.float32)),
            "view1_b2": tf.Variable(tf.zeros([view_1_hidden_units], dtype=tf.float32)),
            # "view1_b3": tf.Variable(tf.zeros([view_1_hidden_units], dtype=tf.float32)),
            "view2_b1": tf.Variable(tf.zeros([view_2_hidden_recog_1], dtype=tf.float32)),
            "view2_b2": tf.Variable(tf.zeros([view_2_hidden_units], dtype=tf.float32)),
            # "view2_b3": tf.Variable(tf.zeros([view_2_hidden_units], dtype=tf.float32)),
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            'b4': tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            "b4_1": tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            "b4_2": tf.Variable(tf.zeros([n_hidden_size // 2], dtype=tf.float32)),
            "out_logit": tf.Variable(tf.zeros([n_c], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_size)),
            'h1_1': tf.Variable(xavier_init(n_c, n_hidden_size)),
            'h2': tf.Variable(xavier_init(n_hidden_size, n_hidden_size)),
            'h3': tf.Variable(xavier_init(n_hidden_size, n_hidden_gener_1)),
            'h4': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'output': tf.Variable(xavier_init(n_hidden_gener_2, n_input))
        }
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            "b1_1": tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_size], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b4': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'b_output': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }
        with tf.variable_scope(name_or_scope='cbm', reuse=tf.AUTO_REUSE):
            all_weights['Alpha'] = tf.get_variable(name='Alpha', shape=[n_z, n_component],
                                                   initializer=tf.truncated_normal_initializer)
            all_weights['Alpha_bias'] = tf.get_variable(name='Alpha_bias', shape=[n_component],
                                                        initializer=tf.zeros_initializer)
            all_weights['Beta'] = tf.get_variable(name='Beta', shape=[n_component, n_z, n_label],
                                                  initializer=tf.truncated_normal_initializer)
            all_weights['Beta_bias'] = tf.get_variable(name="Beta_bias", shape=[n_component, n_label],
                                                       initializer=tf.zeros_initializer)
        return all_weights

    @property
    def alpha(self):
        return self.sess.run(self.network_weights['Alpha'])

    @property
    def alpha_bias(self):
        return self.sess.run(self.network_weights['Alpha_bias'])

    @property
    def beta(self):
        return self.sess.run(self.network_weights['Beta'])

    @property
    def beta_bias(self):
        return self.sess.run(self.network_weights['Beta_bias'])

    @staticmethod
    def _multi_layer_network(input_tensor, hidden_units, activation=tf.nn.relu):
        model = keras.Sequential([
            Dense(units=hidden_units[0], input_shape=(input_tensor.shape[1].value,),
                  kernel_initializer='glorot_uniform'),
            Activation('relu'),
            BatchNormalization()
        ])
        # net = tf.contrib.layers.fully_connected(input_tensor, hidden_units[0], activation)
        for num_hidden_units in hidden_units[1:]:
            model.add(Dense(units=num_hidden_units, kernel_initializer='glorot_uniform'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            # net = tf.contrib.layers.fully_connected(net, num_hidden_units, activation)
        return model(input_tensor)

    def _recognition_network(self, weights, biases):
        """
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        """
        with tf.variable_scope(name_or_scope='recognition_layers_view_1', reuse=tf.AUTO_REUSE):
            view_1_layer1 = self.transfer_fct(tf.add(tf.matmul(self.x_view1, weights['view1_h1']), biases['view1_b1'])),
            view_1_layer1 = view_1_layer1[0]
            view_1_layer1 = tf.layers.batch_normalization(view_1_layer1)
            # view_1_layer2 = self.transfer_fct(
            #     tf.add(tf.matmul(view_1_layer1, weights['view1_h2']), biases['view1_b2'])),
            # view_1_layer2 = view_1_layer2[0]
            # view_1_layer2 = tf.layers.batch_normalization(view_1_layer2)
            view_1_hidden_units = self.transfer_fct(
                tf.add(tf.matmul(view_1_layer1, weights['view1_h2']), biases["view1_b2"]))

        with tf.variable_scope(name_or_scope='recognition_layers_view_2', reuse=tf.AUTO_REUSE):
            view_2_layer1 = self.transfer_fct(tf.add(tf.matmul(self.x_view2, weights['view2_h1']), biases['view2_b1'])),
            view_2_layer1 = view_2_layer1[0]
            view_2_layer1 = tf.layers.batch_normalization(view_2_layer1)
            # view_2_layer2 = self.transfer_fct(
            #     tf.add(tf.matmul(view_2_layer1, weights['view2_h2']), biases['view2_b2'])),
            # view_2_layer2 = view_2_layer2[0]
            # view_2_layer2 = tf.layers.batch_normalization(view_2_layer2)
            view_2_hidden_units = self.transfer_fct(
                tf.add(tf.matmul(view_2_layer1, weights['view2_h2']), biases['view2_b2']))

        # self.input = tf.nn.sigmoid(tf.concat([view_1_hidden_units, view_2_hidden_units], axis=1))
        self.input = tf.concat([view_1_hidden_units, view_2_hidden_units], axis=1)
        assert self.input.shape[1].value == self.network_architecture['n_input']

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.input, weights['h1']),
                                           biases['b1']))
        layer_1 = tf.layers.batch_normalization(layer_1)
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        layer_2 = tf.layers.batch_normalization(layer_2)
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']),
                                           biases['b3']))
        layer_3 = tf.layers.batch_normalization(layer_3)
        layer_4 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h4']),
                                           biases['b4']))
        layer_4 = tf.layers.batch_normalization(layer_4)
        layer_4_1 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h4_1']),
                                             biases['b4_1']))
        layer_4_1 = tf.layers.batch_normalization(layer_4_1)
        layer_4_2 = self.transfer_fct(tf.add(tf.matmul(layer_4_1, weights['h4_2']),
                                             biases['b4_2']))
        layer_4_2 = tf.layers.batch_normalization(layer_4_2)

        z_mean = tf.add(tf.matmul(layer_4, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_4, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        logits = \
            tf.add(tf.matmul(layer_4_2, weights["out_logit"]),
                   biases['out_logit'])

        return (z_mean, z_log_sigma_sq, logits)

    def _generator_network(self, weights, biases):
        """
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        """
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_1_1 = self.transfer_fct(tf.add(tf.matmul(self.c, weights['h1_1']),
                                             biases['b1_1']))
        layer_synthesis = \
            tf.multiply(layer_1, layer_1_1)
        # layer_synthesis = \
        #     tf.add(layer_1,layer_1_1)

        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_synthesis, weights['h2']),
                                           biases['b2']))
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']),
                                           biases['b3']))
        layer_4 = self.transfer_fct(tf.add(tf.matmul(layer_3, weights['h4']),
                                           biases['b4']))

        input_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['output']),
                                 biases['b_output']))
        return input_reconstr_mean

    def _create_loss_optimizer(self):
        """
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # xent_loss = metrics.binary_crossentropy(self.x,self.x_reconstr_mean)
        # x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
        # BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)


        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.

        # kl_loss = - 0.5 * K.sum(1 + self.z_log_sigma_sq - K.square(self.z_mean) - K.exp(self.z_log_sigma_sq), axis=-1)
        # 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean - yh) - K.exp(z_log_var), axis=-1)
        # y = Input(shape=(num_classes,))  # 输入类别
        # yh = Dense(latent_dim)(y)  # 这里就是直接构建每个类别的均值
        # KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder),reduction_indices=1)

        """
        reconstr_loss = \
            -tf.reduce_sum(self.input * tf.log(self.epsilon + tf.clip_by_value(self.input_reconstr_mean, 1e-15, 1))
                           + (1 - self.input) * tf.log(
                self.epsilon + 1 - tf.clip_by_value(self.input_reconstr_mean, 1e-15, 1)),
                           1)

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

        categories_prior_loss = tf.reduce_sum(self.q_c * (self.log_q_c - tf.log(1.0 / self.categories)), 1)

        self.loss1 = tf.reduce_mean(reconstr_loss)
        self.loss2 = tf.reduce_mean(latent_loss)
        self.loss3 = tf.reduce_mean(categories_prior_loss)

        n_z = self.network_architecture["n_z"]
        # the number of samples L per datapoint can be set to 1
        # as long as the minibatch size M is large enough, e.g. M = 128
        sample_eps = tf.random_normal((self.sample_size, n_z), 0, 1, dtype=tf.float32)

        # ----------------------------- An implement of CBM: Gate function p(z|x)---------------------------------------
        # p(z|x) gate function   Mulitnomial Logistics Regression
        def eqx_zx(eps):
            x = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            Ax = tf.matmul(x, self.network_weights['Alpha']) + self.network_weights['Alpha_bias']
            probs = tf.nn.softmax(Ax + self.epsilon)
            return tf.log(probs + self.epsilon)

        # E_q(x)[ln p(z|x)]
        E_qx_ln_pzx = tf.reduce_mean(tf.map_fn(eqx_zx, sample_eps), axis=0)

        # -----------------------------An implement of CBM: Bernoulli distribution p(y|x,z)-----------------------------
        def eqx_y(eps):
            x = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            betas = self.network_weights['Beta']
            betas_bias = self.network_weights['Beta_bias']
            logits = tf.map_fn(lambda beta: tf.matmul(x, beta), betas)
            logits = tf.transpose(logits, [1, 0, 2]) + betas_bias
            y_tile = tf.tile(tf.expand_dims(self.y_label, 1), [1, self.network_architecture["n_component"], 1])
            mu = tf.nn.sigmoid(logits)
            log_mu = tf.log(mu + self.epsilon)
            log_one_sub_mu = tf.log(1 - mu + self.epsilon)
            log_probs = log_mu * y_tile + log_one_sub_mu * (1 - y_tile)
            return tf.reduce_sum(log_probs, axis=2)

        # E_q(x)[ln p(y|x,z)]
        E_qx_ln_pyxz = tf.reduce_mean(tf.map_fn(eqx_y, sample_eps), axis=0)
        # gamma_logits = tf.expand_dims(E_qx_ln_px12, 1) + E_qx_ln_pzx + E_qx_ln_pyxz
        gamma_logits = E_qx_ln_pzx + E_qx_ln_pyxz
        # CAVI
        gamma = tf.nn.softmax(gamma_logits)
        # smooth_gamma = gamma + 1e-7
        self.gamma = gamma

        # l1 表示变分下界的第一项，表示模型逼近真实先验的能力，就是(负号） ---KL(q(z)||p(z))
        # l2 表示变分下界的第二项，即表示模型生成能力的一项，就是表征模型生成能力的一项
        l2 = gamma * (E_qx_ln_pzx + E_qx_ln_pyxz - tf.log(gamma + self.epsilon))
        # 至此完成了整个变分下界的放缩
        elbo = tf.reduce_sum(l2, axis=1)

        # l2_regularization
        l2_regularization = 0.5 * (
                tf.nn.l2_loss(self.network_weights['Alpha'])
                + tf.nn.l2_loss(self.network_weights['Beta'])
        )
        self.cost_prior = tf.reduce_mean(reconstr_loss + latent_loss + categories_prior_loss)  # average over batch
        # self.cost = 0.2*self.cost_prior - 0.8*tf.reduce_mean(elbo) + l2_regularization
        self.cost = -tf.reduce_mean(elbo) + l2_regularization
        # self.cost = -tf.reduce_mean(elbo)

        # Use ADAM optimizer
        self.max_gradient_norm = 5
        # Use ADAM optimizer
        with tf.variable_scope(name_or_scope='optimizer', reuse=tf.AUTO_REUSE):
            # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # train_vars.remove(self.network_weights['Alpha'])
            # train_vars.remove(self.network_weights['Alpha_bias'])

            trainable_params = tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.cost, trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            # Update the model
            # self.train_all_op  = self.optimizer.apply_gradients(zip(clip_gradients,trainable_params),
            #                                              global_step = self.glob_step)
            self.train_all_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            # self.train_all_op = self.optimizer.minimize(self.cost)

            # self.train_beta_op = self.optimizer.minimize(self.cost, var_list=train_vars)
            # self.train_alpha_op = self.optimizer.minimize(self.cost, var_list=[self.network_weights['Alpha'],
            #                                                                    self.network_weights['Alpha_bias']])

    # -----------------------------------------------------------------------------------------------------------------#
    #                                           halving line  of CBM                                                   #
    # -----------------------------------------------------------------------------------------------------------------#

    def _create_predictor(self):
        network_weights = self.network_weights

        mv_normal_dist = MultivariateNormalDiag(self.z_mean, tf.sqrt(tf.exp(self.z_log_sigma_sq)))
        sample_x = tf.squeeze(mv_normal_dist.sample(1), 0)

        z_logits = tf.matmul(sample_x, network_weights['Alpha']) + network_weights['Alpha_bias']
        # z_logits = tf.clip_by_norm(z_logits, 5)
        z_probs = tf.nn.softmax(z_logits + self.epsilon)
        # z_probs = self.gate_model(sample_x)
        cate = Categorical(probs=z_probs)
        ds = tf.contrib.distributions
        y_logits = [
            tf.matmul(sample_x, network_weights['Beta'][i]) + network_weights['Beta_bias'][i]
            for i in range(self.network_architecture["n_component"])
        ]
        self.y_logits = y_logits

        components = [
            ds.Independent(Bernoulli(y_logits[i]))
            for i in range(self.network_architecture["n_component"])
        ]
        cbm = Mixture(cat=cate, components=components)
        self.cat = cate
        self.components = components
        self.cbm = cbm

    def partial_fit(self, x_view1, x_view2, y, train_alpha=False):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        self.epochs += 1

        # if train_alpha:
        #     train_op = self.train_alpha_op
        # else:
        #     train_op = self.train_beta_op

        train_op = self.train_all_op

        opt, cost = self.sess.run(
            (train_op, self.cost),
            feed_dict={self.x_view1: x_view1, self.x_view2: x_view2, self.y_label: y, self.tau: 0.1}
        )
        return cost

    def _partial_predict(self, x_view1, x_view2):
        sample_y = self.sess.run(
            tf.transpose(self.cbm.sample(self.predict_sample_size), [1, 0, 2]),
            feed_dict={self.x_view1: x_view1, self.x_view2: x_view2}
        )
        results = [_most_common((map(tuple, y))) for y in sample_y]

        return np.array(results, dtype=np.int32)

    def predict(self, x_view1, x_view2, batch_size=128):
        len_x = x_view1.shape[0]
        sections = len_x // batch_size + 1
        x_view1_split = np.array_split(x_view1, sections)
        x_view2_split = np.array_split(x_view2, sections)
        results = []
        for x_view1_sp, x_view2_sp in zip(x_view1_split, x_view2_split):
            results.append(self._partial_predict(x_view1_sp, x_view2_sp))
        return np.concatenate(results)

    def fit(self, x_view1_train, x_view2_train, y_train, x_view1_test, x_view2_test, y_test):
        batch_size = self.batch_size
        epochs = self.epochs
        train_batch = BatchGenerator(x_view1_train, x_view2_train, y_train)
        total_batch = len(x_view1_train) // batch_size + 1
        epoch_list = []
        time_dict = {}
        for epo in range(1, epochs + 1):
            start_time = time.time()
            pbar = trange(total_batch * 10, desc="Epoch %d" % epo)
            # Loop over all batches
            for _ in pbar:
                view1, view2, labels = train_batch.next_batch(batch_size)
                cost = self.partial_fit(view1, view2, labels)
                pbar.set_postfix_str("Train Beta, Cost: %s" % cost)
            end_time = time.time()
            time_dict[epo] = (end_time - start_time) * 1000 + time_dict.get(epo - 1, 0)
            time_dis = time_dict.get(epo, 0)
            y_predict = self.predict(x_view1_test, x_view2_test)
            epoch_metrics = get_metrics(y_test, y_predict)
            epoch_metrics['time'] = time_dis
            print(epoch_metrics)
            epoch_list.append(epoch_metrics)
        self.sess.close()
        return sorted(epoch_list)[0]

    def score(self, x_view1, x_view2, y):
        y_predict = self.predict(x_view1, x_view2)
        # acc_score = accuracy_score(y_predict, y)
        # ham_loss = hamming_loss(y_predict, y)
        # jac_score = jaccard_similarity_score(y_predict, y)
        result_dict = get_metrics(y, y_predict)
        return result_dict

    def save_model(self):
        saver = tf.train.Saver()
        ts = int(time.time())
        save_path = saver.save(self.sess, "/tmp/model%d.ckpt" % ts)
        print("Model saved in path: %s" % save_path)


if __name__ == "__main__":

    # 分别载入四个数据集
    # X, y = ds_util.load_scene()
    # X, y = ds_util.load_emotions()
    X, y = ds_util.load_human()
    # X, y = ds_util.load_plant()
    # scaler = StandardScaler()
    # X0_scaler = scaler.fit_transform(X[0])
    # X1_scaler = scaler.fit_transform(X[1])
    # X = (X0_scaler, X1_scaler)

    print("X_view1 : %s" % str(X[0].shape))
    print("X_view2 : %s" % str(X[1].shape))
    print("y : %s" % str(y.shape))

    dataset_generator = ds_util.split_dataset2(X[0], X[1], y)

    result_list = {}
    for split, (train, test) in enumerate(dataset_generator, start=1):
        x_view1_train, x_view2_train, y_train = train
        x_view1_test, x_view2_test, y_test = test
        network_architecture = {
            "n_input_view_1": x_view1_train.shape[1],
            "n_input_view_2": x_view2_train.shape[1],
            "n_label": y_train.shape[1],
            # view_1
            "view_1_hidden_recog_1": 200,  # view_1 1st layer encoder neurons
            "view_1_hidden_recog_2": 100,  # view_1 2nd layer encoder neurons
            "view_1_hidden_units": 50,  # view_1 contributed encoder neurons
            # view 2
            "view_2_hidden_recog_1": 200,  # view_2 1st layer encoder neurons
            "view_2_hidden_recog_2": 100,  # view_2 2nd layer encoder neurons
            "view_2_hidden_units": 50,  # view_2 contributed encoder neurons
            # fused view shared representation :concated view_1_hidden_units and view_2 hidden_units
            "n_hidden_recog_1": 30,
            "n_hidden_recog_2": 30,
            "n_hidden_size": 50,
            "n_hidden_gener_1": 50,
            "n_hidden_gener_2": 50,
            "n_z": 200,  # dimensionality of latent space 80-100
            "n_c": 10,  # categorical_latent_size : categorical latent space dimension (constant =K)
            "n_input": 100,  # fused view
            "n_component": 30
        }
        model = VariationalAutoencoder(network_architecture, epochs=6, learning_rate=0.001, batch_size=128,
                                       sample_size=8,
                                       predict_sample_size=2000)
        print("Split %d\n" % split)
        result = model.fit(x_view1_train, x_view2_train, y_train, x_view1_test, x_view2_test, y_test)
        for key, value in result.items():
            if key not in result_list.keys():
                result_list[key] = [value]
            else:
                result_list[key].append(value)
    print_result(result_list)
