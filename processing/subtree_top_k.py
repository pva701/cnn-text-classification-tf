__author__ = 'pva701'

import tensorflow as tf

import utils as tfu

FUN_ACT = tf.sigmoid

class SubtreeTopK:
    def __init__(self, k, mode='processing',
                 leaf_size=None,
                 backend=None,
                 w_backend=None,
                 num_filters=None,
                 lstm_hidden=None,
                 consider_weights_in_weights=False):
        self.k = k
        self.mode = mode
        self.leaf_size = leaf_size
        self.real_backend = backend
        self.weight_backend = w_backend
        self.weight_backend = w_backend
        self.num_filters = num_filters
        self.lstm_hidden = lstm_hidden
        self.consider_weight_in_weight = consider_weights_in_weights
        self.filter_sizes = [self.k]

        if self.real_backend != 'SUM' and self.real_backend != 'CNN' and self.real_backend != 'LSTM':
            raise Exception('Unexpected SubtreeInnerTopK backend')

        if self.weight_backend != 'LSTM' and self.weight_backend != 'CNN':
            raise Exception('Unexpected weight backend')

        if mode == 'processing':
            self.init_with_scope = self.__process_init_with_scope
            self.build_graph = self.__processing_build_graph
        elif mode == 'outer':
            self.init_with_scope = self.__outer_init_with_scope
            self.build_graph = self.__outer_build_graph
        elif mode == 'symbiosis':
            self.init_with_scope = self.__outer_init_with_scope
            self.build_graph = self.__symbiosis_build_graph
        else:
            raise Exception('Unexpected mode')

        self.var_scope = "subtree-top-k-{}".format(self.mode)

    def __outer_init_with_scope(self, window_vec_size, processed_vec_size):
        self.in_size = processed_vec_size
        self.__init_with_scope()

    def __process_init_with_scope(self, window_vec_size):
        self.in_size = window_vec_size
        self.__init_with_scope()

    def __init_with_scope(self):
        with tf.variable_scope(self.var_scope):
            with tf.variable_scope("real_vectors") as scope:
                if self.real_backend == 'SUM':
                    tfu.linear(self.k, 1, "sum")
                elif self.real_backend == 'CNN':
                    self.leaf_size = self.output_vector_size() if self.mode == 'processing' else None
                    tfu.create_cnn(self.filter_sizes, self.num_filters, self.leaf_size or self.in_size)
                elif self.real_backend == 'LSTM':
                    self.leaf_size = self.lstm_hidden if self.mode == 'processing' else None
                    self.cell, self.initial_state = \
                        tfu.create_lstm(self.lstm_hidden, self.leaf_size or self.in_size, scope)

            if self.weight_backend == 'LSTM':
                self.weight_out_size = self.lstm_hidden
            elif self.weight_backend == 'CNN':
                self.weight_out_size = self.num_filters * len(self.filter_sizes)
            else:
                raise Exception('Unexpected weight backend')

            if self.mode == 'processing':
                self.real_in_size = self.leaf_size or self.in_size
                if self.leaf_size:
                    tfu.linear(self.in_size, self.leaf_size, "top")
                    tfu.linear(self.in_size, self.weight_out_size, "top_w")

            elif self.mode == 'outer':
                self.real_in_size = self.in_size
            elif self.mode == 'symbiosis':
                self.real_in_size = self.in_size
                tfu.linear(self.in_size, self.weight_out_size, "top_w")

            with tf.variable_scope("weights_vectors") as scope:
                if self.consider_weight_in_weight:
                    if self.weight_backend == 'LSTM':
                        self.cell_w, self.initial_state_w = tfu.create_lstm(self.lstm_hidden,
                                                                            self.in_size + self.weight_out_size, scope)
                    elif self.weight_backend == 'CNN':
                        tfu.create_cnn(self.filter_sizes, self.num_filters, self.in_size + self.weight_out_size)
                else:
                    if self.weight_backend == 'LSTM':
                        self.cell_w, self.initial_state_w = tfu.create_lstm(self.lstm_hidden, self.real_in_size, scope)
                    elif self.weight_backend == 'CNN':
                        tfu.create_cnn(self.filter_sizes, self.num_filters, self.real_in_size)

    def __symbiosis_build_graph(self, out_repr, _0, n_words, _1, _2, _3, _4,
                                euler, euler_l, euler_r, dropout_keep):

        leaves_vectors = self.__compute_leaves(n_words, out_repr)

        def apply_inner(vectors, i):
            weig = tf.gather(vectors, euler[euler_l[i]:euler_r[i] - 1])[:, self.output_vector_size():]
            r, w = self.subtree_fn(
                tf.gather(out_repr, euler[euler_l[i]:euler_r[i] - 1]),
                weig)
            cn = tf.expand_dims(tf.concat([r, w], 0), 0)
            return tf.concat([vectors, cn], 0)

        all_vectors = tf.foldl(apply_inner, tf.range(0, n_words - 1), initializer=leaves_vectors)

        return all_vectors[:, :self.output_vector_size()]

    def __processing_build_graph(self, n_words, window_vecs, _1, _2, _3, _4, euler, euler_l, euler_r, dropout_keep):
        leaves_vectors = self.__compute_leaves(n_words, window_vecs)

        def apply_inner(vectors, i):
            real = tf.gather(vectors, euler[euler_l[i]:euler_r[i] - 1])[:, :self.output_vector_size()]
            weig = tf.gather(vectors, euler[euler_l[i]:euler_r[i] - 1])[:, self.output_vector_size():]
            r, w = self.subtree_fn(real, weig)
            cn = tf.expand_dims(tf.concat([r, w], 0), 0)
            return tf.concat([vectors, cn], 0)

        all_vectors = tf.foldl(apply_inner, tf.range(0, n_words - 1), initializer=leaves_vectors)

        return all_vectors[:, :self.output_vector_size()]

    def __outer_build_graph(self, out_repr, _0, n_words, _1, _2, _3, _4,
                            euler, euler_l, euler_r, dropout_keep):
        leaves_vectors = tf.map_fn(lambda v: self.subtree_fn(tf.expand_dims(v, 0)), out_repr[:n_words],
                                   dtype=tf.float32)

        inner_vectors = tf.map_fn(lambda i: self.subtree_fn(tf.gather(out_repr, euler[euler_l[i]:euler_r[i] - 1])),
                                  tf.range(0, n_words - 1), dtype=tf.float32)

        return tf.concat([leaves_vectors, inner_vectors], 0)

    def subtree_fn(self, real_vecs, weights_vecs=None):
        with tf.variable_scope(self.var_scope):
            if weights_vecs is None:
                extended_real_vecs = tf.concat([real_vecs, tf.zeros([self.k, self.real_in_size])], 0)
                max_lengths = tf.reduce_sum(tf.square(extended_real_vecs), 1)
                _, indices = tf.nn.top_k(max_lengths, self.k, False, "top-vectors")

                with tf.variable_scope("real_vectors") as scope:
                    scope.reuse_variables()
                    return self.__apply_real_backend(tf.gather(extended_real_vecs, indices), self.real_backend, scope)
            else:
                extended_real_vecs = tf.concat([real_vecs, tf.zeros([self.k, self.real_in_size])], 0)
                extended_weights_vecs = tf.concat([weights_vecs, tf.zeros([self.k, self.weight_out_size])], 0)
                max_lengths = tf.reduce_sum(tf.square(extended_weights_vecs), 1)
                _, indices = tf.nn.top_k(max_lengths, self.k, False, "top-vectors")

                with tf.variable_scope("real_vectors") as scope:
                    scope.reuse_variables()
                    real_slice = tf.gather(extended_real_vecs, indices)
                    real_result = self.__apply_real_backend(real_slice, self.real_backend, scope)

                with tf.variable_scope("weights_vectors") as scope:
                    scope.reuse_variables()
                    if self.consider_weight_in_weight:
                        weig_slice = tf.gather(extended_weights_vecs, indices)
                        weights_result = self.__apply_weight_backend(tf.concat([real_slice, weig_slice], 1),
                                                                     self.weight_backend,
                                                                     scope)
                    else:
                        weights_result = self.__apply_weight_backend(real_slice,
                                                                     self.weight_backend,
                                                                     scope)

                return real_result, weights_result

    def __compute_leaves(self, n_words, windows_or_out_repr_raw):
        if self.mode == 'processing':
            with tf.variable_scope(self.var_scope):
                W_top = tf.get_variable("W_top")
                biases_top = tf.get_variable("biases_top")
                windows_or_out_repr = FUN_ACT(tf.nn.xw_plus_b(windows_or_out_repr_raw, W_top, biases_top))
        else:
            windows_or_out_repr = windows_or_out_repr_raw

        if self.consider_weight_in_weight:
            with tf.variable_scope(self.var_scope) as scope:
                scope.reuse_variables()

                W_top = tf.get_variable("W_top_w")
                biases_top = tf.get_variable("biases_top_w")
                #TODO fix it!!! VVVVVVVVVVVVV
                leaves_weig = tf.nn.tanh(tf.nn.xw_plus_b(windows_or_out_repr_raw, W_top, biases_top))

            def apply_leaf(i):
                r, w = self.subtree_fn(tf.expand_dims(windows_or_out_repr[i], 0),
                                       tf.expand_dims(leaves_weig[i], 0))
                return tf.concat([r, w], 0)

        else:
            def apply_leaf(i):
                r, w = self.subtree_fn(tf.expand_dims(windows_or_out_repr[i], 0),
                                       tf.zeros([0, self.weight_out_size]))
                return tf.concat([r, w], 0)
        return tf.map_fn(apply_leaf, tf.range(0, n_words), dtype=tf.float32)

    def __apply_real_backend(self, vecs, backend, scope):
        if backend == 'SUM':
            W_sum = tf.get_variable("W_sum")
            biases_sum = tf.get_variable("biases_sum")
            return tf.transpose(FUN_ACT(tf.nn.xw_plus_b(tf.transpose(vecs), W_sum, biases_sum)))
        elif backend == 'CNN':
            return tfu.pass_cnn(
                self.filter_sizes,
                self.num_filters,
                vecs,
                self.real_in_size,
                seq_len=self.k,
                use_padding=False)
        elif backend == 'LSTM':
            output, _ = tfu.pass_static_lstm(self.cell, self.initial_state, self.k, vecs, scope)
            return tf.reshape(output, [self.lstm_hidden])

    def __apply_weight_backend(self, vecs, backend, scope):
        if backend == 'CNN':
            return tfu.pass_cnn(
                self.filter_sizes,
                self.num_filters,
                vecs,
                self.real_in_size,
                seq_len=self.k,
                use_padding=False)
        elif backend == 'LSTM':
            output, _ = tfu.pass_static_lstm(self.cell_w, self.initial_state_w, self.k, vecs, scope)
            return tf.reshape(output, [self.lstm_hidden])
        else:
            raise Exception('Unknown weight backend')

    def output_vector_size(self):
        if self.real_backend == 'SUM':
            return self.real_in_size
        elif self.real_backend == 'CNN':
            return self.num_filters * len(self.filter_sizes)
        elif self.real_backend == 'LSTM':
            return self.lstm_hidden

    def l2_loss(self):
        return tf.constant(0.0)
