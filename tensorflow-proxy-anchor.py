import tensorflow as tf

# use tensorflow graph with tf 1.xx version

def proxy_anchor_loss(embeddings, target, n_classes, n_unique, input_dim, scale, margin):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    # define proxy weights
    proxy = tf.get_variable(name='proxy', shape=[n_classes, input_dim],
                            dtype=tf.float32,
                            trainable=True)
    embeddings_l2 = tf.nn.l2_normalize(embeddings, axis=1)
    proxy_l2 = tf.nn.l2_normalize(proxy, axis=1)

    pos_target = tf.one_hot(target, self.n_classes, dtype=tf.float32)
    neg_target = 1.0 - pos_target

    sim_mat = tf.matmul(embeddings_l2, proxy_l2, transpose_b=True)

    pos_mat = sim_mat * pos_target
    neg_mat = sim_mat * neg_target

    # n_unique = batch_size // n_instance
    pos_term = 1.0 / n_unique * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(tf.exp(-self.alpha * (pos_mat - self.delta)), axis=0)))
    neg_term = 1.0 / n_classes * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(tf.exp(self.alpha * (neg_mat + self.delta)), axis=0)))

    loss = pos_term + neg_term

    return loss


