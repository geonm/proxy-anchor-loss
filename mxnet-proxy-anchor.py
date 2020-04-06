import mxnet

class proxy_anchor_loss(mx.gluon.loss.Loss):
    def __init__(self, input_dim, n_classes, n_unique, scale, margin, ctx):
        super(proxy_anchor_loss, self).__init__()
        self.proxy = mxnet.gluon.Parameter('proxy', shape=(n_classes, input_dim))
        self.proxy.initialize(ctx=ctx)
        self.n_classes = n_classes
        self.n_unique = n_unique # n_unique = batch_size // n_instance
        self.alpha = scale
        self.delta = margin

    def hybrid_forward(self, F, embeddings, target):
        embeddings_L2 = F.L2Normalization(embeddings)
        proxy_l2 = F.L2Normalization(self.proxy)

        sim_mat = F.dot(embeddings_L2, proxy_l2, transpose_b=True)

        pos_target = F.one_hot(target, self.n_classes)
        neg_target = 1.0 - pos_target
        
        pos_mat = sim_mat * pos_target
        neg_mat = sim_mat * neg_target

        pos_term = 1.0 / self.n_unique * F.sum(F.log(1.0 + F.sum(F.exp(-self.alpha * (pos_mat - self.delta)), axis=0)))
        neg_term = 1.0 / self.n_classes * F.sum(F.log(1.0 + F.sum(F.exp(self.alpha * (neg_mat + self.delta)), axis=0)))
        
        loss = pos_term + neg_term

        return loss
