import tensorflow as tf

class BaseModel(tf.keras.Model):
    """
    Base model implementing common functionality for all neural network models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    def weights(self):
        raise NotImplementedError(
            "Weights have to be specified by a concrete model implementation")

    def score(self, feats):
        return self(feats, train=False)

    def save(self, file_prefix):
        tf.contrib.eager.Saver(self.weights()).save(file_prefix)

    def restore(self, file_prefix):
        tf.contrib.eager.Saver(self.weights()).restore(file_prefix)