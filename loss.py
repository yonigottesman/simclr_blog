import tensorflow as tf


class SimCLRLoss(tf.keras.losses.Loss):
    LARGE_NUM = 1e9

    def __init__(self, temperature: float = 0.05, **kwargs):

        super().__init__(**kwargs)
        self.temperature = temperature

    def contrast(self, hidden1, hidden2):

        batch_size = tf.shape(hidden1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * SimCLRLoss.LARGE_NUM

        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / self.temperature
        loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))

        return loss_a

    def call(self, hidden1, hidden2):
        hidden1 = tf.math.l2_normalize(hidden1, -1)
        hidden2 = tf.math.l2_normalize(hidden2, -1)
        loss_a = self.contrast(hidden1, hidden2)
        loss_b = self.contrast(hidden2, hidden1)

        return loss_a + loss_b
