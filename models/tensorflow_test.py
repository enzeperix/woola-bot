import tensorflow as tf

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)

# Test simple GPU computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Result of Matrix Multiplication on GPU:\n", c)

