import tensorflow as tf
import mnist2.mnist_new as mn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnistMode = True
if mnistMode:
    channels = 1
    num_classes = 10
    X_heigh = 28
    X_width = 28
    reshape = False
    dataPath = 'mnist_data'
else:
    channels = 3
    num_classes = 12
    X_heigh = 144
    X_width = 256
    reshape = True
    dataPath = 'notmnist_data'

mnist = mn.read_data_sets(dataPath, one_hot=True, num_classes=num_classes, channels=channels)

mb_size = 16
n_sample = 16
X_dim = mnist.train.images.shape[1] * channels
Y_dim = mnist.train.labels.shape[1]
z_dim = 100
h_dim = 128
lam = 10
n_disc = 5
lr = 1e-4
test_times = 100000000
print_freq = 1000
condition_label_index = 7

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if reshape:
            plt.imshow(sample.reshape(X_heigh, X_width, channels))
        else:
            plt.imshow(sample.reshape(X_heigh, X_width), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
Y = tf.placeholder(tf.float32, shape=[None, Y_dim])
Y_fill = tf.placeholder(tf.float32, shape=[None, X_heigh, X_width, Y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

def sample_z(m, n):
    #return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(0, 1, size=[m, n])


def D_Deep(images, labels_fill, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        images = tf.reshape(images, [-1, X_heigh, X_width, channels])
        inputs = tf.concat([images, labels_fill], 3)

        d_w1 = tf.get_variable('d_w1', [5, 5, channels + Y_dim, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=inputs, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # First fully connected layer
        numWeights = (X_heigh/4) * (X_width/4) * 64
        d_w3 = tf.get_variable('d_w3', [numWeights, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, numWeights])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

def G_Deep(samples, labels):
    inputs = tf.concat(axis=1, values=[samples, labels])
    print 'inputs:', inputs
    g_w1 = tf.get_variable('g_w1', [z_dim+Y_dim, X_heigh*X_width*4*channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [X_heigh*X_width*4*channels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(inputs, g_w1) + g_b1
    print 'g_w1:', g_w1
    print 'g_b1:', g_b1
    print 'g1:', g1
    g1 = tf.reshape(g1, [-1, X_heigh*2, X_width*2, channels])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)
    print 'g1:', g1

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, channels, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [X_heigh*2, X_width*2])
    print 'g2:', g2

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [X_heigh*2, X_width*2])
    print 'g3:', g3

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [channels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)
    print 'g4:', g4
    return tf.reshape(g4, [-1, X_dim])

G_sample = G_Deep(z, Y)
D_real = D_Deep(X, Y_fill)
D_fake = D_Deep(G_sample, Y_fill, reuse_variables=True)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D_Deep(X_inter, Y_fill, reuse_variables=True), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=d_vars))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=g_vars))

tf.get_variable_scope().reuse_variables()

saver = tf.train.Saver()

if not os.path.exists('out/'):
    os.makedirs('out/')
if not os.path.exists('out/model/'):
    os.makedirs('out/model/')

with tf.Session() as sess:
    if os.path.exists('out/model/model.ckpt.meta'):
        print ("Model Restore")
        saver.restore(sess, 'out/model/model.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    i = 0

    for it in range(test_times):
        for _ in range(n_disc):
            X_mb, Y_mb = mnist.train.next_batch(mb_size)
            X_mb =  X_mb.reshape(X_mb.shape[0], X_dim)
            y_label = Y_mb.reshape([mb_size, 1, 1, Y_dim])
            y_fill =  y_label * np.ones([mb_size, X_heigh, X_width, Y_dim])
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), Y:Y_mb, Y_fill: y_fill}
            )

        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: sample_z(mb_size, z_dim), Y:Y_mb, Y_fill: y_fill}
        )
        if it % print_freq == 0:
            print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                .format(it, D_loss_curr, G_loss_curr))

            y_sample = np.zeros(shape=[n_sample,Y_dim])
            y_sample[:, condition_label_index] = 1
            samples = sess.run(G_sample, feed_dict={z: sample_z(n_sample, z_dim), Y:y_sample})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight', dpi= 300)
            i += 1
            plt.close(fig)
            save_path = saver.save(sess, "out/model/model.ckpt")
