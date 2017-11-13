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

mb_size = 32
n_sample = 16
X_dim = mnist.train.images.shape[1] * channels
Y_dim = mnist.train.labels.shape[1]
z_dim = 10
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

D_W1 = tf.Variable(xavier_init([X_dim + Y_dim,h_dim]), name = "d_w1")
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name = "d_b1")

D_W2 = tf.Variable(xavier_init([h_dim, 1]), name = "d_w2")
D_b2 = tf.Variable(tf.zeros(shape=[1]), name = "d_b2")

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim + Y_dim, h_dim]), name = "g_w1")
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name = "g_b1")

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]), name = "g_w2")
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name = "g_b2")

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def G(z,Y):
    inputs = tf.concat(axis=1, values=[z, Y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def D(X,Y):
    inputs = tf.concat(axis=1, values=[X, Y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = G(z, Y)
D_real = D(X, Y)
D_fake = D(G_sample, Y)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter,Y), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

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
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), Y:Y_mb}
            )

        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: sample_z(mb_size, z_dim), Y:Y_mb}
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
