import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mwcnn import IWT, MWCNN


def test_iwt():
    x = tf.random.normal([1, 191, 241, 4])
    x_res = IWT()(x)
    x1 = (x[..., 0:1] / 2).numpy()
    x2 = (x[..., 1:2] / 2).numpy()
    x3 = (x[..., 2:3] / 2).numpy()
    x4 = (x[..., 3:4] / 2).numpy()
    x_expected = np.zeros([1, 2*191, 2*241, 1])
    x_expected[:, 0::2, 0::2] = x1 - x2 - x3 + x4
    x_expected[:, 1::2, 0::2] = x1 - x2 + x3 - x4
    x_expected[:, 0::2, 1::2] = x1 + x2 - x3 - x4
    x_expected[:, 1::2, 1::2] = x1 + x2 + x3 + x4
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllClose(x_res, x_expected)

def test_mwcnn():
    model = MWCNN(
        n_filters_per_scale=[2, 4, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=2,
    )
    shape = [1, 80, 64, 1]
    res = model(tf.zeros(shape))

    assert res.shape.as_list() == shape

def test_mwcnn_conference():
    model = MWCNN(
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=0,
        first_conv_n_filters=0,
        bn=True,
    )
    shape = [1, 32, 32, 1]
    res = model(tf.zeros(shape))
    assert res.shape.as_list() == shape

def test_mwcnn_change(x):
    model = MWCNN(
        n_filters_per_scale=[4, 8, 16],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
    )
    y = x
    # model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='adam', loss='mse')
    for r in range(82):
        model.train_on_batch(x[r], x[r])
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
    plt.imshow(model.predict(x[10]).reshape(64, 80))
    plt.show()
