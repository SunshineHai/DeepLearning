# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """
    返回一个元组做为 Mnist 数据集，该元组包含 训练集、验证集和测试集。
    返回的训练集是一个元组包含2个元素。 第一个元素 包含实际训练图像，这是含有 50000条数据的 ndarray 对象，
    每条 mnist 图像中含有784个值，代表 28*28=784像素，即 training_data[0].shape = (50000, 784)。
    training_data 元组中的第二个元素也是一个 numpy ndarray 数组，包含50000个数据，维度是(50000,)。
    这些值是 (0, 9),表示 training_data 元组中第一个值相对应的图片的标签。

    验证集和测试集是相似的，除了每个元素都仅包含 10000 张图片。

    这是一个比较好的数据格式，但是对于在神经网络的使用，training_data 需要修改一下格式。
    wrapper 函数 `load_data_wrapper()`` 就做了这个功能，请往下看.


    # training_data : ((50000, 784), (50000,)) 元组：2个元素，都是 NumPy对象， 维度如前所示
    # validation_data : ((10000, 784), (10000,))
    # test_data : ((10000, 784), (10000,))
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
        注： zip() 返回元组的迭代器，其中第 i 个元组包含的是每个参数迭代器的第 i 个元素。
        使用 for循环/list()后， 返回训练集、验证集和测试集数据，格式如下：
        训练集：[((784, 1), (10, 1)), ((784, 1), (10, 1)), ... ]
        验证集和测试集格式一样： [((784, 1), (标签)), ((784, 1), (标签)), ...]
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]    # [(784, 1), (784, 1), ...]
    training_results = [vectorized_result(y) for y in tr_d[1]]      # [(10, 1), (10, 1)]
    training_data = zip(training_inputs, training_results)          # 组合成zip()对象(使用for或list()后) : ((784， 1), (10， 1))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]  # [(784, 1), (784, 1), ...]
    validation_data = zip(validation_inputs, va_d[1])               # 返回zip()对象 : ((784， 1), 标签)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    返回 1个索引j的位置是1.0，其它位置是0.0的 10维的单元向量。这样做是为了把 (0, 9)
    转化成 相应的 神经网络的期望输出。

    返回的是二维数组 (10, 1)
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
