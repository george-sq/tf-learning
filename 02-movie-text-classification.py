#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2020-05-28 16:42:43
@Author : nlpqs-George
@File   : 02-movie-text-classification.py
@Todo   : demo-电影评论文本分类
"""

import logging

import keras
import matplotlib.pyplot as plt

_log_format = "%(asctime)s [ %(levelname)s ] | %(message)-100s || %(filename)s(line:%(lineno)s)-%(process)d(%(thread)d)"
_date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
logging.basicConfig(level=logging.DEBUG,
                    format=_log_format,
                    datefmt=_date_format)
logger = logging.getLogger()


def decode_text(word_index: dict = None, encoded_seq=None):
    d_index_word = dict([(value, key) for (key, value) in word_index.items()])
    pass
    return ' '.join([d_index_word.get(i, '<???>') for i in encoded_seq])


def show_image(dict_history):
    # 模型训练过程状态
    d_history = dict_history.history
    print(d_history.keys())

    acc = d_history['accuracy']
    val_acc = d_history['val_accuracy']
    loss = d_history['loss']
    val_loss = d_history['val_loss']
    epochs = range(1, len(acc) + 1)

    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()  # 清除数字

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    pass


def app():
    # 数据集加载
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    logger.info("Raw train entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    logger.info("Raw test  entries: {}, labels: {}".format(len(test_data), len(test_labels)))
    print()
    # 获取数据集的 词汇索引字典
    d_word_index = imdb.get_word_index()
    # l_words = ['\t'.join([str(word), str(w_index)]) for word, w_index in d_word_index.items()][:10]
    # some_words = "\n".join(l_words)
    # print(f"{some_words}")

    # 新增 词汇符号 进入词汇索引字典，并调整词汇索引
    # print(decode_text(d_word_index, train_data[0]))  # 文本原文(调整词汇索引前，貌似转换的内容不对)
    # print()
    d_word_index = {k: (v + 3) for k, v in d_word_index.items()}
    d_word_index["<PAD>"] = 0
    d_word_index["<START>"] = 1
    d_word_index["<UNK>"] = 2  # unknown
    d_word_index["<UNUSED>"] = 3

    # 查看具体的文本数据情况
    # print(train_data[0])  # 文本数字化情况
    print(decode_text(d_word_index, train_data[0]))  # 文本原文
    # print(f"train_data[0] len: {len(train_data[0])}")  # 不同文本的单词长度可能不同
    # print(f"train_data[1] len: {len(train_data[1])}")

    # 将原始文本数据，进行 预处理编码
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=d_word_index["<PAD>"], padding='post',
                                                            maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=d_word_index["<PAD>"], padding='post',
                                                           maxlen=256)
    logger.info("Train shape: {}, Test shape: {}".format(train_data.shape, test_data.shape))
    # 创建交叉验证集
    x_val = train_data[:10000]
    y_val = train_labels[:10000]
    partial_x_train = train_data[10000:]
    partial_y_train = train_labels[10000:]

    # 模型设计
    # 输入形状是用于电影评论的词汇数目（10,000 词）
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()  # 查看模型结构

    # 设置模型训练参数
    optimizer = "adam"
    loss_fnc = "binary_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss_fnc, metrics=metrics)

    # 进行模型训练
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)
    # 模型训练过程状态
    show_image(history)

    # 模型评估
    results = model.evaluate(test_data, test_labels, verbose=2)
    print(results)

    pass


if __name__ == "__main__":
    app()
    pass

# %%
