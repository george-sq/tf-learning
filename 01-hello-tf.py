#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2020-05-28 10:37:04
@Author : nlpqs-George
@File   : hello-tf1.py
@Todo   : description
"""

import logging

import keras
import tensorflow as tf

_log_format = "%(asctime)s [ %(levelname)s ] | %(message)-100s || %(filename)s(line:%(lineno)s)-%(process)d(%(thread)d)"
_date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
logging.basicConfig(level=logging.DEBUG,
                    format=_log_format,
                    datefmt=_date_format)
logger = logging.getLogger()


def app():
    # 数据集加载
    logger.info(f"数据集加载: mnist")
    # mnist = tf.keras.datasets.mnist
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # 设计模型结构
    logger.info("初始化 模型结构:")
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    # 设置模型训练参数
    logger.info("设置模型训练参数:")
    optimizer = "adam"
    loss = "sparse_categorical_crossentropy"
    metrics = ['accuracy']
    logger.info(f">>> optimizer: {optimizer}")
    logger.info(f">>> loss: {loss}")
    logger.info(f">>> metrics: {metrics}")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 进行模型训练
    logger.info("进行模型训练:")
    num_epoch = 5
    logger.info(f"epoch number: {num_epoch}")
    logger.info(f"train data X size: {len(x_train)}")
    logger.info(f"train data Y size: {len(y_train)}")
    model.fit(x_train, y_train, epochs=num_epoch)

    # 模型效果评估
    logger.info("进行模型评估:")
    logger.info(f"test data X size: {len(x_test)}")
    logger.info(f"test data Y size: {len(y_test)}")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    logger.info(f"Test result: loss={test_loss}, accuracy={test_acc}")
    pass


if __name__ == "__main__":
    app()
    pass
