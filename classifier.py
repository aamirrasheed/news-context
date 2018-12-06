import tensorflow as tf
from tensorflow import keras
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from learnDocVectors import readData, getTrainedVectors
import random
from xMeans import getLabeledData
def getData():
    # read embeddings in from file
    # read cluster labels in from file
    return

def define_model(numTrainLabels):
    model = keras.Sequential([
        keras.layers.Dense(30, activation=tf.nn.relu),
        keras.layers.Dense(numTrainLabels, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    train_labels, train_data, test_labels, test_data = getLabeledData()
    print("train_labels shape: ", train_labels.shape)
    print("train_data shape: ", train_data.shape)
    print("test_labels shape: ", test_labels.shape)
    print("test_data shape: ", test_data.shape)

    model = define_model(len(train_labels))
    model.fit(train_data, train_labels, epochs=5)
    _, test_acc = model.evaluate(test_data, test_labels)
    print("Test accuracy: ", test_acc)


def examineData():
    train_labels, train_data, test_labels, test_data = getLabeledData()
    seen_labels = []
    data = []


main()