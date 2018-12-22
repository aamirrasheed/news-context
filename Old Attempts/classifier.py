import tensorflow as tf
from tensorflow import keras
# import different optimizer?
from pretrainedModelWikipedia import main as getEmbeddings

def getData():
    # read embeddings in from file
    embeddings = getEmbeddings()

    # read cluster labels in from file

    return

def define_model(numTrainLabels):
    model = keras.Sequential([
        keras.layers.Dense(200, input_shape=(300,), activation=tf.nn.relu),
        keras.layers.Dense(50, activation=tf.nn.relu),
        keras.layers.Dense(numTrainLabels, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    train_labels, train_data, test_labels, test_data = getData()
    print("train_labels shape: ", train_labels.shape)
    print("train_data shape: ", train_data.shape)
    print("test_labels shape: ", test_labels.shape)
    print("test_data shape: ", test_data.shape)

    model = define_model(len(train_labels))
    model.fit(train_data, train_labels, epochs=5)
    _, test_acc = model.evaluate(test_data, test_labels)
    print("Test accuracy: ", test_acc)


main()