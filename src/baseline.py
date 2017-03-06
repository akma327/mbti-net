
import sys 
import tensorflow as tf
import numpy as np
from load_data import *
from glove import *

USAGE_STR = """

python baseline.py /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/mbti-net/data/mbti_shuffled_data.txt 0.1

"""

GLOVE_DIMENSION = 50


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def prep_data(DATA_FILE, PERCENTAGE_TRAIN):
    """
        n = number of data points 
        d = size of glove word embedding 
        c = 16 mbti classes 
        train_X, test_X : Generate n x d matrix with each row being an average over 
        all word embeddings within a sentence 

        train_y, test_y : Generate n x c matric with each row being a one hot 
        vector representation of the labeled personality type. 

    """

    print("prepping data ...")

    def process_rows(data, glove_vectors):
        x_matrix = []
        y_matrix = []

        num_data_points = len(data)

        ### For every row 
        for i, (word_arr, y_arr) in enumerate(data):
            if(i % 100 == 0):
                print(str(i) +"/" + str(num_data_points))
            num_words = 0
            avg_glove = np.array([0.0]*GLOVE_DIMENSION)

            for word in word_arr:
                if(word in glove_vectors):
                    embedding = glove_vectors[word]
                    avg_glove += embedding
                    num_words += 1

            avg_glove /= num_words
            x_matrix.append(avg_glove)
            y_matrix.append(y_arr)

        x_matrix = np.array(x_matrix)
        y_matrix = np.array(y_matrix)

        return x_matrix, y_matrix



    train_data, test_data = load_data(DATA_FILE, PERCENTAGE_TRAIN)
    glove_vectors = loadWordVectors("../data/glove.6B.50d.txt", GLOVE_DIMENSION)
    train_X, train_y = process_rows(train_data, glove_vectors)
    test_X, test_y = process_rows(test_data, glove_vectors)
    
    print(train_X, len(train_X), len(train_X[0]))
    print(train_y, len(train_y), len(train_y[0]))


    return train_X, test_X, train_y, test_y
    



def main(DATA_FILE, PERCENTAGE_TRAIN):
    train_X, test_X, train_y, test_y = prep_data(DATA_FILE, PERCENTAGE_TRAIN)

    print("Running Feed Forward Neural Network ...")

    # Layer's sizes
    x_size = train_X.shape[1]   
    h_size = 50                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes


    # Symbols
    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Weight initializations
    xavier_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable("W", shape=(h_size, y_size), initializer=xavier_initializer)
    b = tf.get_variable("b", shape=(y_size), initializer=tf.constant_initializer(0))

    # Forward propagation
    yhat = tf.matmul(x, W) + b 
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    lr = 0.001
    updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={x: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={x: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={x: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    (DATA_FILE, PERCENTAGE_TRAIN) = (sys.argv[1], float(sys.argv[2]))
    main(DATA_FILE, PERCENTAGE_TRAIN)

