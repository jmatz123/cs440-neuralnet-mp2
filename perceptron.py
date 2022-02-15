# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # initialize weights and biases to zero
    # return the trained weight and bias parameters

    # W = {}
    # b = 0
    # labels = {}

    # for i in range(len(train_set)) :
    #     labels[i] = 0

    # for i in range(len(train_set[0])) :
    #     W[i]= 0

    # labels = [0]*len(train_set)
    # W = [0]*len(train_set[0])
    # b  = 0

    # print("W = " + W)

    # for i in range(len(train_labels)) :
    #     if train_labels[i] :
    #         labels[i] = 1
    #     else :
    #         labels[i] = -1
    

    weight = np.zeros(len(train_set[0])+1)
    for i in range(max_iter) :
        for j in range(len(train_set)) :
            feature = train_set[j]
            label = train_labels[j]
            dot_prod = np.dot(weight[1:], feature) + weight[0]
            # val = np.sign(dot_prod)

            # if (val == labels[j]) :
            #     continue
            # else :
            #     b += learning_rate * labels[i]
            #     W += train_set[j] * learning_rate * labels[i]
            guess = (0, 1)[dot_prod > 0]

            # if (j == 0) :
            weight[0] += learning_rate * (label - guess)
            # if (j >= 1) :
            weight[1:] += learning_rate * feature * (label - guess)
            
    W = weight[1:]
    b = weight[0]

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    weight, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    # guesses = [0]*len(dev_set)
    # # guesses  = []
    # # for i in range(len(dev_set)) :
    # #     guesses[i] = 0
    
    # for i in range(len(dev_set)) :
    #     inner_function = np.dot(weight, dev_set[i]) + b
    #     val = np.sign(inner_function)

    #     if (val == 1) :
    #         guesses[i] = 1

    # return guesses

    dev_labels = []

    for image in dev_set :
        inner_function = np.dot(weight, image) + b
        val = np.sign(inner_function)
        
        guess = (1, 0)[val <= 0]
        
        dev_labels.append(guess)

    return dev_labels