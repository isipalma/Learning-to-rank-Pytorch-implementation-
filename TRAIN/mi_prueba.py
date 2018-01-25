
import numpy
import os

q_train = numpy.load('train.questions.npy')
a_train = numpy.load('train.answers.npy')

print(len(a_train[68]))