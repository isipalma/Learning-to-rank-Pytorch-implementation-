from datetime import datetime
from sklearn import metrics

import pickle
import numpy
import os
import sys

import time
from tqdm import tqdm

import Classes
from torch import optim, nn
from sklearn import utils
from Functions import *
from torch.autograd import Variable
from torch import cat, from_numpy

import warnings
warnings.filterwarnings("ignore") # to do remove


# ZEROUT_DUMMY_WORD = False
ZEROUT_DUMMY_WORD = True

## Load data
# mode = 'TRAIN-ALL'
mode = 'train'
if len(sys.argv) > 1:
    mode = sys.argv[1]
    if not mode in ['TRAIN', 'TRAIN-ALL']:
        print("ERROR! The two possible training settings are: ['TRAIN', 'TRAIN-ALL']")
        sys.exit(1)

print("Running training in the {} setting".format(mode))

data_dir = mode

if mode in ['TRAIN-ALL']:
    q_train = numpy.load(os.path.join(data_dir, 'train-all.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train-all.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(data_dir, 'train-all.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(data_dir, 'train-all.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train-all.labels.npy'))
else:
    q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

q_dev = numpy.load(os.path.join(data_dir, 'dev.questions.npy'))
a_dev = numpy.load(os.path.join(data_dir, 'dev.answers.npy'))
q_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
a_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
y_dev = numpy.load(os.path.join(data_dir, 'dev.labels.npy'))
qids_dev = numpy.load(os.path.join(data_dir, 'dev.qids.npy'))

q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))

# x_train = numpy.load(os.path.join(data_dir, 'train.overlap_feats.npy'))
# x_dev = numpy.load(os.path.join(data_dir, 'dev.overlap_feats.npy'))
# x_test = numpy.load(os.path.join(data_dir, 'test.overlap_feats.npy'))

# feats_ndim = x_train.shape[1]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# print "Scaling overlap features"
# x_train = scaler.fit_transform(x_train)
# x_dev = scaler.transform(x_dev)
# x_test = scaler.transform(x_test)

print('y_train', numpy.unique(y_train, return_counts=True))
print('y_dev', numpy.unique(y_dev, return_counts=True))
print('y_test', numpy.unique(y_test, return_counts=True))

print('q_train', q_train.shape)
print('q_dev', q_dev.shape)
print('q_test', q_test.shape)

print('a_train', a_train.shape)
print('a_dev', a_dev.shape)
print('a_test', a_test.shape)


## Get the word embeddings from the nnet trained on SemEval
# ndim = 40
# nnet_outdir = 'exp/ndim=60;batch=100;max_norm=0;learning_rate=0.1;2014-12-02-15:53:14'
# nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
# params_fname = os.path.join(nnet_outdir, 'best_dev_params.epoch=00;batch=14640;dev_f1=83.12;test_acc=85.00.dat')
# train_nnet, test_nnet = nn_layers.load_nnet(nnet_fname, params_fname)

numpy_rng = numpy.random.RandomState(123)
q_max_sent_size = q_train.shape[1]
a_max_sent_size = a_train.shape[1]
# print 'max', numpy.max(a_train)
# print 'min', numpy.min(a_train)

ndim = 5
print("Generating random vocabulary for word overlap indicator features with dim:", ndim)
dummy_word_id = numpy.max(a_overlap_train)
# vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
print("Gaussian")
vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.25
# vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.05
# vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
vocab_emb_overlap[-1] = 0

# Load word2vec embeddings
fname = os.path.join(data_dir, 'emb_aquaint+wiki.txt.gz.ndim=50.bin.npy')

print("Loading word embeddings from", fname)

vocab_emb = numpy.load(fname)
ndim = vocab_emb.shape[1]
dummpy_word_idx = numpy.max(a_train)
print("Word embedding matrix size:", vocab_emb.shape)

#######
n_outs = 2

n_epochs = 25
batch_size = 50
learning_rate = 0.1
max_norm = 0

print('batch_size', batch_size)
print('n_epochs', n_epochs)
print('learning_rate', learning_rate)
print('max_norm', max_norm)

## 1st conv layer.
#ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]
ndim = vocab_emb.shape[1]

print(vocab_emb.shape[1])
print(vocab_emb_overlap.shape[1])

#funcion de activacion

dropout_rate = 0.5
nkernels = 100
q_k_max = 1
a_k_max = 1

filter_width = 5

num_input_channels = 1
#input_shape = (batch_size, num_input_channels, q_max_sent_size + 2 * (max(q_filter_widths) - 1), ndim)

model1 = Classes.SiameseNetwork(dropout=dropout_rate, x_join_length= nkernels * 2, filter_width=filter_width, n_kernels=nkernels)

#criterion = nn.NLLLoss()
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model1.parameters(), lr=learning_rate)

#Train
train_loss = []
lookup = Classes.LookUpWord(vocab_emb)

#preparamos el dev set
dev_q_test = lookup.out_matrix(q_dev)
dev_a_test = lookup.out_matrix(a_dev)
best_dev = 0
best = False

for epoch in range(1, n_epochs+1):

    start_epoch = time.time()

    q_train, a_train, y_train = utils.shuffle(q_train, a_train, y_train)
    batch_id = 0
    for x_trainq, x_traina, y_train1 in zip(batch_gen(q_train, 50), batch_gen(a_train, 50), batch_gen(y_train, 50)):
        #model.h_t, model.c_t = repackage_hidden((model.h_t, model.c_t))
        model1.zero_grad() # reinicia el batch, para que no se quede con los gradientes del batch anterior
        batch_id += 1
        x_query =lookup.out_matrix(x_trainq)  # puede ser mas eficiente esto?

        x_answer = lookup.out_matrix(x_traina)
          #inputsq, inputsa, labels = Variable(x_query), Variable(x_answer), Variable(label)

        labels = Variable(from_numpy(numpy.array(y_train1)))
        output = model1(x_query, x_answer)
        batch_size= output.size(0)
        output = output.view(batch_size,2)

        labels = labels.long()
        loss = criterion(output, labels)
        train_loss.append(loss.data[0])

        optimizer.zero_grad()
          # del x_query?
        loss.backward()
        optimizer.step()
    end_epoch = time.time()
    took2 = end_epoch - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took2))

    dev_pred = model1(dev_q_test, dev_a_test)
    normal_sc = score(dev_pred, y_dev)
    map_sc = map_score(qids_dev, dev_pred, y_dev)
    if map_sc > best_dev:
        best_dev = map_sc
        best = True

    if epoch % 5 == 0:
        # revisamos el mejor puntaje usando dev
        if not best:
            print("No ha habido mejora")
            break
        else:
            best = False

#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTook: {:.2f}'.format(
 #       epoch, batch_id * len(labels), len(q_train.shape[0]),100. * batch_id / len(q_train.shape[1]), loss.data[0], took))  # cambiar tamaños, no confiar!

  #torch.save(model.state_dict(), './model-epoch-%s.pth' % epoch)


# prediccion de ejemplo
trad = Classes.Traductor()

qq = trad.traducir(q_test[1])
print("")
aa = trad.traducir(a_test[1])
print(qq)
print(aa)
print(y_test[1])

#test score

query_t = lookup.out_matrix(q_test)
answer_t = lookup.out_matrix(a_test)
# y_test = Variable(y_test)

y_pred = model1(query_t, answer_t)
print(y_pred[1])
print(y_pred[4])

print(score(y_pred, y_test))
print(map_score(qids_test, y_pred, y_test))



