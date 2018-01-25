import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pickle

class LookUpWord:

    def __init__(self, W=None):
        self.W = W
        self.embedding = nn.Embedding(self.W.shape[0], self.W.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(self.W))
        # para "congelar" los parametros
        self.embedding.weight.requires_grad = False

    def out_matrix(self, input):
        # el input debe ser una lista de tipo int
        lista = input.copy()
        lookup_tensor = torch.LongTensor(lista.tolist())
        out = self.embedding(Variable(lookup_tensor))
        return out

    def __repr__(self):
        return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())


class SiameseNetwork(nn.Module):
    def __init__(self, dropout=0, x_join_length=200, filter_width=5, n_kernels=100):
        super(SiameseNetwork, self).__init__()
        self.p_drop = dropout
        self.x_join_l = x_join_length
        self.f_width = filter_width
        self.n_k = n_kernels

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.n_k, padding=(self.f_width - 1, 0), kernel_size=(self.f_width, 50))
        self.drop = nn.Dropout(p=self.p_drop)
        self.fc1 = nn.Linear(self.x_join_l, 2)

    def forward_once(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1,x.size()[1], x.size()[2])
        #output = F.max_pool2d(F.tanh(self.conv1(x)), (x.size()[2] - filter_width + 1, 1))
        output = F.max_pool2d(F.relu(self.conv1(x)), (x.size()[2] - self.f_width + 1, 1))

        return output

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # ahora los unimos
        out = torch.cat((output1, output2), 1)
        out = out.view(batch_size, 1, 200)
        #final_out = F.relu(self.fc1(out))
        final_out = self.fc1(out)

        final_out = self.drop(final_out)
        final_out =  F.softmax(final_out)
        return final_out

class Traductor:
    #ojo, es muy dificil cambiar de matriz a palabra, por lo tanto, esto hay que aplicarlo antes de pasar la frase por los word embeddings

    def __init__(self):
        self.alfabeto = {}
        with open("vocab.pickle", 'rb') as pickle_file:
            content = pickle.load(pickle_file, encoding='latin1')

            # ahora cambiamos el diccionario, esto es por mientras no mas
            for key in content:
                self.alfabeto[content[key]] = key

    def traducir(self, lista):
        frase = []
        for numero in lista:
            try:
                traducida = " " + self.alfabeto[int(numero)] + " "
            except KeyError:
                traducida = ""
            frase.append(traducida)
        return " ".join(frase)


