# Learning-to-rank-Pytorch-implementation-

Implementacion en Pytorch del paper "learning to rank short text pairs with convolutional deep neural networks"

## Dependencies
1. Python 3.X
2. Pytorch (compatible con la version de python que utilices)

## TO DO

- Overleap -> x_feat
- Crear y utilizar la matriz M -> x_sim
- Revisar problema del softmax (distribuciones)
- Funcion para calcular P@30
- Funcion para calcular MRR
- Revisar que este bien el calculo de MAP
- Utilizar TRAIN-ALL

## Para correrlo

Es necesario clonar el repo y correr training.py tal como esta, no es necesario cambiar de lugar los archivos. El repo del trabajo original es https://github.com/aseveryn/deep-qa.

Las clases que componen la red se encuentran en el archivo Clases, y algunas funciones necesarias estan en el archivo Functions.
