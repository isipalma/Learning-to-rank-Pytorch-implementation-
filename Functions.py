
import math
from collections import defaultdict


def batch_gen(X, batch_size):
    n_batches = X.shape[0] / float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0] / float(batch_size)) * batch_size
    n = 0
    for i in range(0, n_batches):
        if i < n_batches - 1:
            if len(X.shape) > 1:
                batch = X[i * batch_size:(i + 1) * batch_size, :]
                yield batch
            else:
                batch = X[i * batch_size:(i + 1) * batch_size]
                yield batch

        else:
            if len(X.shape) > 1:
                batch = X[end:, :]
                n += X[end:, :].shape[0]
                yield batch
            else:
                batch = X[end:]
                n += X[end:].shape[0]
                yield batch


def score(predictions, labels):
    # de las probabilidades toma el mas grande y lo compara con su respectivo label
    corrects = 0
    total = 0
    for prediction, label in zip(predictions, labels):
        #pred = prediction.data.numpy()
        if prediction.data[0][0] > prediction.data[0][1]:
            result = 0
        else:
            result = 1
        #print("resultado: {}".format(result))
        if result == int(label):
            corrects += 1
        total += 1

    return (corrects / total) * 100


def map_score(q_ids, predictions, labels):
   # por cada pregunta le calculamos su AP
    # primero los ponemos en un diccionario
    qid2cand = defaultdict(list)
    for qid, pred, label in zip(q_ids, predictions, labels):
        if pred.data[0][0] > pred.data[0][1]:
            result = 0
        else:
            result = 1
        if result == int(label):
            score = 1
        else:
            score = 0
        qid2cand[qid].append(score)

    average_precs = []
    for q_id, results in qid2cand.items():
        average_prec = 0
        running_correct_count = 0
        #for i, score in enumerate(sorted(results, reverse=True), 1):
        for i, score in enumerate(results, 1):
            if score == 1:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score



def MRR_score(qids, preds, labels):
    pass

#y_true = np.array([0, 0, 1, 1])
#y_scores = np.array([0.1, 0.4, 0.35, 0.8])
#print(average_precision_score(y_true, y_scores))
#print()