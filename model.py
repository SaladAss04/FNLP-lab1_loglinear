'''
Implementation of the log_linear model
'''
import math
import numpy as np
from tqdm import tqdm
'''
consts
'''
reg = 1e-2
lr = 1e-3
class model:
    def __init__(self, dims) -> None:
        self.d = dims
        self.w = np.random.rand(4, self.d)
        self.b = np.random.rand(4, 1)
        self.cache = []
    def softmax(self, bois):
        '''
        bois: n x 4
        return: softmax score: n x 4 , sum_of_exp: n
        '''
        out = []
        sum_logged = []
        for i in range(bois.shape[0]):
            try:
                exp_sum = sum([math.exp(y) for y in bois[i]])
                sum_logged.append(np.log(exp_sum))
                out.extend([math.exp(y) / exp_sum for y in bois[i]])
            except:
                print(bois[i])
        out = np.reshape(out, (bois.shape[0], 4))
        return out, sum_logged

    def forward(self, labels, inputs):
        '''
        labels: n x 1
        inputs: n x d
        returns: loss
        '''     
        n = labels.shape[0]
        scores = np.matmul(inputs, self.w.transpose()) + np.repeat(self.b.transpose(), n, axis = 0)# n x 4
        ml_scores, ml_sums_logged = self.softmax(scores)
        '''
        lg_scores = np.log(ml_scores)
        index_ll = [(i, labels[i]) for i in range(n)] 
        ll_1 = [scores[x[0], x[1]] for x in index_ll]
        ll = sum(ll_1)
        ll -= sum(ml_sums_logged)
        loss = - ll / n + reg * np.linalg.norm(self.w, ord = 2) / n
        '''
        index_ll = [(i, labels[i]) for i in range(n)] 
        loss = - sum([math.log(ml_scores[x[0], x[1]]) for x in index_ll]) + reg * np.linalg.norm(self.w, ord = 2)
        self.cache.append(inputs)
        self.cache.append(scores)
        self.cache.append(ml_scores)
        self.cache.append(labels)
        return loss
    def backward(self):
        '''
        self.cache
        self.co: 4 x d
        loss: 1 x 1
        updates self.co
        '''
        inputs, scores, ml_scores, label = tuple(self.cache)
        n, d = inputs.shape
        grad_lw = np.zeros((4, d))
        grad_lb = np.zeros((4,1))
        # empirical counts
        for i in range(n): 
            grad_lw[label[i], :] -= inputs[i] #modifying 1 x d
            #grad_lw += np.dot(np.reshape(ml_scores[i, :], (4,1)), np.reshape(inputs[i, :],(1, d))) #4x1 and 1xd . modifying 4 x d
            grad_lw += np.outer(ml_scores[i], inputs[i])
            # print(grad_lb.shape, ml_scores[i].shape)
            grad_lb += np.reshape(ml_scores[i], (4, 1))
            grad_lb[label[i]] -= 1
        '''
        expanded_label = np.zeros((n, 4))
        expanded_label[[(i, label[i]) for i in range(n)]] = 1 #label of n x 4
        grad_ls = - expanded_label / ml_scores #gimme a n x 4
        grad_lz = np.zeros((n, 4))
        for i in range(4):
            grad_sz = np.zeros((n, 4))
            grad_sz[:, i] = ml_scores[:, i] * (1 - ml_scores[:, i])
            for j in range(4):
                if j == i:
                    continue
                grad_sz[:, j] = - ml_scores[:, i] * ml_scores[:, j]
            grad_lz[:, i] = np.sum(grad_ls * grad_sz, axis = 1)
        grad_lw =  np.dot(grad_lz.transpose(), inputs)# gimme a 4 x d
        grad_lb =  grad_lz.sum(axis = 0)
        #print("ml_scores,", ml_scores[0])
        #print("grads", grad_lw, grad_lb)
        '''
        grad_lw += 2 * reg * self.w
        self.cache = []
        return grad_lw, grad_lb
    def optimize(self, grad_w, grad_b):
        self.w -= lr * grad_w
        self.b -= lr * grad_b
    def test(self, inputs):
        '''
        inputs: n x d
        return labels n x 1
        '''
        n = inputs.shape[0]
        scores = np.matmul(inputs, self.w.transpose()) + np.repeat(self.b.transpose(), n, axis = 0) # n x 4
        ml_scores, _ = self.softmax(scores)
        ret = [np.argmax(ml_scores[i]) for i in range(n)]
        return np.reshape(ret, (n, 1))