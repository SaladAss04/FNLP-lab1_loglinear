'''
Implementation of the pipeline of training and testing
'''
import os
import numpy as np
import utils as ut
import model as md
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

boy = md.model(ut.length)
epoch = 5
train_losses = []
train_acc = []

def train(model, regen):
    loader = ut.data_loader(16, ut.get_dir(0), 'saved_data', regen, True) #train loader
    for e in range(epoch):
        e_loss = 0
        loader.b_num = 0
        print("training, epoch: ", e)
        for batch in tqdm(loader, total=len(loader.data_raw) // loader.b):
            try:
                y, x = batch
            except:
                break
            # y = np.array(y).reshape((x.shape[0],))
            y = np.array([int(boi) - 1 for boi in y])
            loss = model.forward(y, x)
            # print("loss_this: ", loss)
            e_loss += loss
            grad_w , grad_b= model.backward()
            model.optimize(grad_w, grad_b)
        e_loss /= len(loader.data_raw) // loader.b
        print("epoch: ", e, " loss: ", e_loss)
        np.save("w.npy", model.w)
        t_acc = inference(model, 0, 0)
        train_losses.append(e_loss)
        train_acc.append(t_acc)
        

def inference(model, regen, where):
    model.w = np.load('w.npy')
    loader = ut.data_loader(32, ut.get_dir(where), 'saved_data', regen, False)
    print("testing")
    acc_cnt = 0
    for batch in tqdm(loader, total=len(loader.data_raw) // loader.b):
        try:
            y, x = batch
        except:
            break
        output = np.reshape(model.test(x), (-1,))
        y = np.array([int(boi) - 1 for boi in y])
        acc_vec = output == y
        acc_cnt += acc_vec.sum()
    acc = acc_cnt / len(loader.data_raw)
    print("test acc: ", acc, acc_cnt, len(loader.data_raw))
    return acc



'''
train_or_test, regen = sys.argv[1], int(sys.argv[2])
if train_or_test == '0':
    train(boy, regen)
else:
    inference(boy, regen)
'''
def draw(e_num, loss, acc):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, e_num + 1), loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('figs', 'loss.png'))
    plt.close()

    # 绘制准确率图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, e_num + 1), acc, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('figs', 'acc.png'))
    plt.close()
    

regen = int(sys.argv[1])
train(boy, regen)
inference(boy, regen, 1)
draw(epoch, train_losses, train_acc)
