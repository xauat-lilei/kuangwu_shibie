
import matplotlib.pyplot as plt
import numpy as np
def plot_history(epochs, Acc, Loss):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1,epochs+1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Valitate')
    plt.legend(['train', 'valitate'], loc='upper left')
    plt.savefig('Loss.png')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['val_accurate'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'valitate'], loc='upper left')
    plt.savefig('Acc.png')
    plt.show()
def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.savefig('jieguo.png')
    plt.imshow(img)
