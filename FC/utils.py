import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import os 
import time
import sys 

def plot_dataset( dataset, number_of_samples, classes = []):
    
    assert len(dataset) >=number_of_samples
    
    # settings
    nrows, ncols = int(np.ceil(number_of_samples/3)), 3  # array of sub-plots
    figsize = [8, 8]     # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        sample,label = dataset[i-1]
        sample = sample.permute(1,2,0)
        if len(classes)>0:
            label = classes[label]
        axi.imshow(sample.squeeze(), cmap='gray', vmin=0, vmax=255)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(label)

    plt.tight_layout(True)
    plt.show()


def plot_tensors_as_images( tensors, number_of_samples):
    
    assert len(tensors) >=number_of_samples
    
    # settings
    nrows, ncols = int(np.ceil(number_of_samples/3)), 3  # array of sub-plots
    figsize = [8, 8]     # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        sample = tensors[i]
        sample = sample.permute(1,2,0)
        axi.imshow(sample, cmap='gray', vmin=0, vmax=255)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols


    plt.tight_layout(True)
    plt.show()



class Mode(Enum):
    training = 1
    validation = 2
    test = 3
    

try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0
    
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_graphs(train_accuracies, val_accuracies, train_losses, val_losses, path_to_save = ''):
    plot_accuracy(train_accuracies, val_accuracies,  path_to_save =  path_to_save )
    plot_loss(train_losses,val_losses, path_to_save =  path_to_save)
    return max(val_accuracies)



def plot_accuracy(train_accuracies, val_accuracies, to_show = True, label='accuracy', path_to_save = ''):
    
    print(f'Best val accuracy was {max(val_accuracies)}, at epoch {np.argmax(val_accuracies)}')
    train_len = len(np.array(train_accuracies))
    val_len = len(np.array(val_accuracies))
    
    xs_train = list(range(0,train_len))
    
    if train_len!= val_len:
        xs_val = list(range(0,train_len,int(train_len/val_len)+ 1))
    else:
        xs_val = list(range(0,train_len))
    
    plt.plot(xs_val, np.array(val_accuracies), label='val '+ label)
    plt.plot(xs_train, np.array(train_accuracies), label='train '+ label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if len(path_to_save)>0:
      plt.savefig(f'{path_to_save}/accuracy_graph.png')
      
    if to_show:
        plt.show()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def plot_loss(train_losses,val_losses,to_show = True,val_label='val loss',train_label='train loss', 
              path_to_save=''):
    
    train_len = len(np.array(train_losses))
    val_len = len(np.array(val_losses))
    
    xs_train = list(range(0,train_len))
    if train_len!= val_len:
        xs_val = list(range(0,train_len,int(train_len/val_len)+ 1))
    else:
        xs_val = list(range(0,train_len))
        
    
    plt.plot(xs_val, np.array(val_losses), label=val_label)
    plt.plot(xs_train, np.array(train_losses), label=train_label)
    
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if len(path_to_save)>0:
      plt.savefig(f'{path_to_save}/loss_graph.png')
    if to_show:
        plt.show()

