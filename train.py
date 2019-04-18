import helper
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

ap = argparse.ArgumentParser(description='Train.py')
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpointlearnrate1.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")

pa = ap.parse_args()
loc = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
model_arch = pa.arch
p_drop = pa.dropout
HL1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs
trainloader, vldtnloader, testloader = helper.load_data(loc)
model, optimizer, criterion = helper.nn_setup(model_arch,p_drop,HL1,lr,power)
helper.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)
helper.save_checkpoint(path,model_arch,HL1,p_drop,lr)
print("All Set and Done. The Model is trained") 