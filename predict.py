import helper
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

ap = argparse.ArgumentParser(
    description='predict.py')
ap.add_argument('input_image', default='paind-project/flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/aipnd-project/checkpointlearnrate1.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
img_path = pa.input_img
num_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

trainloader, vldtnloader, testloader = helper.load_data()
helper.load_checkpoint(path)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
p = helper.predict(img_path, model, num_outputs, power)
labels = [cat_to_name[str(index + 1)] for index in np.array(p[1][0])]
probability = np.array(p[0][0])
i=0
while i < num_outputs:
    print("{} has a probability of {}".format(labels[i], probability[i]))
    i += 1
print("finished")