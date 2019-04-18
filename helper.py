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

model_arch = {"vgg16":25088,
              "alexnet":9216,
              "densenet121":1024
             }

def load_data(loc  = "./flowers" ):
    data_dir = loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(180),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

    vldtn_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    vldtn_data = datasets.ImageFolder(valid_dir, transform = vldtn_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    vldtnloader = torch.utils.data.DataLoader(vldtn_data, batch_size = 64)

    return trainloader , vldtnloader, testloader


def nn_setup(m_arch='densenet121',p_drop=0.5, HL1 = 17500,lr = 0.00001,power=gpu):

    if m_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_ft = model.classifier[0].in_features
    elif m_arch == 'densenet161':
        model = models.densenet121(pretrained=True)
        in_ft = model.classifier.in_features
    elif m_arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        in_ft = model.classifier[1].in_features
    else:
        print("Please choose between vgg16, densenet121 or alexnet")
    
    for p in model.parameters():
        p.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(in_ft, HL1)),
                                                ("relu1",nn.ReLU()),
                                                ("dropout1", nn.Dropout(p_drop)),
                                                ("bn1", nn.BatchNorm1d(HL1)),
                                                ("out_layer", nn.Linear(HL1, 102)),
                                                ("softmax", nn.LogSoftmax(dim=1))]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lr )

        if torch.cuda.is_available() and power = 'gpu':
            model.cuda()

        return model, criterion, optimizer


def train_model(model, criterion, optimizer, epochs = 5, print_every = 40, loader =trainloader, power='gpu'):
    model.train()
    for q in keep_awake(range(1)):
        steps = 0
        for e in range(epochs):
            running_loss = 0
            for i, (inputs, labels) in enumerate(loader):
                steps += 1
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        vldtn_loss = 0
                        correct = 0
                        total = 0
                        for data in vldtnloader:
                            images, vlabels = data
                            images, vlabels = images.to('cuda'), vlabels.to('cuda')
                            vldtn_outputs = model(images)
                            _, predicted = torch.max(vldtn_outputs.data, 1)
                            total += vlabels.size(0)
                            correct += (predicted == vlabels).sum().item()
                            vldtn_loss += criterion(vldtn_outputs, vlabels).item()
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(vldtn_loss/len(vldtnloader)),
                        "Validation Accuracy: {:.3f}".format((correct/total)*100))
                    running_loss = 0
                    model.train()
        print("DONE, Model has been trained")


def save_checkpoint(path='checkpoint.pth',model_arch ='vgg16', HL1=17500,p_drop=0.5,lr=0.00001,epochs=5):
    model.class_to_idx = train_data.class_to_idx
    torch.save({'model_name' :'vgg16',
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'classifier': classifierHL1},
                'checkpointlearnrate1.pth')


def load_checkpoint(path='checkpointlearnrate1.pth'):
    checkpoint = torch.load(path)
    m_name = checkpoint["model_name"]
    m_sd = checkpoint["state_dict"]
    m_classifier = checkpoint["classifier"]
    model.classifier = m_classifier
    model.load_state_dict(m_sd)
    model.class_to_idx = checkpoint["class_to_idx"]
    return model


def process_image(image_path):
    p_img = Image.open(image)
   
    preproc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    t_img = preproc(p_img)
    
    return t_img

def predict(image_path, model, topk=5,power='gpu'):
    model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)