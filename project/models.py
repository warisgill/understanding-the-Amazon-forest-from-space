import torch
import math
from torch import nn
from torchvision import models
from basenet import *


def set_activation(model, af):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, af)
        else:
            set_activation(child, af)

# **************************** RESNET ARchitectures ************************************

def get_res18(num_classes, pt = None, path = None, pretrained=True):
    net = models.resnet18(pretrained=pretrained)

    net.fc = nn.Linear(net.fc.in_features, num_classes)
     # If we have the pretrained classifier
    if pt:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_res34(num_classes, pt = None, path = None, pretrained=True):
    net = models.resnet34(pretrained=pretrained)

    net.fc = nn.Linear(net.fc.in_features, num_classes)
     # If we have the pretrained classifier
    if pt:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_res50(num_classes, pt = None, path = None, pretrained=True):
    net = models.resnet50(pretrained=pretrained)
    # Plug our own classifier as the last layer
    # for x in net.parameters():
    #     x.requires_grad = False

    net.fc = nn.Linear(net.fc.in_features, num_classes)
     # If we have the pretrained classifier
    if pt:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_res101(num_classes, pt = None, path = None,pretrained=True):
    net = models.resnet18(pretrained=pretrained)

    net.fc = nn.Linear(net.fc.in_features, num_classes)
     # If we have the pretrained classifier
    if pt:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_res152(num_classes, pt = None, path = None, pretrained=True):
    net = models.resnet152(pretrained=pretrained)
    # Plug our own classifier as the last layer
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    # If we have the pretrained classifier
    if pt:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

# *********************** Freeze & UnFreeze***********************************************
def modelFreeze(model):

    for layer in model.parameters():
        layer.requires_grad = False
    
    model.fc.requires_grad = True

    return model

def modelUnFreeze(model):
    
    for layer in model.parameters():
        layer.requires_grad = True

    return model


def get_densenet161(num_classes, pretrained = None, path = None):
    net = models.densenet161(pretrained=True)

    # Plug our own classifier as the last layer
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)

    # If we have the pretrained classifier
    if pretrained:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_googlenet(num_classes, pretrained = None, path = None):
    net = models.googlenet()
    # Plug our own classifier as the last layer
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    # If we have the pretrained classifier
    if pretrained:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_vgg19(num_classes, pretrained = None, path = None):
    net = models.vgg19_bn()
    print(net.classifier[-1])
    # Plug our own classifier as the last layer
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    print(net.classifier)
    # If we have the pretrained classifier
    if pretrained:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

def get_basenet(num_classes, pretrained = None, path = None):
    # net = resnet(3, num_classes)
    net = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=17, alpha=1e-2)

    if pretrained:
        print("pretrained")
        net.load_state_dict(torch.load(path))

    return net

# Add more models over here
def get_from_models(name, num_classes, pretrained = False ,path = None, actual_pretrained=True):
    
    if name == "resnet18":
        print("Getting {}".format(name))
        model = get_res18(num_classes, pretrained, path, actual_pretrained)
    
    elif name == "resnet34":
        print("Getting {}".format(name))
        model = get_res34(num_classes, pretrained, path, actual_pretrained) 

    elif name == "resnet50":
        print("Getting {}".format(name))
        model = get_res50(num_classes, pretrained, path, actual_pretrained)

    elif name == "resnet101":
        print("Getting {}".format(name))
        model = get_res101(num_classes, pretrained, path, actual_pretrained)

    elif name == "resnet152":
        print("Getting {}".format(name))
        model = get_res152(num_classes, pretrained, path, actual_pretrained)

    elif name =="densenet161":
        print("Getting densenet161")
        model = get_densenet161(num_classes, pretrained, path)
    elif name =="googlenet":
        print("Getting googlenet")
        model = get_googlenet(num_classes, pretrained, path)
    elif name =="vgg19":
        print("Getting vgg19 with batch normalization")
        model = get_vgg19(num_classes, pretrained, path)
    elif name == "basenet":
        print("Getting basenet")
        model = get_basenet(num_classes, pretrained, path)

    # set_activation(model, nn.LeakyReLU(inplace=True))
    return model