import numpy as np
import torch.optim as optim
import math
import argparse
import sys

from torch import nn
from dataloader import *
from models import *
from helper import *
from trainAndtest import *
import torch.optim.lr_scheduler as lr_scheduler

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", help='For training, give name for saving')
    parser.add_argument("-v", help='For validation testing, give percentage of data wanted in validation, try using 0.25 but default is 0.05', nargs='?', const=0.05, type=float)
    parser.add_argument("-p", help='For prediction, give percentage of data split, try using 0.01 but default is 0.05', nargs='?',const=0.05, type=float)
    parser.add_argument("-m", help='For loading saved model, give the model name')
    parser.add_argument("-mt", help='Give the type of the model e.g. resnet152')
    parser.add_argument("-l", help='To load the best values from the file, try using t')

    return parser

if __name__== "__main__":

    # Handling the input arguments
    parser = parseArguments()
    args = parser.parse_args()

    # Put model type and model name to load
    if args.mt:
        model = get_from_models(args.mt, 17, not(args.m==None), args.m)
        model = model.cuda()
    else:
        print("Require model type, -h for help")
        sys.exit(0)

    print("================ Model Created ===============")

    # model = modelFreeze(model)

    # Created the loss function and optimizer
    loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias']},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'weight_decay': 1e-4}
    ], lr=1e-2, momentum=0.9, nesterov=True)

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.Adagrad(model.parameters(), lr=1e-2, weight_decay=1e-4)
    # optimizer = optim.Adadelta(model.parameters(), lr=1e-2, weight_decay=0)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005, nesterov=True)

    best_score, best_threshold = saveCreds(not(args.l == None))
    best_model = model

    if args.p == None and args.t == None and args.v == None:
        print("Usage test.py runtype, use -h for help")
        sys.exit(0)

    # Data loading and processing
    X_train, X_val, train_nc, X_test = DataProcessing(float(0.05))
    train_loader = DataLoader(X_train, batch_size=16, shuffle=True, num_workers=16)
    val_loader = DataLoader(X_val, batch_size=16, shuffle=True, num_workers=16)
    test_loader = DataLoader(X_test, batch_size=16, shuffle=True, num_workers=16)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

    if args.t or args.v:
        for epoch in range(0, 20):
            if args.t:
                train(model, epoch, train_loader, optimizer, loss_fn)
                torch.save(model.state_dict(),"./{}_train.pth".format(args.t))
            v_score, threshold, ff2 = validate(model, val_loader, loss_fn)

            with open("testsb.csv", "a") as f:
                f.write("{},{},{}\n".format(epoch, ff2, v_score))


            if v_score < best_score:
                best_model = model
                best_threshold = threshold
                best_score = v_score
                saveCreds(save = True, threshold = best_threshold, v_score = best_score)
                torch.save(model.state_dict(),"./{}_train.pth".format("bestest"))

            # scheduler.step()

            # X_train.set_transformation()
            # train_loader = DataLoader(X_train, batch_size=16, shuffle=True, num_workers=16)

            if epoch == 10:
                # print("Unfreezing")
                # model = modelUnFreeze(model)
                # optimizer = optim.SGD([
                #     {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias']},
                #     {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
                #      'weight_decay': 1e-4}
                # ], lr=1e-2, momentum=0.9, nesterov=True)
                TRANSFORMATIONS = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(180)
                ]
                
    if args.p:
        predict(test_loader, best_model, best_threshold, X_test, "Prediction")
