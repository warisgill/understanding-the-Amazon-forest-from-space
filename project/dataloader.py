import numpy as np
import torch
import pandas as pd
import math

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataset import Dataset
import random
from random import randint

RESIZE_VAL = 224

TRANSFORMATIONS = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(RESIZE_VAL),
        transforms.RandomRotation(180)
]

# Built on top of the base code provided by the 
# Kaggle
class KaggleAmazonDataset(Dataset):

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        # Extracts the data and the images
        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def name(self):
        return self.X_train

    def __len__(self):
        return len(self.X_train.index)

    def splits(self, valx, valy):
        self.X_train = pd.Series(self.X_train.tolist()[valx:valy])
        self.y_train = self.y_train[valx:valy]

    def getLabelEncoder(self):
        return self.mlb

    def numClasses(self):
        return self.y_train.shape[1]

    def classesName(self):
        return self.mlb.inverse_transform(np.array([[1] * 17]))
    ## newly added
    def set_transformation(self):
        num_of_transf = randint(1,len(TRANSFORMATIONS))
        rand_transf = random.sample(TRANSFORMATIONS, k=num_of_transf)
        rand_transf.extend([ transforms.ToTensor(),
            transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])])
        self.transform = transforms.Compose(rand_transf)
        

IMG_PATH = '../train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'train_v2.csv'

TEST_DATA = 'test_v2_file_mapping.csv'
TEST_PATH = '../test-jpg/'

def DataProcessing(per):
    ############### TO DO #####################
    #
    # Try differnt transformations
    #
    ###########################################

    # Run once to calculate the mean and the std
    # cal_mean_and_std(train_loader)

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(RESIZE_VAL),
        # transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(RESIZE_VAL),
        transforms.ToTensor(),
        transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
    ])

    print("=============== Loading Data ===============")

    ###########################################
    #
    # Load the data from Kaggle and split into
    # train, validation and test sets
    #
    ###########################################

    X_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transform)
    X_val = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transform_val)
    X_test = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transform_val)
    
    x = int(math.floor((2*per) * len(X_train)))
    
    X_train.splits(0, len(X_train) - x)
    X_val.splits(len(X_val) - x, len(X_val) - (x/2))
    X_test.splits(len(X_test) - (x/2), len(X_test))
    
    # train_loader = DataLoader(X_train, batch_size=16, shuffle=True, num_workers=16)
    # val_loader = DataLoader(X_val, batch_size=16, shuffle=True, num_workers=16)
    # test_loader = DataLoader(X_test, batch_size=16, shuffle=True, num_workers=16)

    return X_train, X_val, X_train.numClasses(), X_test


def TestDataProcessing(per):
    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
    ])

    print("=============== Loading Training Data ===============")

    X_test = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transform_test)

    X_test.splits(len(X_test) - int(math.floor(per * len(X_test))), len(X_test))

    test_loader = DataLoader(X_test, batch_size=16, shuffle=True, num_workers=16)

    return X_test, test_loader, X_test.numClasses()

def saveCreds(load = None, save = None, threshold = None, v_score = None):
    if save:
        with open("bestval.txt", "w") as f:
            th = [str(x) for x in threshold]
            f.write("val_loss {}\n".format(v_score))
            f.write("threshold {}".format(",".join(th)))

        return True

    if load:
        with open("bestval.txt", "r") as f:
            lines = f.readlines()
            b_val = lines[0].split()[1]
            th = lines[1].split()[1].split(",")
            th = [float(x) for x in th]
        return b_val, th
    else:
        return 9999999, [0.2]*17
    

# val_loss 0.6645
# threshold 0.13759634,0.13380216,0.21532345,0.20997113,0.87679719,0.25989875,0.0570383,0.87954109,0.11004215,0.12596493,0.19495283,0.09373639,0.32412975,0.09473451,0.34160667,0.41929216,0.05962171