# code for data prepare
import os
import json
import numpy as np
from torchvision import datasets, transforms
from utils.utils_sampling import iid, noniid



trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
class TestParser():
    def __init__(self):
        # federated arguments
        self.num_users = 4
        self.shard_per_user = 2
        self.target_usr = 4 # [0, 9]
        self.log_dir = ''

        # training arguments
        self.epochs_per_valid = 100
        self.total_hnet_epoch = 1000  # [20, 10000 ]
        self.lr = 0.01 #0.05
        self.total_epoch = 1 # [2000, 1]
        self.seed = 1 # [0, 3]
        self.solver_type = 'linear' #epo
        self.momentum = 0.5
        self.trainN = 2000
        self.num_workers = 0
        self.train_baseline = False
        self.local_bs = 512
        self.sample_ray = True
        self.epochs_per_valid = 1
        self.total_ray_epoch = 200  # [1, 1000 ]
        self.lr_prefer = 0.01
        self.alpha = 0.2
        self.eps_prefer = 0.1
        self.std = 0.1
        self.sigma = 0.1

        # learning setup arguments
        self.gpus = 0
        self.data_root = 'data/'
        self.dataset = 'cifar10' # ['adult', 'cifar10','synthetic']
        self.target_dir = 'synthetic'  # ['adult', 'cifar10']

        # model structure
        self.n_classes = 10
        self.entropy_weight = 0.0
        self.n_hidden = 1 #[3, 3]
        self.embedding_dim = 5
        self.input_dim = 20
        self.output_dim = 2
        self.hidden_dim = 100
        self.spec_norm = False

        self.outputs_root = 'outputs'
        self.auto_deploy = True

args = TestParser()
import torch
from torch.utils.data import Dataset, DataLoader
class dataset_prediction(Dataset):
    '''
    将传入的数据集，转成Dataset类，方面后续转入Dataloader类
    注意定义时传入的data_features,data_target必须为numpy数组
    '''

    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.targets = torch.from_numpy(data_target)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

def simple_data(args):

    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dict_users_test = {}

    if args.dataset == "adult":
        train_data_dir = os.path.join(args.data_root, "adult/train/mytrain.json")
        test_data_dir = os.path.join(args.data_root, "adult/test/mytest.json")

        with open(train_data_dir, "r") as f:
            train_data = json.load(f)["user_data"]
        dataset_train = [(np.array(x).astype(np.float32),float(y)) for x, y in zip(train_data["phd"]["x"], train_data["phd"]["y"])] + \
                        [(np.array(x).astype(np.float32),float(y)) for x, y in zip(train_data["non-phd"]["x"], train_data["non-phd"]["y"])]
        dict_users_train[0] = [i for i in range(len(train_data["phd"]["y"]))] 
        dict_users_train[1] = [i+len(dict_users_train[0]) for i in range(len(train_data["non-phd"]["y"])) ]

        with open(test_data_dir, "r") as f:
            test_data = json.load(f)["user_data"]
        dataset_test = [(np.array(x).astype(np.float32),float(y)) for x, y in zip(test_data["phd"]["x"], test_data["phd"]["y"])] + \
                        [(np.array(x).astype(np.float32),float(y)) for x, y in zip(test_data["non-phd"]["x"], test_data["non-phd"]["y"])]
        dict_users_test[0] = [i for i in range(len(test_data["phd"]["y"]))] 
        dict_users_test[1] = [i+len(dict_users_test[0]) for i in range(len(test_data["non-phd"]["y"])) ]

        return dataset_train, dataset_test, dict_users_train, dict_users_test       


    elif args.dataset == "synthetic": # generate dataset  dataset_train = [[x, y], [x, y],...,[x, y]]
                                      # dict_user_train is a dict {5:[1,2,3,...,10]}
        args.num_users = 6
        args.testN = int(0.5*args.trainN)
        v = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std**2 * np.eye(args.input_dim)

        for usr in [0,1,2,3,4,5]:
            tmp_trainN =  args.trainN
            tmp_testN =  args.testN 
            # if usr == 2:
            #     tmp_trainN = 10 * args.trainN
            #     tmp_testN = 10 * args.testN 
            # else:
            #     tmp_trainN = args.trainN
            #     tmp_testN = args.testN 
            r_0 = np.random.multivariate_normal(mean, cov)
            u_m = v + r_0

            # if usr !=3:
 
            x_m = np.random.uniform(-1.0, 1.0, (tmp_trainN+tmp_testN, args.input_dim))
            if usr in [0,1,2]:
                y_m = np.dot(x_m, u_m) + np.random.normal(0, args.sigma**2, (tmp_trainN+tmp_testN,))
            elif usr in [3,4,5]: 
                y_m = -np.dot(x_m, u_m) + np.random.normal(0, args.sigma**2, (tmp_trainN+tmp_testN,))
            else:
                print("error usr in generating synthetic data.")

            dataset_train.extend([ (x.astype(np.float32),y) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN]) ])
            dataset_test.extend([ (x.astype(np.float32),y) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:]) ])

            try:
                dict_users_train[usr] = [i+ dict_users_train[usr-1][-1]+1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ dict_users_test[usr-1][-1]+1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i+ 0  for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ 0  for i in range(tmp_testN)]              

        return dataset_train, dataset_test, dict_users_train, dict_users_test

def civil_data(args):
    data_dir = '../DecentralizedAdaptiveLearning/data/task3/'
    # train_y = np.loadtxt(data_dir+'joy_labels_for_danger_detection.csv', delimiter=",", dtype=str).astype(int)#[ids]
    train_y = np.load(data_dir+'test_action_labels.npy')
    dataset_train = np.load(data_dir+'train_action_imgs.npy')
    test_y = np.load(data_dir+'test_action_labels.npy')
    dataset_test = np.load(data_dir+'test_action_imgs.npy')
    print("*" * 100)

    dataset_train = dataset_prediction(data_features=dataset_train, data_target=train_y)
    dataset_test = dataset_prediction(data_features=dataset_test, data_target=test_y)


    # sample users
    if not args.iid:
        dict_users_train = iid(dataset_train, args.num_users)
        dict_users_test = iid(dataset_test, args.num_users)
    else:
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user,
                                               rand_set_all=rand_set_all)
    return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_root, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.data_root, train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)


    elif args.dataset == "adult" or args.dataset == "synthetic":
        
        dataset_train, dataset_test, dict_users_train, dict_users_test = simple_data(args)

    elif args.dataset == "eicu":
        dataset_train, dataset_test, dict_users_train, dict_users_test = eicu_data(args)

    elif args.dataset == "danger_detection":
        dataset_train, dataset_test, dict_users_train, dict_users_test = civil_data(args)
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users_train, dict_users_test




# -*- coding: utf-8 -*-
# File              : danger_detection/images_processing.py
# Author            : Joy
# Create Date       : 2022/12/16
# Last Modified Date: 2022/12/16
# Last Modified By  : Joy
# Reference         : pip install -U -r requirements0706.txt --default-timeout=1000
# Description       : 1. swim_transformer
# ******************************************************
import os
import numpy as np


def read_images(img_dir='./data/danger_detection/images/',
                img_embedding_path='./data/danger_detection/image_embedding_for_danger_detection',
                preprocess_modeling='MoblieNet'):
    x = []

    for f in sorted(os.listdir(img_dir), key=lambda x: int(x.split('(')[1].split(')')[0])):
        if not os.path.exists(img_dir + f):
            continue
        image = cv2.imread(img_dir + f, -1)
        print(f, ": ", image.shape)
        x.append(cv2.resize(image, (224, 224)))

    if preprocess_modeling == 'MobileViT':
        vit = MobileVideoVit(size=256, num_classes=2)
        image_embedding = vit(np.array(x) / 255.0)
    print("Finished : ", image_embedding.shape)
    np.save(img_embedding_path, np.array(image_embedding))
    return image_embedding


if __name__ == '__main__':
    image_embedding = read_images()
