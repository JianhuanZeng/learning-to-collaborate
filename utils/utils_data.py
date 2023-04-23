# code for data prepare
import os
import torch
import json
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from utils.utils_sampling import iid, noniid
from torch.utils.data import Dataset, DataLoader



trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

class torch_like_dataset(Dataset):
    '''
    将传入的数据集，转成Dataset类，方面后续转入Dataloader类
    注意定义时传入的data_features,data_target必须为numpy数组
    '''

    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.data = torch.from_numpy(data_features)
        self.targets = torch.from_numpy(data_target)
        self.classes = ['', '', '']
        self.class_to_idx = {'':0, '':1, '':2}
        self.meta = {'filename': 'batches.meta', 'key': 'label_names', 'md5': '5ff9c542aee3614f3951f8cda6e48888'}

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
        args.num_users = 4
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

def civil_data(n_classes, data_dir = '../DecentralizedAdaptiveLearning/data/task3/'):
    # train_y = np.loadtxt(data_dir+'joy_labels_for_danger_detection.csv', delimiter=",", dtype=str).astype(int)#[ids]
    # data_dir = '../DecentralizedAdaptiveLearning/data/task3/'
    train_x = np.load(data_dir+'train_action_imgs_vit.npy')
    train_y = np.load(data_dir+'train_action_labels.npy') # 1065
    test_x = np.load(data_dir+'test_action_imgs_vit.npy')
    test_y = np.load(data_dir+'test_action_labels.npy')
    print("train: ", train_x.shape, "class distr: ", {i: (train_y==i).mean() for i in range(n_classes)})
    print("test : ", test_x.shape, "class distr: ", {i: (test_y == i).mean() for i in range(n_classes)})
    print("*" * 100)

    dataset_train = [(train_x[i], train_y[i]) for i in range(len(train_y))]
    dataset_test = [(test_x[i], test_y[i]) for i in range(len(test_y))]
    # dataset_train = torch_like_dataset(data_features=dataset_train, data_target=train_y)
    # dataset_test = torch_like_dataset(data_features=dataset_test, data_target=test_y)
    return dataset_train, dataset_test

def civil_noniid(dataset,args):
    dict_users = {}
    if args.iid:
        ids = np.random.randint(0, len(dataset), len(dataset))
    else:
        ids = list(range(len(dataset)))
    num_samples_per_usr = int(len(ids) / args.num_users)
    for usr in range(args.num_users-1):
        dict_users[usr] = ids[usr * num_samples_per_usr:(usr + 1) * num_samples_per_usr]
    dict_users[args.num_users-1] = ids[(usr + 1) * num_samples_per_usr:]
    return dict_users

def get_data(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
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
        dict_users_train = iid(dataset_train, args.num_users)
        dict_users_test = iid(dataset_test, args.num_users)

    elif args.dataset == "adult" or args.dataset == "synthetic":
        
        dataset_train, dataset_test, dict_users_train, dict_users_test = simple_data(args)

    elif args.dataset == "civil":
        dataset_train, dataset_test = civil_data(args.n_classes)
        dict_users_train = civil_noniid(dataset_train, args)
        dict_users_test = civil_noniid(dataset_test, args)

    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users_train, dict_users_test
