# main.py
# python main.py --dataset cifar10 --num_users 10 --target_usr 9 --total_hnet_epoch 10000 --total_ray_epoch 1000 --total_epoch 1 --seed 3 --local_bs 512 --lr 0.01 --lr_prefer 0.01 --solver_type linear --sample_ray --n_hidden 3 --embedding_dim 5 --input_dim 20 --output_dim 2 --hidden_dim 100  --gpus 0

import os
import pickle
import argparse
import torch

from utils.utils_data import get_data
from utils.utils_func import construct_log, get_random_dir_name, setup_seed
from hyper_model.train import Training_all

class TestParser():
    def __init__(self):
        # federated arguments
        self.num_users = 4
        self.shard_per_user = 2
        self.target_usr = 1 # [0, 9]
        self.log_dir = ''
        self.iid = True #False

        # training arguments
        self.epochs_per_valid = 100
        self.total_hnet_epoch = 100  # [20, 10000 ]
        self.lr = 0.01 #0.05
        self.total_epoch = 1 # [2000, 1]
        self.seed = 1 # [0, 3]
        self.solver_type = 'epo' #[ epo, linear]
        self.momentum = 0.5
        self.trainN = 100
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
        self.dataset = 'civil' # ['adult', 'cifar10','synthetic']
        self.target_dir = 'civil0405'  # ['adult', 'cifar10']

        # model structure
        self.n_classes = 3
        self.entropy_weight = 0.0
        self.n_hidden = 1 #[3, 3]
        self.embedding_dim = 5
        self.input_dim = 1280
        self.output_dim = 3
        self.hidden_dim = 100
        self.spec_norm = False

        self.outputs_root = 'outputs'
        self.auto_deploy = True


parser = argparse.ArgumentParser()
# federated arguments
parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user (each user has the num of classes)")
parser.add_argument('--target_usr', type=int, default=0, help="target usr id")



# training arguments
parser.add_argument('--epochs_per_valid', type=int, default=50, help="rounds of valid")
parser.add_argument('--total_hnet_epoch', type=int, default=5, help="hnet update innner steps")
parser.add_argument('--total_ray_epoch', type=int, default=1, help="hnet update innner steps")
parser.add_argument('--total_epoch', type=int, default=2000, help="hnet update innner steps")
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--local_bs', type=int, default=512, help="local batch size: B")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--lr_prefer', type=float, default=0.01, help="learning rate for preference vector")
parser.add_argument('--alpha', type=float, default=0.2, help="alpha for sampling the preference vector")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--num_workers', type=int, default=0, help="the number of workers for the dataloader.")
parser.add_argument('--eps_prefer', type=float, default=0.1, help="learning rate for preference vector")
parser.add_argument('--sigma', type=float, default=0.1, help="learning rate for preference vector")
parser.add_argument('--std', type=float, default=0.1, help="learning rate for preference vector")
parser.add_argument('--trainN', type=int, default=2000, help="the number of generated train samples .")
parser.add_argument('--testN', type=int, default=1000, help="the number of generated test samples.")
parser.add_argument('--solver_type', type=str, default="epo", help="the type of solving the model")
parser.add_argument('--sample_ray', action='store_true', help='whether sampling alpha for learning Pareto Front')
parser.add_argument('--train_baseline', action='store_true', help='whether train baseline for eicu dataset')
parser.add_argument('--baseline_type', type=str, default="fedave", help="the type of training baseline (fedave, local)")



# model structure
parser.add_argument('--n_classes', type=int, default=10, help="the number of classes.")
parser.add_argument('--entropy_weight', type=float, default=0.0, help="the number of classes.")
parser.add_argument('--n_hidden', type=int, default=2, help="hidden layer for the hypernet.")
parser.add_argument('--embedding_dim', type=int, default=5, help="embedding dim for eicu embedding the categorical features")
parser.add_argument('--input_dim', type=int, default=20, help="input dim (generate dim) for the hypernet.")
parser.add_argument('--output_dim', type=int, default=2, help="hidden layer for the hypernet.")
parser.add_argument('--hidden_dim', type=int, default=100, help="hidden dim for the hypernet.")
parser.add_argument('--spec_norm', action='store_true', help='whether using spectral norm not')


# learning setup arguments
parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--auto_deploy', action='store_true', help='whether auto deploy not')
# devices
parser.add_argument('--gpus', type=str, default="1", help='gpus for training')
# dataset/log/outputs/ dir
parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
parser.add_argument('--data_root', type=str, default='data', help="name of dataset")
parser.add_argument('--outputs_root', type=str, default='outputs', help="name of dataset")
parser.add_argument('--target_dir', type=str, default='', help=" dir name of for saving all generating data")
args = parser.parse_args()
args = TestParser()


if __name__ == '__main__':

    if  args.target_dir == "":
        args.log_dir = os.path.join(args.outputs_root, get_random_dir_name())
    else:
        args.log_dir = os.path.join(args.outputs_root, args.target_dir)
    setup_seed(seed = args.seed)
    # prepare for learning
    initial_device = torch.device('cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() and args.gpus != -1 else 'cpu')

    args.hnet_model_dir = os.path.join(args.log_dir, "hnet_model_saved")
    args.local_hnet_model_dir = os.path.join(args.log_dir, "local_hnet_model_saved")
    args.local_tnet_model_dir = os.path.join(args.log_dir, "local_tnet_model_saved")
    args.eps_prefer = 1.0/(3*args.num_users)
    logger = construct_log(args)


    if args.dataset == "adult":
        args.input_dim = 99
        args.output_dim  = 2
        args.num_users = 2
        args.local_bs = -1
    elif args.dataset == "synthetic":
        args.output_dim = 1
        args.num_users = 6
        args.local_bs = -1
    elif args.dataset == "cifar10":
        args.local_bs = 512
        args.num_users = 3
    elif args.dataset == "civil":
        args.input_dim = 384
        args.output_dim  = 3
        args.num_users = 4
        args.local_bs = -1

 
    if args.train_baseline and args.baseline_type == "local":
        users_used = [args.target_usr]
    else:
        users_used = [i for i in range(args.num_users)]

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    # print("*"*100)
    # print(dict_users_train)
    # print("*"*100)
    # print(dict_users_train[0])
    # print("*" * 20)
    # print(type(dict_users_train), type(dict_users_train[0]), type(dict_users_train[0][0]), type(dict_users_train[0][1]))
    # print("*" * 20)
    # print(len(dict_users_train))
    model = Training_all(args, logger, dataset_train, dataset_test, dict_users_train, dict_users_test, users_used = users_used)

    if args.auto_deploy:
        try:
            model.train()
            with open(os.path.join(args.log_dir, "pickle.pkl"), "wb") as f:
                pickle.dump(model.pickle_record, f)
            os.makedirs( os.path.join(args.log_dir, "done"), exist_ok = True)
        except Exception as e:
            logger.info("error info: {}.".format(e))
    else:
        model.train()
        with open(os.path.join(args.log_dir, "pickle.pkl"), "wb") as f:
            pickle.dump(model.pickle_record, f)
        os.makedirs( os.path.join(args.log_dir, "done"), exist_ok = True)       

