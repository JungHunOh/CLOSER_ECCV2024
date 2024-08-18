import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('--save', type=str, default=None)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_soft', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=12)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('--eval_only', action='store_true', default=False)

    parser.add_argument('--support_shot', type=int, default=5)
    parser.add_argument('--support_way', type=int, default=5)
    parser.add_argument('--query_base_shot', type=int, default=10)
    parser.add_argument('--query_base_way', type=int, default=20)
    parser.add_argument('--query_new_shot', type=int, default=10)
    parser.add_argument('--lr_fc', type=float, default=0.1)

    parser.add_argument('--t_depths', type=int, default=5)
    parser.add_argument('--t_heads', type=int, default=1)

    parser.add_argument('--plot', action='store_true', default=False)

    parser.add_argument('--all_lambda', type=float, default=1)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--lambda_base', type=float, default=1)
    parser.add_argument('--lambda_new', type=float, default=1)
    parser.add_argument('--train_backbone', action='store_true', default=False)
    parser.add_argument('--channel_shuffle', action='store_true', default=False)

    parser.add_argument('--base_K', type=int, default=10000)
    parser.add_argument('--new_K', type=int, default=100)
    parser.add_argument('--linkage_mode', type=str, default='single', choices=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'])
    parser.add_argument('--cluster_mode', type=str, default='maxclust', choices=['maxclust', 'inconsistent', 'distance', 'monocrit', 'maxclust_monocrit'])

    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--score', action='store_true', default=False)
    parser.add_argument('--warmup', type=int, default=-1)
    parser.add_argument('--split', type=int, default=1)

    parser.add_argument('--train_way', type=int, default=20)
    parser.add_argument('--train_query_shot', type=int, default=5)
    parser.add_argument('--train_shot', type=int, default=5)
    parser.add_argument('--mp', action='store_true', default=False)
    parser.add_argument('--iter_per_epoch', type=int, default=50)

    parser.add_argument('--num_inds', type=int, default=32)
    parser.add_argument('--dim_hidden', type=int, default=512)
    parser.add_argument('--mode', type=int, default=-1)

    parser.add_argument('--num_aug', type=int, default=5)
    parser.add_argument('--supcon_temp', type=float, default=0.1)
    parser.add_argument('--ce_temp', type=float, default=16)
    parser.add_argument('--ce1_temp', type=float, default=16)
    parser.add_argument('--fc_dim', type=int, default=4096)
    parser.add_argument('--cos_lamb', type=float, default=1)
    parser.add_argument('--ssl_lamb', type=float, default=0.1)
    parser.add_argument('--ce_lamb', type=float, default=1)

    parser.add_argument('--closer', action='store_true', default=False)

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    #pprint(vars(args))
    args.num_gpu = set_gpu(args)

    if args.eval_only:
        args.model_dir = f'./checkpoint/{args.dataset}/base/{args.save}/session0_max_acc.pth'
        args.epochs_base = 1

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()


