# Import modules
import time
import argparse
# Import custom modules
from task.preprocessing import preprocessing
from task.training import training
from task.testing import testing
# Utils
from utils import str2bool, path_check, set_random_seed

def main(args):

    # Time setting
    total_start_time = time.time()

    # Path setting
    path_check(args)

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)

    if args.testing:
        testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--multiprocessing', action='store_true')
    # Path setting
    parser.add_argument('--preprocess_path', default='./preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/mnt/storage1/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--data_name', default='nsmc', type=str,
                        help='Data name; Default is nsmc')
    parser.add_argument('--cnn_dailymail_ver', default='3.0.0', type=str,
                        help='; Default is 3.0.0')
    parser.add_argument('--model_save_path', default='/mnt/storage1/kyohoon1/model_checkpoint/ood', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default='/mnt/storage1/kyohoon1/results/ood', type=str,
                        help='Results file path')
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='Ralamb', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--min_len', default=4, type=int, 
                        help="Sentences's minimum length; Default is 4")
    parser.add_argument('--src_max_len', default=200, type=int, 
                        help="Sentences's minimum length; Default is 200")
    parser.add_argument('--num_epochs', default=5, type=int, 
                        help='Training epochs; Default is 5')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int,    
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-4')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--label_smoothing_eps', default=0.05, type=float,
                        help='')
    # Testing setting
    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')
    parser.add_argument('--beam_size', default=5, type=int, 
                        help='Beam search size; Default is 5')
    parser.add_argument('--beam_alpha', default=0.7, type=float, 
                        help='Beam search length normalization; Default is 0.7')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, 
                        help='Beam search repetition penalty term; Default is 1.3')
    # Seed & Logging setting
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed; Default is 42')
    parser.add_argument('--print_freq', default=300, type=int, 
                        help='Print training process frequency; Default is 300')
    args = parser.parse_args()

    main(args)