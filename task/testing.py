# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import psutil
import h5py
import pickle
import logging
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification
# Import custom modules
from dataset import CustomDataset
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import model_save_name

def testing(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start testing!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name)

    with h5py.File(os.path.join(save_path, 'test_processed.hdf5'), 'r') as f:
        test_src_input_ids = f.get('test_src_input_ids')[:]
        test_src_attention_mask = f.get('test_src_attention_mask')[:]
        test_trg_list = f.get('test_label')[:]

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        src_language = data_['src_language']
        num_labels = data_['num_labels']
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')

    model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
    model = model.to(device)

    # lode model
    total_model_dict = dict()

    for total_phase in ['train_original', 'train_bt', 'train_eda', 'train_ood']:
        total_model_dict[total_phase] = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
        save_file_name = os.path.join(args.model_save_path, args.data_name, total_phase.split('_')[-1])
        save_file_name += 'checkpoint.pth.tar'
        total_model_dict[total_phase].load_state_dict(torch.load(save_file_name)['model'])
        total_model_dict[total_phase].to(device)
        model.eval()
        write_log(logger, f'Loaded model from {save_file_name}')

    # 2) Dataloader setting
    test_dataset = CustomDataset(src_list=test_src_input_ids, src_att_list=test_src_attention_mask,
                                  trg_list=test_trg_list, src_max_len=args.src_max_len)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=args.test_batch_size, shuffle=False,
                                 pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(test_dataset)}, {len(test_dataloader)}")

    #===================================#
    #============Inference==============#
    #===================================#

    ground_truth_list = list()
    predicted_list = list()

    for total_phase in ['train_original', 'train_bt', 'train_eda', 'train_ood']:
        model = total_model_dict[total_phase]
        with torch.no_grad():
            for i, batch_iter in enumerate(tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Input setting
                src_sequence = batch_iter[0]
                src_att = batch_iter[1]
                trg_label = batch_iter[2]

                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)

                with torch.no_grad():
                    predicted = model(input_ids=src_sequence, attention_mask=src_att)['logits']

                predicted_list.extend(predicted.max(dim=1)[1].cpu().tolist())
                ground_truth_list.extend(trg_label.cpu().tolist())
                
        ground_truth_list = [int(x) for x in ground_truth_list]

        if args.data_name == 'korean_hate_speech':

            hate_data_path = os.path.join(args.data_path,'korean-hate-speech-detection')
            test_dat = pd.read_csv(os.path.join(hate_data_path, 'test.hate.no_label.csv'))
            test_dat['label'] = predicted_list
            test_dat.to_csv(f'./{total_phase}_submission.csv', index=False)

        else:
            test_acc = sum(np.array(ground_truth_list) == np.array(predicted_list)) / len(ground_truth_list)
            with open('./test_results.txt', 'a') as f:
                f.write(f'{total_phase}')
                f.write('\t')
                f.write(f'{test_acc}')
                f.write('\n')
            write_log(logger, f'Mode: {total_phase}')
            write_log(logger, f'Test Accuracy: {test_acc}')