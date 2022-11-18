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

    if args.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        tb_writer.add_text('args', str(args))

    write_log(logger, 'Start testing!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed.hdf5'

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'r') as f:
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

    if args.model_type == 'custom_transformer':
        model = Transformer(task=args.task,
                            src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                            pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                            d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                            dim_feedforward=args.dim_feedforward,
                            num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                            num_decoder_layer=args.num_decoder_layer,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                            trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                            variational=args.variational,
                            variational_mode_dict=variational_mode_dict,
                            parallel=args.parallel)
    elif args.model_type == 'bert':
        if src_language == 'kr':
            model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
    elif args.model_type == 'bart':
        if src_language == 'kr':
            model = BartForSequenceClassification.from_pretrained('cosmoquester/bart-ko-mini', num_labels=num_labels)
    model = model.to(device)

    # lode model
    save_file_name = model_save_name(args)
    model.load_state_dict(torch.load(save_file_name)['model'])
    model = model.eval()
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
        test_dat.to_csv('./submission.csv', index=False)

    else:
        test_acc = sum(np.array(ground_truth_list) == np.array(predicted_list)) / len(ground_truth_list)
        write_log(logger, f'Test Accuracy: {test_acc}')