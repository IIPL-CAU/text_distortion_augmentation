# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import psutil
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

def training(args):
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

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name)
    save_name = f'processed.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        valid_trg_list = f.get('valid_label')[:]
        valid_trg_list = F.one_hot(torch.tensor(valid_trg_list, dtype=torch.long)).numpy()

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        src_language = data_['src_language']
        num_labels = data_['num_labels']
        del data_

    with h5py.File(os.path.join(save_path, 'aug_' + save_name), 'r') as f:
        train_bt_src_input_ids = f.get('train_bt_src_input_ids')[:]
        train_bt_src_attention_mask = f.get('train_bt_src_attention_mask')[:]
        train_bt_trg_list = f.get('train_bt_label')[:]
        train_bt_trg_list = F.one_hot(torch.tensor(train_bt_trg_list, dtype=torch.long)).numpy()
        train_eda_src_input_ids = f.get('train_eda_src_input_ids')[:]
        train_eda_src_attention_mask = f.get('train_eda_src_attention_mask')[:]
        train_eda_trg_list = f.get('train_eda_label')[:]
        train_eda_trg_list = F.one_hot(torch.tensor(train_eda_trg_list, dtype=torch.long)).numpy()
        train_ood_src_input_ids = f.get('train_ood_src_input_ids')[:]
        train_ood_src_attention_mask = f.get('train_ood_src_attention_mask')[:]
        train_ood_trg_list = torch.full((len(train_trg_list), num_labels), 1 / num_labels).numpy()

        # train_src_input_ids = np.append(train_src_input_ids, aug_input_ids, axis=0)
        # train_src_attention_mask = np.append(train_src_attention_mask, aug_attention_mask, axis=0)
        # train_trg_list = np.append(train_trg_list, aug_label, axis=0)

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')

    total_model_dict = dict()
    for total_phase in ['train_original', 'train_bt', 'train_eda', 'train_ood']:
        total_model_dict[total_phase] = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=num_labels)
        total_model_dict[total_phase].to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train_original': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask, 
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'train_bt': CustomDataset(src_list=np.append(train_src_input_ids, train_bt_src_input_ids, axis=0), 
                                  src_att_list=np.append(train_src_attention_mask, train_bt_src_attention_mask, axis=0), 
                                  trg_list=train_trg_list, src_max_len=args.src_max_len),
        'train_eda': CustomDataset(src_list=np.append(train_src_input_ids, train_eda_src_input_ids, axis=0), 
                                   src_att_list=np.append(train_src_attention_mask, train_eda_src_attention_mask, axis=0), 
                                   trg_list=train_trg_list, src_max_len=args.src_max_len),
        'train_ood': CustomDataset(src_list=np.append(train_src_input_ids, train_ood_src_input_ids, axis=0), 
                                   src_att_list=np.append(train_src_attention_mask, train_ood_src_attention_mask, axis=0), 
                                   trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train_original': DataLoader(dataset_dict['train_original'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'train_bt': DataLoader(dataset_dict['train_bt'], drop_last=True,
                               batch_size=args.batch_size, shuffle=True, pin_memory=True,
                               num_workers=args.num_workers),
        'train_eda': DataLoader(dataset_dict['train_eda'], drop_last=True,
                                batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                num_workers=args.num_workers),
        'train_ood': DataLoader(dataset_dict['train_ood'], drop_last=True,
                                batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train_original'])}, {len(dataloader_dict['train_original'])}")
    
    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(total_model_dict['train_original'], args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
        save_file_name = os.path.join(save_path, 
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')

    with open('./results.txt', 'w') as f:
        f.write('phase')
        f.write('\t')
        f.write('epoch')
        f.write('\t')
        f.write('loss')
        f.write('\t')
        f.write('acc')
        f.write('\n')

    for total_phase in ['train_original', 'train_bt', 'train_eda', 'train_ood']:
        best_val_loss = 1e+10
        model = total_model_dict[total_phase]
        
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            start_time_e = time()
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                    loader_phase = total_phase
                else:
                    write_log(logger, 'Validation start...')
                    val_loss = 0
                    val_acc = 0
                    model.eval()
                    loader_phase = phase

                for i, batch_iter in enumerate(tqdm(dataloader_dict[loader_phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                    # Optimizer setting
                    optimizer.zero_grad(set_to_none=True)

                    # Input setting
                    src_sequence = batch_iter[0]
                    src_att = batch_iter[1]
                    trg_label = batch_iter[2]

                    src_sequence = src_sequence.to(device, non_blocking=True)
                    src_att = src_att.to(device, non_blocking=True)
                    trg_label = trg_label.to(device, non_blocking=True)

                    # Train
                    if phase == 'train':
                        with autocast():
                            predicted = model(input_ids=src_sequence, attention_mask=src_att)['logits']
                            loss = F.cross_entropy(predicted, trg_label)

                        scaler.scale(loss).backward()
                        if args.clip_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        # loss.backward()
                        # clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        # optimizer.step()

                        if args.scheduler in ['constant', 'warmup']:
                            scheduler.step()
                        if args.scheduler == 'reduce_train':
                            scheduler.step(loss)

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloader_dict[loader_phase]):
                            acc = (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label)
                            iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                                (epoch, i, len(dataloader_dict['train']), loss, acc*100, optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                            write_log(logger, iter_log)
                            freq = 0
                        freq += 1

                    # Validation
                    if phase == 'valid':
                        with torch.no_grad():
                            predicted = model(input_ids=src_sequence, attention_mask=src_att)['logits']
                            loss = F.cross_entropy(predicted, trg_label.argmax(dim=1))
                        val_loss += loss
                        val_acc += (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label.argmax(dim=1))
                        
                if phase == 'valid':

                    if args.scheduler == 'reduce_valid':
                        scheduler.step(val_loss)
                    if args.scheduler == 'lambda':
                        scheduler.step()

                    val_loss /= len(dataloader_dict[loader_phase])
                    val_acc /= len(dataloader_dict[loader_phase])
                    write_log(logger, f'Mode: {loader_phase}')
                    write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                    write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))

                    save_file_name = os.path.join(args.model_save_path, args.data_name, loader_phase.split('_')[-1])
                    save_file_name += 'checkpoint.pth.tar'
                    if val_loss < best_val_loss:
                        write_log(logger, 'Checkpoint saving...')
                        torch.save({
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict()
                        }, save_file_name)
                        best_val_loss = val_loss
                        best_epoch = epoch
                    else:
                        else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 2)}) is better...'
                        write_log(logger, else_log)

                    with open('./results.txt', 'a') as f:
                        f.write(f'{loader_phase}')
                        f.write('\t')
                        f.write(f'{epoch}')
                        f.write('\t')
                        f.write(f'{val_loss}')
                        f.write('\t')
                        f.write(f'{val_acc}')
                        f.write('\n')

    # # 3) Results
    # write_log(logger, f'Best Epoch: {best_epoch}')
    # write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')