# Import modules
import os
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

    if args.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        tb_writer.add_text('args', str(args))

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
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

    if args.train_with_aug:
        save_path = os.path.join(args.preprocess_path, args.aug_data_name, args.tokenizer)
        if args.tokenizer == 'spm':
            save_name = f'aug_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
        else:
            save_name = f'aug.hdf5'

        with h5py.File(os.path.join(save_path, save_name), 'r') as f:
            aug_input_ids = f.get('aug_input_ids')[:]
            aug_attention_mask = f.get('aug_attention_mask')[:]
            aug_label = f.get('aug_label')[:]
            aug_label = torch.full((len(aug_label), num_labels), 1 / num_labels).numpy()

        train_src_input_ids = np.append(train_src_input_ids, aug_input_ids, axis=0)
        train_src_attention_mask = np.append(train_src_attention_mask, aug_attention_mask, axis=0)
        train_trg_list = np.append(train_trg_list, aug_label, axis=0)

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
            model = BertForSequenceClassification.from_pretrained('beomi/kobert', num_labels=num_labels)
    elif args.model_type == 'bart':
        if src_language == 'kr':
            model = BartForSequenceClassification.from_pretrained('cosmoquester/bart-ko-mini', num_labels=num_labels)
    model = model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               trg_list=train_trg_list, src_max_len=args.src_max_len),
        'valid': CustomDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                               trg_list=valid_trg_list, src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")
    
    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
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

    best_val_loss = 1e+10

    write_log(logger, 'Traing start!')

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, batch_iter in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

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

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), loss, acc*100, optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                    if args.use_tensorboard:
                        acc = (predicted.max(dim=1)[1] == trg_label.argmax(dim=1)).sum() / len(trg_label)
                        
                        tb_writer.add_scalar('TRAIN/Loss', loss, (epoch-1) * len(dataloader_dict['train']) + i)
                        tb_writer.add_scalar('TRAIN/Accuracy', acc*100, (epoch-1) * len(dataloader_dict['train']) + i)
                        tb_writer.add_scalar('USAGE/CPU_Usage', psutil.cpu_percent(), (epoch-1) * len(dataloader_dict['train']) + i)
                        tb_writer.add_scalar('USAGE/RAM_Usage', psutil.virtual_memory().percent, (epoch-1) * len(dataloader_dict['train']) + i)
                        tb_writer.add_scalar('USAGE/GPU_Usage', torch.cuda.memory_allocated(device=device), (epoch-1) * len(dataloader_dict['train']) + i) # MB Size

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

                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))

                if args.use_tensorboard:
                    tb_writer.add_scalar('VALID/Total_Loss', val_loss, epoch)
                    tb_writer.add_scalar('VALID/Accuracy', val_acc * 100, epoch)

                save_file_name = model_save_name(args)
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

    # 3) Print results
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Loss: {round(best_val_loss.item(), 2)}')
    if args.use_tensorboard:
        tb_writer.add_text('VALID/Best Epoch&Loss', f'Best Epoch: {best_epoch}\nBest Loss: {round(best_val_loss.item(), 4)}')