import os
import torch
from torch.nn import functional as F

def input_to_device(args, batch_iter, device):
    # Input, output setting
    if args.task in ['translation', 'style_transfer', 'summarization', 'reconstruction']:
        src_sequence = batch_iter[0]
        src_att = batch_iter[1]
        trg_sequence = batch_iter[2]
        trg_att = batch_iter[3]
        src_img = None
        trg_label = None

        src_sequence = src_sequence.to(device, non_blocking=True)
        src_att = src_att.to(device, non_blocking=True)
        trg_sequence = trg_sequence.to(device, non_blocking=True)
        trg_att = trg_att.to(device, non_blocking=True)

    elif args.task in ['classification']:
        src_sequence = batch_iter[0]
        src_att = batch_iter[1]
        trg_label = batch_iter[2]
        src_img = None
        trg_sequence = None
        trg_att = None

        src_sequence = src_sequence.to(device, non_blocking=True)
        src_att = src_att.to(device, non_blocking=True)
        trg_label = trg_label.to(device, non_blocking=True)

    elif args.task in ['multi-modal_classification']:
        src_sequence = batch_iter[0]
        src_att = batch_iter[1]
        src_img = batch_iter[2]
        trg_label = batch_iter[3]
        trg_sequence = None
        trg_att = None

        src_sequence = src_sequence.to(device, non_blocking=True)
        src_att = src_att.to(device, non_blocking=True)
        src_img = src_img.to(device, non_blocking=True)
        trg_label = trg_label.to(device, non_blocking=True)

    return src_sequence, src_att, src_img, trg_label, trg_sequence, trg_att


def label_smoothing_loss(pred, gold, trg_pad_idx, smoothing_eps=0.1):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(trg_pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).mean()
    return loss

def model_save_name(args):
    save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
    save_name_pre = 'checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # SentencePiece
    if args.tokenizer == 'spm':
        save_name_pre += f'_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}'

    # Variational
    if args.variational:
        save_name_pre += f'_v_{args.variational_model}'
        save_name_pre += f'_token_{args.variational_token_processing}'
        save_name_pre += f'_with_target_{args.variational_with_target}'
        save_name_pre += f'_cnn_encoder_{args.cnn_encoder}'
        save_name_pre += f'_cnn_decoder_{args.cnn_decoder}'
        save_name_pre += f'_latent_add_{args.latent_add_encoder_out}'
        
    save_name_pre += '.pth.tar'
    save_file_name = os.path.join(save_path, save_name_pre)
    return save_file_name

def results_save_name(args):
    if not os.path.exists(os.path.join(args.result_path, args.task)):
        os.mkdir(os.path.join(args.result_path, args.task))
    if not os.path.exists(os.path.join(args.result_path, args.task, args.data_name)):
        os.mkdir(os.path.join(args.result_path, args.task, args.data_name))
    result_path_ = os.path.join(args.result_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(result_path_):
        os.mkdir(result_path_)
    if args.tokenizer == 'spm':
        save_name_pre = f'Result_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.csv'
    else:
        save_name_pre = f'Result_v_{args.variational_mode}_p_{args.parallel}.csv'
    save_result_name = os.path.join(result_path_, save_name_pre)

    return save_result_name