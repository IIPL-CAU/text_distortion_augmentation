import os
import time
import h5py
import pickle
import logging
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from koaeda import AEDA
from transformers import AutoTokenizer
# Import custom modules
from task.utils import total_data_load
from task.multi import multi_bt, multi_eda
from task.augmentation.kor_eda import EDA
from utils import TqdmLoggingHandler, write_log

from datasets import load_dataset

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    src_list, trg_list = total_data_load(args)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    for phase in ['train', 'valid', 'test']:
        encoded_dict = \
        tokenizer(
            src_list[phase],
            max_length=args.src_max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
        processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #===========Augmentation============#
    #===================================#

    write_log(logger, 'Augmenting...')
    start_time = time.time()

    processed_sequences['bt'] = dict()
    processed_sequences['bt']['label'] = list()
    processed_sequences['eda'] = dict()
    processed_sequences['ood'] = dict()
    processed_sequences['ood2'] = dict()

    # Back-translation
    write_log(logger, 'Back-translation...')

    # bt_src, bt_trg = multi_bt(src_list['train'], trg_list['train'])

    # bt_src = list(bt_src)
    # encoded_dict = \
    # tokenizer(
    #     bt_src,
    #     max_length=args.src_max_len,
    #     padding='max_length',
    #     truncation=True
    # )
    # processed_sequences['bt']['input_ids'] = encoded_dict['input_ids']
    # processed_sequences['bt']['attention_mask'] = encoded_dict['attention_mask']
    # processed_sequences['bt']['token_type_ids'] = encoded_dict['token_type_ids']

    ko_aeda = AEDA()
    bt_src = list()
    for text in tqdm(src_list['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
        bt_src.append(ko_aeda(text))

    encoded_dict = \
    tokenizer(
        bt_src,
        max_length=args.src_max_len,
        padding='max_length',
        truncation=True
    )
    processed_sequences['bt']['input_ids'] = encoded_dict['input_ids']
    processed_sequences['bt']['attention_mask'] = encoded_dict['attention_mask']
    processed_sequences['bt']['token_type_ids'] = encoded_dict['token_type_ids']

    bt_trg = trg_list['train']

    write_log(logger, 'Back-translation Tokenizing Done')

    # Easy Data Augmentation
    write_log(logger, 'EDA...')

    eda_src = list()
    for i in tqdm(range(len(src_list['train'])), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
        eda_src.append(EDA(src_list['train'][i], num_aug=1)[1])

    # eda_src, eda_trg = multi_eda(src_list['train'], trg_list['train'])

    encoded_dict = \
    tokenizer(
        eda_src,
        max_length=args.src_max_len,
        padding='max_length',
        truncation=True
    )
    processed_sequences['eda']['input_ids'] = encoded_dict['input_ids']
    processed_sequences['eda']['attention_mask'] = encoded_dict['attention_mask']
    processed_sequences['eda']['token_type_ids'] = encoded_dict['token_type_ids']

    # Out-of-Domain
    write_log(logger, 'OoD...')
    korpora_data_path = os.path.join(args.data_path, 'korpora')
    dat = pd.read_csv(os.path.join(korpora_data_path, 'pair_kor.csv'), names=['kr']).dropna()
    ood_src = dat['kr'].tolist()
    if args.data_name == 'klue_tc':
        hate_data_path = os.path.join(args.data_path,'korean-hate-speech-detection')
        dat = pd.read_csv(os.path.join(hate_data_path, 'train.hate.csv'))
        ood_src = dat['comments'].tolist()

    # aihub_data_path = os.path.join(args.data_path,'AI_Hub_KR_EN')
    # dat = pd.read_csv(os.path.join(aihub_data_path, '1_구어체(1).csv')).dropna()
    # ood_src += dat['KR'].tolist()

    encoded_dict = \
    tokenizer(
        ood_src,
        max_length=args.src_max_len,
        padding='max_length',
        truncation=True
    )
    processed_sequences['ood']['input_ids'] = encoded_dict['input_ids']
    processed_sequences['ood']['attention_mask'] = encoded_dict['attention_mask']
    processed_sequences['ood']['token_type_ids'] = encoded_dict['token_type_ids']

    # Out-of-Domain2
    write_log(logger, 'OoD2...')
    if args.data_name == 'korean_hate_speech':
        nsmc_data_path = os.path.join(args.data_path,'nsmc')
        train_dat = pd.read_csv(os.path.join(nsmc_data_path, 'ratings_train.txt'), 
                                sep='\t', names=['id', 'description', 'label'], header=0).dropna()
        ood2_src = train_dat['description'].tolist()

        encoded_dict = \
        tokenizer(
            ood2_src,
            max_length=args.src_max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences['ood2']['input_ids'] = encoded_dict['input_ids']
        processed_sequences['ood2']['attention_mask'] = encoded_dict['attention_mask']
        processed_sequences['ood2']['token_type_ids'] = encoded_dict['token_type_ids']

    if args.data_name == 'nsmc':
        hate_data_path = os.path.join(args.data_path,'korean-hate-speech-detection')

        train_dat = pd.read_csv(os.path.join(hate_data_path, 'train.hate.csv'))
        ood2_src = train_dat['comments'].tolist()

        encoded_dict = \
        tokenizer(
            ood2_src,
            max_length=args.src_max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences['ood2']['input_ids'] = encoded_dict['input_ids']
        processed_sequences['ood2']['attention_mask'] = encoded_dict['attention_mask']
        processed_sequences['ood2']['token_type_ids'] = encoded_dict['token_type_ids']

    else:

        processed_sequences['ood2']['input_ids'] = processed_sequences['train']['input_ids']
        processed_sequences['ood2']['attention_mask'] = processed_sequences['train']['attention_mask']
        processed_sequences['ood2']['token_type_ids'] = processed_sequences['train']['token_type_ids']

    # write_log(logger, 'C-BERT...')

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.data_name)
    save_name = f'processed.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('train_src_input_ids', data=processed_sequences['train']['input_ids'])
        f.create_dataset('train_src_attention_mask', data=processed_sequences['train']['attention_mask'])
        f.create_dataset('valid_src_input_ids', data=processed_sequences['valid']['input_ids'])
        f.create_dataset('valid_src_attention_mask', data=processed_sequences['valid']['attention_mask'])
        f.create_dataset('train_label', data=np.array(trg_list['train']).astype(int))
        f.create_dataset('valid_label', data=np.array(trg_list['valid']).astype(int))

    with h5py.File(os.path.join(save_path, 'aug_' + save_name), 'w') as f:
        f.create_dataset('train_bt_src_input_ids', data=processed_sequences['bt']['input_ids'])
        f.create_dataset('train_bt_src_attention_mask', data=processed_sequences['bt']['attention_mask'])
        f.create_dataset('train_bt_label', data=np.array(bt_trg).astype(int))
        f.create_dataset('train_eda_src_input_ids', data=processed_sequences['eda']['input_ids'])
        f.create_dataset('train_eda_src_attention_mask', data=processed_sequences['eda']['attention_mask'])
        f.create_dataset('train_eda_label', data=np.array(trg_list['train']).astype(int))
        f.create_dataset('train_ood_src_input_ids', data=processed_sequences['ood']['input_ids'])
        f.create_dataset('train_ood_src_attention_mask', data=processed_sequences['ood']['attention_mask'])
        f.create_dataset('train_ood_label', data=np.array(trg_list['train']).astype(int))
        f.create_dataset('train_ood2_src_input_ids', data=processed_sequences['ood2']['input_ids'])
        f.create_dataset('train_ood2_src_attention_mask', data=processed_sequences['ood2']['attention_mask'])
        f.create_dataset('train_ood2_label', data=np.array(trg_list['train']).astype(int))

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_sequences['test']['input_ids'])
        f.create_dataset('test_src_attention_mask', data=processed_sequences['test']['attention_mask'])
        f.create_dataset('test_label', data=np.array(trg_list['test']).astype(int))

    # Word2id pickle file save
    word2id_dict = {
        'src_language' : 'kr', 
        'src_word2id' : tokenizer.get_vocab(),
        'num_labels': len(set(trg_list['train']))
    }

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')