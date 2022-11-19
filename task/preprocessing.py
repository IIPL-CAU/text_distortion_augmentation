import os
import time
import h5py
import pickle
import random
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from ktextaug import TextAugmentation
from transformers import  AutoTokenizer
# Import custom modules
from task.utils import total_data_load
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
    agent = TextAugmentation(tokenizer="mecab", num_processes=1)

    # processed_sequences['bt'] = dict()
    # processed_sequences['bt']['label'] = list()
    processed_sequences['eda'] = dict()
    processed_sequences['ood'] = dict()

    # Back-translation
    # write_log(logger, 'Back-translation...')
    # manager = multiprocessing.Manager()
    # bt_src = manager.list()
    # bt_trg = manager.list()

    # len_ = len(src_list['train'])
    # def list_chuck(arr, n):
    #     return [arr[i: i + n] for i in range(0, len(arr), n)]
    # src_chunk = list_chuck(src_list['train'], int(len(src_list['train'])/8))
    # trg_chunk = list_chuck(trg_list['train'], int(len(trg_list['train'])/8))

    # def bt_multi(a_list, b_list):
    #     for i in range(len(a_list)):
    #         bt_src.append(agent.generate(a_list[i]))
    #         bt_trg.append(b_list[i])

    # manager = multiprocessing.Manager()
    # process1 = multiprocessing.Process(target=bt_multi, args=[src_chunk[0], trg_chunk[0]])
    # process2 = multiprocessing.Process(target=bt_multi, args=[src_chunk[1], trg_chunk[1]])
    # process3 = multiprocessing.Process(target=bt_multi, args=[src_chunk[2], trg_chunk[2]])
    # process4 = multiprocessing.Process(target=bt_multi, args=[src_chunk[3], trg_chunk[3]])
    # process5 = multiprocessing.Process(target=bt_multi, args=[src_chunk[4], trg_chunk[4]])
    # process6 = multiprocessing.Process(target=bt_multi, args=[src_chunk[5], trg_chunk[5]])
    # process7 = multiprocessing.Process(target=bt_multi, args=[src_chunk[6], trg_chunk[6]])
    # process8 = multiprocessing.Process(target=bt_multi, args=[src_chunk[7], trg_chunk[7]])

    # process1.start()
    # process2.start()
    # process3.start()
    # process4.start()
    # process5.start()
    # process6.start()
    # process7.start()
    # process8.start()

    # process1.join()
    # process2.join()
    # process3.join()
    # process4.join()
    # process5.join()
    # process6.join()
    # process7.join()
    # process8.join()

    # print(list(bt_src))
    # print(len(bt_src))
    # assert len(bt_src) == len(src_list['train'])

    # write_log(logger, 'Back-translation Done')
    # write_log(logger, 'Back-translation Tokenizing...')

    # # for i in tqdm(range(len(src_list['train'])), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
    # #     bt_src.append(agent.generate(src_list['train'][i]))

    # encoded_dict = \
    # tokenizer(
    #     list(bt_src),
    #     max_length=args.src_max_len,
    #     padding='max_length',
    #     truncation=True
    # )
    # processed_sequences['bt']['input_ids'] = encoded_dict['input_ids']
    # processed_sequences['bt']['attention_mask'] = encoded_dict['attention_mask']
    # processed_sequences['bt']['token_type_ids'] = encoded_dict['token_type_ids']

    # write_log(logger, 'Back-translation Tokenizing Done')

    # Easy Data Augmentation
    write_log(logger, 'EDA...')
    eda_src = list()
    for i in tqdm(range(len(src_list['train'])), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
        eda_ = random.choice(['random_insert', 'random_delete', 'random_swap', 'synonym_replace'])
        eda_src.append(agent.generate(src_list['train'][i], mode=eda_))

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

    aihub_data_path = os.path.join(args.data_path,'AI_Hub_KR_EN')
    dat = pd.read_csv(os.path.join(aihub_data_path, '1_구어체(1).csv')).dropna()
    ood_src += dat['KR'].tolist()

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
        # f.create_dataset('train_bt_src_input_ids', data=processed_sequences['bt']['input_ids'])
        # f.create_dataset('train_bt_src_attention_mask', data=processed_sequences['bt']['attention_mask'])
        # f.create_dataset('train_bt_label', data=processed_sequences['bt']['label'])
        f.create_dataset('train_eda_src_input_ids', data=processed_sequences['eda']['input_ids'])
        f.create_dataset('train_eda_src_attention_mask', data=processed_sequences['eda']['attention_mask'])
        f.create_dataset('train_ood_src_input_ids', data=processed_sequences['ood']['input_ids'])
        f.create_dataset('train_ood_src_attention_mask', data=processed_sequences['ood']['attention_mask'])
        f.create_dataset('train_label', data=np.array(trg_list['train']).astype(int))

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'w') as f:
        f.create_dataset('test_src_input_ids', data=processed_sequences['test']['input_ids'])
        f.create_dataset('test_src_attention_mask', data=processed_sequences['test']['attention_mask'])
        f.create_dataset('test_label', data=np.array(trg_list['test']).astype(int))

    # Word2id pickle file save
    word2id_dict = {
        'src_language' : src_language, 
        'src_word2id' : word2id_src,
        'num_labels': len(set(trg_list['train']))
    }

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')