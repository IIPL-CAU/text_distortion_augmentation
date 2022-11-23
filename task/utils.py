import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset

def data_split_index(seq, valid_ration: float = 0.1, test_ratio: float = 0.03):

    paired_data_len = len(seq)
    valid_num = int(paired_data_len * valid_ration)
    test_num = int(paired_data_len * test_ratio)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def total_data_load(args):

    src_list = dict()
    trg_list = dict()

    if args.data_name == 'korean_hate_speech':
        hate_data_path = os.path.join(args.data_path,'korean-hate-speech-detection')

        train_dat = pd.read_csv(os.path.join(hate_data_path, 'train.hate.csv'))
        valid_dat = pd.read_csv(os.path.join(hate_data_path, 'dev.hate.csv'))
        test_dat = pd.read_csv(os.path.join(hate_data_path, 'test.hate.no_label.csv'))

        train_dat['label'] = train_dat['label'].replace('none', 0)
        train_dat['label'] = train_dat['label'].replace('hate', 1)
        train_dat['label'] = train_dat['label'].replace('offensive', 2)
        valid_dat['label'] = valid_dat['label'].replace('none', 0)
        valid_dat['label'] = valid_dat['label'].replace('hate', 1)
        valid_dat['label'] = valid_dat['label'].replace('offensive', 2)

        src_list['train'] = train_dat['comments'].tolist()
        trg_list['train'] = train_dat['label'].tolist()
        src_list['valid'] = valid_dat['comments'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()
        src_list['test'] = test_dat['comments'].tolist()
        trg_list['test'] = [0 for _ in range(len(test_dat))]

    if args.data_name == 'nsmc':
        nsmc_data_path = os.path.join(args.data_path,'nsmc')

        train_dat = pd.read_csv(os.path.join(nsmc_data_path, 'ratings_train.txt'), 
                                sep='\t', names=['id', 'description', 'label'], header=0).dropna()
        test_dat = pd.read_csv(os.path.join(nsmc_data_path, 'ratings_test.txt'), 
                                    sep='\t', names=['id', 'description', 'label'], header=0).dropna()

        train_index, valid_index, test_index = data_split_index(train_dat, test_ratio=0)

        src_list['train'] = [train_dat['description'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['label'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['description'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['label'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['description'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'klue_topic':
        klue_data_path = os.path.join(args.data_path,'KLUE_TC')

        with open(os.path.join(klue_data_path, 'ynat-v1.1_train.json'), 'r') as f:
            dat = json.load(f)
        with open(os.path.join(klue_data_path, 'ynat-v1.1_dev.json'), 'r') as f:
            test_dat = json.load(f)

        title_list = [t['title'] for t in dat]
        label_list = [t['label'] for t in dat]
        test_title_list = [t['title'] for t in test_dat]
        test_label_list = [t['label'] for t in test_dat]

        train_dat = pd.DataFrame({
            'description': title_list,
            'label': label_list
        })
        test_dat = pd.DataFrame({
            'description': test_title_list,
            'label': test_label_list
        })
        train_dat['label'] = train_dat['label'].replace('IT과학', 0)
        train_dat['label'] = train_dat['label'].replace('경제', 1)
        train_dat['label'] = train_dat['label'].replace('사회', 2)
        train_dat['label'] = train_dat['label'].replace('생활문화', 3)
        train_dat['label'] = train_dat['label'].replace('세계', 4)
        train_dat['label'] = train_dat['label'].replace('스포츠', 5)
        train_dat['label'] = train_dat['label'].replace('정치', 6)

        test_dat['label'] = test_dat['label'].replace('IT과학', 0)
        test_dat['label'] = test_dat['label'].replace('경제', 1)
        test_dat['label'] = test_dat['label'].replace('사회', 2)
        test_dat['label'] = test_dat['label'].replace('생활문화', 3)
        test_dat['label'] = test_dat['label'].replace('세계', 4)
        test_dat['label'] = test_dat['label'].replace('스포츠', 5)
        test_dat['label'] = test_dat['label'].replace('정치', 6)

        train_index, valid_index, test_index = data_split_index(train_dat, test_ratio=0)

        src_list['train'] = [train_dat['description'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['label'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['description'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['label'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['description'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    return src_list, trg_list

def aug_data_load(args):

    aug_src_list = dict()
    aug_trg_list = dict()

    if 'korpora' in args.aug_data_name:
        korpora_data_path = os.path.join(args.data_path, 'korpora')

        dat = pd.read_csv(os.path.join(korpora_data_path, 'pair_kor.csv'), names=['kr']).dropna()

        aug_src_list['aug'] = dat['kr']

    # AIHUB

    if 'aihub' in args.aug_data_name:
        args.data_path = os.path.join(args.data_path,'AI_Hub_KR_EN')
        dat = pd.read_csv(os.path.join(args.data_path, '1_구어체(1).csv')).dropna()

        if 'kr' in args.aug_data_name:
            aug_src_list['aug'] = dat['KR']

        if 'en' in args.aug_data_name:
            aug_src_list['aug'] = dat['EN']

    # Korean hate speech

    if 'korean_hate_speech' in args.aug_data_name:
        hate_data_path = os.path.join(args.data_path,'korean-hate-speech-detection')
        if args.aug_type == 'half':
            dat = pd.read_csv(os.path.join(hate_data_path, 'train_half.tsv'), sep='\t').dropna()
            aug_src_list['aug'] = dat['comments']
            args.aug_type = 'half'

        elif args.aug_type == 'origin':
            dat = pd.read_csv(os.path.join(hate_data_path, 'train_origin.tsv'), sep='\t').dropna()
            aug_src_list['aug'] = dat['comments']
            aug_trg_list['aug'] = dat['label'].tolist()
            args.aug_type = 'origin'
        else:
            raise Exception("OOD?!")

    # NSMC

    if 'nsmc' in args.aug_data_name:
        nsmc_data_path = os.path.join(args.data_path,'nsmc')
        if args.aug_type == 'half':
            dat = pd.read_csv(os.path.join(nsmc_data_path, 'train_half.tsv'), sep='\t', names=['comments', 'label']).dropna()
            aug_src_list['aug'] = dat['comments']
            args.aug_type = 'half'

        elif args.aug_type == 'origin':
            dat = pd.read_csv(os.path.join(nsmc_data_path, 'train_origin.tsv'), sep='\t', names=['comments', 'label']).dropna()
            aug_src_list['aug'] = dat['comments']
            aug_trg_list['aug'] = dat['label'].tolist()
            args.aug_type = 'origin'
        elif args.aug_type == 'bt':
            train_dat = pd.read_csv(os.path.join(nsmc_data_path, 'ratings_train.txt'), 
                                    sep='\t', names=['id', 'description', 'label'], header=0).dropna()
            agent = TextAugmentation(tokenizer="mecab", num_processes = 1)

        else:
            raise Exception("OOD?!")

    if len(aug_trg_list) == 0:
        aug_trg_list['aug'] = [-1 for _ in range(len(aug_src_list['aug']))]

    return aug_src_list, aug_trg_list