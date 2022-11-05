import os
import time
import h5py
import pickle
import logging
import numpy as np
# Import custom modules
from task.preprocessing.tokenizer.spm_tokenize import spm_tokenizing
from task.preprocessing.tokenizer.plm_tokenize import plm_tokenizing
from task.preprocessing.tokenizer.spacy_tokenize import spacy_tokenizing
from task.preprocessing.data_load import aug_data_load
from utils import TqdmLoggingHandler, write_log

from datasets import load_dataset

def augmenting(args):

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

    src_list, trg_list = aug_data_load(args)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.aug_data_name in ['korpora_kr', 'aihub_kr']:
        src_language = 'kr'
    elif args.aug_data_name in ['korpora_en', 'aihub_en']:
        src_language = 'en'

    if args.tokenizer == 'spm':
        processed_src, word2id_src = spm_tokenizing(src_list, args, domain='src')
    else:
        processed_src, word2id_src = plm_tokenizing(src_list, args, domain='src', language=src_language)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.aug_data_name, args.tokenizer)

    if args.tokenizer == 'spm':
        save_name = f'aug_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'aug.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('aug_input_ids', data=processed_src['aug'])
        f.create_dataset('aug_attention_mask', data=processed_src['aug_mask'])
        f.create_dataset('aug_label', data=np.array(trg_list['aug']).astype(int))