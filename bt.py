import time
import random
import multiprocessing
from tqdm import tqdm
from googletrans import Translator
from ktextaug import TextAugmentation
import os
import time
import h5py
import pickle
import logging
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
# Import custom modules
from task.utils import total_data_load
from task.multi import multi_bt, multi_eda
from task.augmentation.kor_eda import EDA
from utils import TqdmLoggingHandler, write_log

def list_chuck(arr, n):
    return [arr[i: i + n] for i in range(0, len(arr), n)]

def main():

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
    data_path = '/mnt/storage1/dataset'

    src_list = dict()
    trg_list = dict()

    hate_data_path = os.path.join(data_path,'korean-hate-speech-detection')

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

    translator = Translator()
    manager = multiprocessing.Manager()
    bt_src = manager.list()
    bt_trg = manager.list()

    len_ = len(src_list['train'])
    src_chunk = list_chuck(src_list['train'], int(len(src_list['train'])/11))
    trg_chunk = list_chuck(trg_list['train'], int(len(trg_list['train'])/11))

    with open('./bt_korean_hate.csv', 'w') as f:
        f.write('source')
        f.write(',')
        f.write('target')
        f.write('\n')

    def bt_multi(a_list, b_list):
        for i in tqdm(range(len(a_list)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
            try:
                out1 = translator.translate(a_list[i], dest='en')
                time.sleep(0.6)
                out2 = translator.translate(out1.text, dest='ko')
                time.sleep(0.6)
                bt_src.append(out2.text)
                bt_trg.append(b_list[i])
                with open('./bt_korean_hate.csv', 'a') as f:
                    f.write(out2.text)
                    f.write(',')
                    f.write(f'{b_list[i]}')
                    f.write('\n')
            except:
                print('except!')
                bt_src.append(a_list[i])
                bt_trg.append(b_list[i])
                with open('./bt_korean_hate.csv', 'a') as f:
                    f.write(f'{a_list[i]}')
                    f.write(',')
                    f.write(f'{b_list[i]}')
                    f.write('\n')

    manager = multiprocessing.Manager()
    process1 = multiprocessing.Process(target=bt_multi, args=[src_chunk[0], trg_chunk[0]])
    process2 = multiprocessing.Process(target=bt_multi, args=[src_chunk[1], trg_chunk[1]])
    process3 = multiprocessing.Process(target=bt_multi, args=[src_chunk[2], trg_chunk[2]])
    process4 = multiprocessing.Process(target=bt_multi, args=[src_chunk[3], trg_chunk[3]])
    process5 = multiprocessing.Process(target=bt_multi, args=[src_chunk[4], trg_chunk[4]])
    process6 = multiprocessing.Process(target=bt_multi, args=[src_chunk[5], trg_chunk[5]])
    process7 = multiprocessing.Process(target=bt_multi, args=[src_chunk[6], trg_chunk[6]])
    process8 = multiprocessing.Process(target=bt_multi, args=[src_chunk[7], trg_chunk[7]])
    process9 = multiprocessing.Process(target=bt_multi, args=[src_chunk[8], trg_chunk[8]])
    process10 = multiprocessing.Process(target=bt_multi, args=[src_chunk[9], trg_chunk[9]])
    process11 = multiprocessing.Process(target=bt_multi, args=[src_chunk[10], trg_chunk[10]])
    process12 = multiprocessing.Process(target=bt_multi, args=[src_chunk[11], trg_chunk[11]])

    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()
    process9.start()
    process10.start()
    process11.start()
    process12.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()
    process9.join()
    process10.join()
    process11.join()
    process12.join()

    assert len(bt_src) == len(src_list['train'])
    assert len(bt_trg) == len(trg_list['train'])

    pd.DataFrame({
        'src': list(bt_src),
        'trg': list(bt_trg)
    }).to_csv('./bt_korean_hate2.csv', index=False)

if __name__ == "__main__":
	main()