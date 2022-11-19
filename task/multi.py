import random
import multiprocessing
from tqdm import tqdm
from googletrans import Translator
from ktextaug import TextAugmentation

def list_chuck(arr, n):
    return [arr[i: i + n] for i in range(0, len(arr), n)]

def multi_bt(train_src_list, train_trg_list):

    translator = Translator()
    manager = multiprocessing.Manager()
    bt_src = manager.list()
    bt_trg = manager.list()

    len_ = len(train_src_list)
    src_chunk = list_chuck(train_src_list, int(len(train_src_list)/11))
    trg_chunk = list_chuck(train_trg_list, int(len(train_trg_list)/11))

    def bt_multi(a_list, b_list):
        for i in tqdm(range(len(a_list)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
            try:
                out1 = translator.translate(a_list[i], dest='en')
                time.sleep(0.6)
                out2 = translator.translate(out1.text, dest='ko')
                time.sleep(0.6)
                bt_src.append(out2.text)
                bt_trg.append(b_list[i])
            except:
                bt_src.append(a_list[i])
                bt_trg.append(b_list[i])

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

    assert len(bt_src) == len(train_src_list)
    assert len(bt_trg) == len(train_trg_list)

    return list(bt_src), list(bt_trg)

def multi_eda(train_src_list, train_trg_list):

    agent = TextAugmentation(tokenizer="mecab", num_processes=1)

    eda_src = manager.list()
    eda_trg = manager.list()

    len_ = len(train_src_list)
    src_chunk = list_chuck(train_src_list, int(len(train_src_list)/11))
    trg_chunk = list_chuck(train_trg_list, int(len(train_trg_list)/11))

    def eda_multi(a_list, b_list):
        for i in tqdm(range(len(a_list)), bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):
            eda_ = random.choice(['random_insert', 'random_delete', 'random_swap'])
            eda_src.append(agent.generate(a_list[i], mode=eda_))
            eda_trg.append(b_list[i])

    manager = multiprocessing.Manager()
    process1 = multiprocessing.Process(target=eda_multi, args=[src_chunk[0], trg_chunk[0]])
    process2 = multiprocessing.Process(target=eda_multi, args=[src_chunk[1], trg_chunk[1]])
    process3 = multiprocessing.Process(target=eda_multi, args=[src_chunk[2], trg_chunk[2]])
    process4 = multiprocessing.Process(target=eda_multi, args=[src_chunk[3], trg_chunk[3]])
    process5 = multiprocessing.Process(target=eda_multi, args=[src_chunk[4], trg_chunk[4]])
    process6 = multiprocessing.Process(target=eda_multi, args=[src_chunk[5], trg_chunk[5]])
    process7 = multiprocessing.Process(target=eda_multi, args=[src_chunk[6], trg_chunk[6]])
    process8 = multiprocessing.Process(target=eda_multi, args=[src_chunk[7], trg_chunk[7]])
    process9 = multiprocessing.Process(target=eda_multi, args=[src_chunk[8], trg_chunk[8]])
    process10 = multiprocessing.Process(target=eda_multi, args=[src_chunk[9], trg_chunk[9]])
    process11 = multiprocessing.Process(target=eda_multi, args=[src_chunk[10], trg_chunk[10]])
    process12 = multiprocessing.Process(target=eda_multi, args=[src_chunk[11], trg_chunk[11]])

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

    assert len(eda_src) == len(train_src_list)
    assert len(eda_trg) == len(train_trg_list)

    return list(eda_src), list(eda_trg)