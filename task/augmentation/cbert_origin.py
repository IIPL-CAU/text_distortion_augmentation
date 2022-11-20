from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import os
from tqdm import tqdm, trange
import argparse

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

import cbert_utils_rf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = cbert_utils_rf.rev_wordpiece(tokens)
    return outputs

def load_model(model_name):
    weights_path = os.path.join(model_name)
    model = torch.load(weights_path)
    return model

def cbert_original(task_name, sentences, label, sample_ratio=7, temp=2.0, num_train_epochs=1):
    
    AugProcessor = cbert_utils_rf.AugProcessor()   
    processors = {
    ## you can add your processor here
    "nsmc": AugProcessor,
    "korean-hate-speech-detection": AugProcessor,
    }
    # args = parser.parse_args()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    
    if task_name == 'nsmc':
        label_list = ['0', '1']
        
    if task_name == 'korean-hate-speech-detection':
        label_list = ['0', '1', '2']
    
    else :
        print('no task, put specifl label_list for own ur task')
    
    train_features, num_train_steps, train_dataloader = \
        cbert_utils_rf.construct_train_dataloader(train_examples=sentences, label_list=label_list, labels=label, max_seq_length=64, train_batch_size=32, num_train_epochs=9.0, tokenizer=tokenizer, device=device)
        
    MASK_id = cbert_utils_rf.convert_tokens_to_ids(['[MASK]'], tokenizer)[0]
        
    for e in trange(int(num_train_epochs), desc="Epoch"):

        torch.cuda.empty_cache()
        cbert_name = "./preprocessed/cbert/korean-hate-speech-detection/BertForMaskedLM_korean-hate-speech-detection_epoch_10" # e+1
        model = load_model(cbert_name)
        model.cuda()
        
        aug_list = []
        aug_label = []

        for _, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch
            input_lens = [sum(mask).item() for mask in input_mask]
            masked_idx = np.squeeze([np.random.randint(0, l, max(l//sample_ratio, 1)) for l in input_lens])
            
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id
            predictions = model(init_ids, input_mask, segment_ids)
            predictions = torch.nn.functional.softmax(predictions[0]/temp, dim=2)            

            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, 1, replacement=True)[idx]
                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred
                    new_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)

                    # print([new_str, seg[0].item()])
                    aug_list.append(new_str)
                    aug_label.append(seg[0].item())
            torch.cuda.empty_cache()
            
        return aug_label, aug_list
        
# if __name__ == "__main__":
#     main()

class CBertAgent:
    def __init__(self):
        pass

    def generate(self, task, sentence, label):
        return cbert_original(task_name=task,
                              sentences=sentence,
                              label=label)