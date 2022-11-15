import argparse
from transformers import  AutoTokenizer
from datasets import load_dataset

def plm_tokenizing(sequence_dict: dict, args: argparse.Namespace, 
                    domain: str = 'src', language: str = 'en'):

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if args.tokenizer == 'bert':
        if language == 'en':
            tokenizer =  AutoTokenizer.from_pretrained('bert-base-cased')
        elif language == 'kr':
            tokenizer =  AutoTokenizer.from_pretrained('beomi/kcbert-base')
        else:
            raise Exception(f'{language} language does not support')
    elif args.tokenizer == 'bart':
        if language == 'en':
            tokenizer = AutoTokenizer.from_pretrained('ainize/bart-base-cnn')
        elif language =='kr':
            tokenizer = AutoTokenizer.from_pretrained('cosmoquester/bart-ko-mini')
        else:
            raise Exception(f'{language} language does not support')
    elif args.tokenizer == 'T5':
        if language == 'en':
            tokenizer = AutoTokenizer.from_pretrained("t5-base")
        elif language == 'kr':
            tokenizer = AutoTokenizer.from_pretrained('KETI-AIR/ke-t5-base')
        else:
            raise Exception(f'{language} language does not support')

    if domain == 'aug':
        encoded_dict = \
        tokenizer(
            sequence_dict['aug'].tolist(),
            max_length=args.src_max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences['aug'] = encoded_dict['input_ids']
        processed_sequences['aug_mask'] = encoded_dict['attention_mask']
        # if args.tokenizer == 'bert':
        #     processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    else:
        for phase in ['train', 'valid', 'test']:
            encoded_dict = \
            tokenizer(
                sequence_dict[phase],
                max_length=max_len,
                padding='max_length',
                truncation=True
            )
            processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
            processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
            if args.tokenizer == 'bert':
                processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

        # BART's decoder input id need to start with 'model.config.decoder_start_token_id'
        if args.tokenizer == 'bart' and domain == 'trg':
            for i in range(len(processed_sequences[phase]['input_ids'])):
                processed_sequences[phase]['input_ids'][i][0] = 2
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id