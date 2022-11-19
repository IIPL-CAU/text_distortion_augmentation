import torch
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM 

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
model.eval()

dat = pd.read_csv('/mnt/storage1/dataset/nsmc/ratings_train.txt', sep='\t')

pos_dat = dat[dat['label']==1]
neg_dat = dat[dat['label']==0]

for i in tqdm(range(len(dat))):
    pos_list = pos_dat['document'].sample(n=2).tolist()
    neg_list = neg_dat['document'].sample(n=2).tolist()
    prompt = '(0)'
    prompt += pos_list[0]
    prompt += '\n (1)'
    prompt += neg_list[0]
    prompt += '\n (0)'
    prompt += pos_list[1]
    prompt += '\n (1)'
    prompt += neg_list[1]
    prompt += '\n (0.5)'
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=300)
        generated = tokenizer.batch_decode(gen_tokens)[0]
    with open('./kogptmix.txt', 'a') as f:
        f.write('\n')
        f.write(generated.split('(0.5)')[1].split('\n')[0])
    if i % 100 == 0:
        print(generated.split('(0.5)')[1].split('\n')[0])