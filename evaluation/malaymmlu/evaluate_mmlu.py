from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm
from threading import Thread
from queue import Queue
import pandas as pd
import json
import os
import click

@click.command()
@click.option('--model')
@click.option('--revision')
@click.option('--batch_size', default = 16)
@click.option('--tp_size', default = 1)
def main(model, revision, batch_size, tp_size):
    model_name = model.split('/')[-1]
    folder = model_name
    os.makedirs(folder, exist_ok = True)
    data = pd.read_json('MalayMMLU_0shot.json')
    inputs = []
    ids = []
    id2key = ['A', 'B', 'C', 'D', 'E']

    for i in tqdm(range(len(data))):
        filename = f'{folder}/{i}.json'
        try:
            with open(filename) as fopen:
                json.load(fopen)
            continue
        except:
            pass
        
        row = data.iloc[i]
        ques = row['prompt']
        p = f"Berikut adalah soalan aneka pilihan tentang {row['subject']}. Sila berikan jawapan sahaja.\n\n" + ques
        inputs.append(p)
        ids.append(i)
    
    if len(inputs):
        model = LLM(
            model, 
            revision=revision,
            max_model_len=1024, 
            enforce_eager=True, 
            tensor_parallel_size=tp_size,
        )
        tokenizer = model.get_tokenizer()
        guided_decoding_params = GuidedDecodingParams(choice=id2key)
        sampling_params = SamplingParams(max_tokens=2, n=5, guided_decoding=guided_decoding_params)

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]
        
        prompts = []
        for q in batch_inputs:
            conversation = [{"role": "user", "content": q}]
            t = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = True)
            prompts.append(t)
            
        result = model.generate(prompts, sampling_params = sampling_params, use_tqdm = False)
        for n in range(len(result)):
            filename = f'{folder}/{batch_ids[n]}.json'
            
            l = [result[n].outputs[k].text for k in range(len(result[n].outputs))]
            voted = max(set(l := l), key=l.count)
            with open(filename, 'w') as fopen:
                json.dump(voted, fopen)
    
    answers = []
    for i in tqdm(range(len(data))):
        filename = f'{folder}/{i}.json'
        with open(filename) as fopen:
            a = json.load(fopen)
        answers.append(a)
    
    full_accs = []
    categories = []
    for cat in data.category.unique():
        keep_ids = list(data[data.category == cat].index)
        match = [answers[i].strip() == data.iloc[i]['key'] for i in keep_ids]
        full_acc = (sum(match) / len(keep_ids)) * 100
        full_accs.append(full_acc)
        categories.append(cat)
    
    df = pd.DataFrame({
        "Model": [model_name] * len(full_accs),
        "Accuracy": full_accs,
        "shot": [0] * len(full_accs),
        "category": categories
    })
    print(df)

    category2amount = dict(data.category.value_counts())
    sum_acc = 0

    for category in category2amount.keys():
        category_df = df[df.category == category]
        if not category_df.empty:
            sum_acc += category2amount[category] * category_df.iloc[0]['Accuracy'] 
    average_acc = sum_acc / len(data)

    accuracy_info = {
        'average accuracy': average_acc
    }
    for i in range(len(df)):
        category = df.iloc[i].category
        accuracy_info[f'accuracy for {category}'] = df.iloc[i].Accuracy

    model_name = df['Model'].iloc[0]
    exp = "full"
    shot = df['shot'].iloc[0]
    print("Model :", model_name)
    print("Metric :",exp)
    print("Shot :",shot)
    for k,v in accuracy_info.items():
        print(k,v)
    
if __name__ == '__main__':
    main()