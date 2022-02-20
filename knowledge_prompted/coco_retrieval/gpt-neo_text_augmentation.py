import os
import json
from tqdm import tqdm
import time
from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

def get_response_generator(generator, prompt):
    response = generator(prompt, do_sample=True, max_length=282) #TODO
    return response

def get_response_gpt_neo(model, tokenizer, prompt, n, device, num_sent_per_generate = 5):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    knowledge_texts = []
    
    num_sent_per_generate = 10
    subloop_num = n//num_sent_per_generate
    for i in range(subloop_num):
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=282, # 218 + 64
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.2, # 1.0 means no penalty
            num_return_sequences=num_sent_per_generate
        )
        # gen_text = tokenizer.batch_decode(gen_tokens)[0]
        gen_text_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # post process 
        for gen_text in gen_text_batch:
            gen_text = gen_text[len(prompt):]
            gen_text = gen_text.split('\n')[0].strip()
            knowledge_texts.append(gen_text)

    return knowledge_texts

""" temperature = 0.9, call generate n=10 times, query = 'a person is connecting something to system'
    ['A connector is used between two things. Many electronic cords, for example, are also called connectors.', 'A person may connect the output of a computer (the computer screen) with the input of another (a keyboard), and then they may use the combination of those inputs to make the system work. For example, someone may have a computer set to recognize as an input a particular button on a keyboard, and then they may', 'A system is usually used in making things move.', 'A system, such as a computer system, a program, an algorithm, or a computer program, has an input interface (like a keyboard) and an output interface (like a screen), which are connected wirelessly.', 'The connectors are often used to connect things like switches.', 'A connector is a device that attaches things.', 'The person is checking something by clicking a button.', 'A system can be anything you need. The system may only be software, or a network connection.', 'A person usually connects something to power source in the system.', 'Something can only be connected to other things, i.e., it cannot be connected directly to the environment.']
"""
def load_knowledge_prompt_template(path):
    ret = ''
    with open(path) as f:
        for line in f:
            ret += line
    return ret

def construct_prompt(prefix, input_query, suffix = 'Knowledge:'):
    input_query = input_query.strip()
    if not input_query.endswith('.'):
        input_query += '.'
    ret = prefix + input_query + '\n' + suffix
    return ret

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def main():
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)

    '''gpt-neo config'''
    ## use generator pipeline
    # generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
    model_name = "EleutherAI/gpt-neo-2.7B"
    print(f'loading {model_name} ...')
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.to(device)
    print(f'done loading {model_name}')

    n = 10 # generate how many knowledge
    print(f'generate num of knowledge: {n}')

    '''load json'''
    # TODO:
    
    augmented_ann = []
    
    '''query api'''
    prompt_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/knowledge_prompt/prompts/p_test2.txt'
    prompt_template = load_knowledge_prompt_template(prompt_path)
    for item in tqdm(original_ann):
        query = item['caption']
        prompt = construct_prompt(prompt_template, query, suffix = 'Knowledge:')
        
        ## use generator
        # response = get_response_generator(generator,prompt)
        # knowledge_texts = [s['generated_text'].strip() for s in response]
        # print(knowledge_texts)
        
        knowledge_texts = get_response_gpt_neo(model,tokenizer,prompt,n,device)
        new_item = item.copy()
        new_item['knowledge'] = knowledge_texts
        augmented_ann.append(new_item)


    '''output'''
    with open(output_txt_jsonl,'w') as out:
        for item in augmented_ann:
            out.write(json.dumps(item))
            out.write('\n')


if __name__ == '__main__':
    main()