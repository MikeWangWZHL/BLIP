import json
import os
from collections import defaultdict

def get_arg0_arg1_from_srl_result(verb_obj, words):
    arg0_indices = []
    arg1_indices = []
    for i in range(len(verb_obj['tags'])):
        tag = verb_obj['tags'][i]
        parsed_tag = tag.split('-')
        if len(parsed_tag) < 2:
            continue
        if parsed_tag[1] == 'ARG0':
            arg0_indices.append(i)
        elif parsed_tag[1] == 'ARG1':
            arg1_indices.append(i)
    if arg0_indices:
        arg0 = ' '.join(words[arg0_indices[0]:arg0_indices[-1]+1])
    else:
        arg0 = ''
    if arg1_indices:
        arg1 = ' '.join(words[arg1_indices[0]:arg1_indices[-1]+1])
    else:
        arg1 = ''
    return arg0, arg1

if __name__ == '__main__':

    input_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/srl_results/srl__coco_karpathy_train.json'

    verbs_to_frequency = defaultdict(int)
    verbs_to_arg0_arg1 = defaultdict(list)
    
    num_caption_without_verb = 0
    num_caption_with_verb = 0

    for item in json.load(open(input_json)):
        caption = item['caption']
        if caption['verbs'] == []:
            print(caption['text'])
            num_caption_without_verb += 1
            continue
        else:
            num_caption_with_verb += 1
            words = caption['words']
            for verb_obj in caption['verbs']:
                verbs_to_frequency[verb_obj['verb']] += 1
                arg0, arg1 = get_arg0_arg1_from_srl_result(verb_obj, words)
                verbs_to_arg0_arg1[verb_obj['verb']].append((arg0, arg1))
    print('num_caption_with_verb:',num_caption_with_verb)
    print('num_caption_without_verb:',num_caption_without_verb)

    with open('verbs_to_frequency.json', 'w') as out:
        json.dump(verbs_to_frequency, out, indent=4)

    with open('verbs_to_arg0_arg1.json', 'w') as out:
        json.dump(verbs_to_arg0_arg1, out, indent=4)
