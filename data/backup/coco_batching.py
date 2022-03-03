import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from data.utils import pre_caption

def compute_embeddings(image_id_2_data, model, processor, device, embedding_mode = 'orignal'):
    image_ids = []
    image_embeddings = []
    text_embeddings = []
    print('computing embeddings...')
    if embedding_mode == 'original':
        for key in tqdm(image_id_2_data.keys()):
            image = Image.open(image_id_2_data[key]['image_path']).convert('RGB')
            texts = image_id_2_data[key]['captions']
            inputs = processor(
                text=texts, images=image, return_tensors="pt", padding=True
            ).to(device)
            outputs = model(**inputs)
            im_emb = outputs.image_embeds # (1,512)
            txt_emb = outputs.text_embeds # (m,512) m is the number of captions corresponding to this image
            image_embeddings.append(im_emb[0].detach().cpu().numpy())
            text_embeddings.append(txt_emb.detach().cpu().numpy())
            image_ids.append(key)
    elif embedding_mode == 'srl_verb':
        print('using srl_verbs to compute text embedding...')
        for key in tqdm(image_id_2_data.keys()):
            image = Image.open(image_id_2_data[key]['image_path']).convert('RGB')
            texts = []
            raw_text_captions = []
            for caption_obj in image_id_2_data[key]['captions']:
                raw_text_captions.append(caption_obj['text'])
                # print(caption_obj)
                assert 'verbs' in caption_obj
                verb_list = [] 
                for verb_obj in caption_obj['verbs']:
                    # print(verb_obj)
                    verb = verb_obj['verb']
                    if verb.lower() not in ['am','is','are','was','were','being','be']:
                        verb_list.append(verb.strip())
                if verb_list:
                    verb_string = ', '.join(verb_list)
                    texts.append(verb_string)
                else:
                    texts.append(caption_obj['text'])
            image_id_2_data[key]['captions'] = raw_text_captions
            # print(texts)
            inputs = processor( 
                text=texts, images=image, return_tensors="pt", padding=True
            ).to(device)
            outputs = model(**inputs)
            im_emb = outputs.image_embeds # (1,512)
            txt_emb = outputs.text_embeds # (m,512) m is the number of captions corresponding to this image
            image_embeddings.append(im_emb[0].detach().cpu().numpy())
            text_embeddings.append(txt_emb.detach().cpu().numpy())
            image_ids.append(key)
                  
    return image_embeddings, text_embeddings, image_ids

def get_groups_coco_train_v_sim_t_dissim_clip(model, processor, image_id_2_data, k1, k2, device, embedding_mode = 'original'):
    ''' grouping image-text by: first group into visually similar groups, then group into textually similar groups
        parameters:
        -----------
        image_id_2_data (dict): {"<image_id>":{"image_path":str,"captions":list(str)}...}
        embedding_mode (str): decides what embedding to use, choose from [
            'original' # use the original caption text for computing text embedding 
            'srl_verb' # use only srl verbs to compute text embedding
        ] 
    '''
    # compute image and text embeddings
    left_out_ids = [] # storing image_ids from those v_groups that has too little samples to be clustered in the second step
    image_embeddings, text_embeddings, image_ids = compute_embeddings(image_id_2_data, model, processor, device, embedding_mode = embedding_mode)

    assert len(image_embeddings) > k1
    image_embeddings = np.array(image_embeddings)
    
    # step1: cluster image embeddings
    print("clustering visual embeddings...")
    kmeans_k1 = KMeans(n_clusters=k1, random_state=0).fit(image_embeddings)
    labels_k1 = kmeans_k1.labels_
    v_groups = []
    for i in range(k1):
        indices_group_i = np.where(labels_k1 == i)[0]
        v_groups.append(indices_group_i)

    # step2: cluster text embeddings
    print("clustering textual embeddings...")
    v_t_groups = []
    for v_group in v_groups:
        text_embeddings_group_i = [text_embeddings[idx][0] for idx in v_group] # FIXME: currently only use the first caption for each image as its text representation 
        if len(text_embeddings_group_i) < k2:
            left_out_ids += [image_ids[idx] for idx in v_group]
            continue
        kmeans_k2 = KMeans(n_clusters=k2, random_state=0).fit(text_embeddings_group_i)
        labels_k2 = kmeans_k2.labels_
        t_groups = []
        for j in range(k2):
            indices_group_j = np.where(labels_k2 == j)[0]
            t_groups.append([v_group[jj] for jj in indices_group_j])
        v_t_groups.append(t_groups)

    # output grouping with image_id
    image_id_groups = v_t_groups.copy()
    is_visited = set()
    for i in range(len(v_t_groups)):
        for j in range(k2):
            for m in range(len(v_t_groups[i][j])):
                idx = v_t_groups[i][j][m]
                imid = image_ids[idx]
                assert imid not in is_visited
                is_visited.add(imid) 
                image_id_groups[i][j][m] = imid
    # sanity check we included all image_ids:
    for l_o_id in left_out_ids:
        assert l_o_id not in is_visited
        is_visited.add(l_o_id)
    assert is_visited == set(image_ids)
    print(f'expected num t_v_groups:{k1*k2}, got num t_v_groups:{len(image_id_groups)*k2} | batch size:{k2}')

    # # vis
    # visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization/srl', n = 'all', order = 'v_t')
    
    return image_id_groups, left_out_ids

def get_groups_coco_train_t_sim_v_dissim_clip(model, processor, image_id_2_data, k1, k2, device, embedding_mode = 'original'):
    '''grouping image-text by: first group into visually similar groups, then group into textually similar groups
        parameters:
        -----------
        image_id_2_data (dict): {"<image_id>":{"image_path":str,"captions":list(str)}...}
        embedding_mode (str): decides what embedding to use, choose from [
            'original' # use the original caption text for computing text embedding 
            'srl_verb' # use only srl verbs to compute text embedding
        ] 
    '''

    # compute image and text embeddings
    left_out_ids = [] # storing image_ids from those v_groups that has too little samples to be clustered in the second step
    image_embeddings, text_embeddings, image_ids = compute_embeddings(image_id_2_data, model, processor, device, embedding_mode = embedding_mode)

    assert len(image_embeddings) > k1
    assert len(image_embeddings) == len(text_embeddings)

    # step1: cluster textual embedding
    print("clustering textual embeddings...")
    text_embeddings_first_caption = np.array([t[0] for t in text_embeddings]) # use the first caption as text representation
    kmeans_k1 = KMeans(n_clusters=k1, random_state=0).fit(text_embeddings_first_caption)
    labels_k1 = kmeans_k1.labels_
    t_groups = []
    for i in range(k1):
        indices_group_i = np.where(labels_k1 == i)[0]
        t_groups.append(indices_group_i)
        
    # step 2: cluster image embedding
    print("clustering visual embeddings...")
    t_v_groups = []
    for t_group in t_groups:
        visual_embeddings_group_i = [image_embeddings[idx] for idx in t_group] 
        if len(visual_embeddings_group_i) < k2:
            left_out_ids += [image_ids[idx] for idx in t_group]
            continue
        kmeans_k2 = KMeans(n_clusters=k2, random_state=0).fit(visual_embeddings_group_i)
        labels_k2 = kmeans_k2.labels_
        v_groups = []
        for j in range(k2):
            indices_group_j = np.where(labels_k2 == j)[0]
            v_groups.append([t_group[jj] for jj in indices_group_j])
        t_v_groups.append(v_groups)

    image_id_groups = t_v_groups.copy()
    is_visited = set()
    for i in range(len(t_v_groups)):
        for j in range(k2):
            for m in range(len(t_v_groups[i][j])):
                idx = t_v_groups[i][j][m]
                imid = image_ids[idx]
                assert imid not in is_visited
                is_visited.add(imid) 
                image_id_groups[i][j][m] = imid
    # sanity check we included all image_ids:
    for l_o_id in left_out_ids:
        assert l_o_id not in is_visited
        is_visited.add(l_o_id)
    assert is_visited == set(image_ids)
    print(f'expected num t_v_groups:{k1*k2}, got num t_v_groups:{len(image_id_groups)*k2} | batch size:{k2}')
    
    # # vis
    # visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization/srl', n = 'all', order = 't_v')
    return image_id_groups, left_out_ids

def get_batches_coco_train_random(image_id_2_data, batch_size):
    all_pairs = []
    for imid,data in image_id_2_data.items():
        for caption_idx in range(len(data['captions'])):
            all_pairs.append((imid,caption_idx))
    random.shuffle(all_pairs)
    batches = []
    for i in range(len(all_pairs)//batch_size + 1):
        batches.append(all_pairs[i*batch_size:(i+1)*batch_size])
    return batches
    
# main function performing batching
def batching(image_root, ann_json, batching_config, device, output_dir = None):
    ''' current batching_config fields:
        -------------------------------
            k1 (int): expected step 1 group number, when batching_root not exist, k1 should be greater than 0
            k2 (int): expected batch number (per gpu), when batching_root not exist, k2 should be greater than 0
            batch_size (int): if k2 is not -1, batch_size == k2
            grouping_function (str): options for grouping functions, choose from ['clip','blip','random']
            order (str): choose from ["v_t", "t_v"]
            mode (str): options for sampling, choose from 
                [
                    "grouped_only": return a subset of the training set with strictly grouped samples
                    "grouped_only_shuffle": return a subset of the training set with strictly grouped samples, but reshuffled
                    "all_duplicate": return a full training set by duplicating image-text pairs that run short during batch sampling within one group
                    "all_add_as_random": return a full training set, which consist of strictly grouped samples and randomly grouped samples
                ]
            model_ckpt (str): path to pretrained model checkpoint if necessary | or model name from huggingface, e.g., openai/clip-vit-base-patch32
    '''
    ### helper functions
    def _add_remaining_to_random_pool(remaining_image_ids, image_id_2_caption_idx, image_id_2_data, random_pool):
        '''helper function for mode: all_add_as_random; this function add all pairs remains in first_level group that cannot 
            be grouped into a flat list to be further randomly batched later 
        '''
        for im_id in remaining_image_ids:
            while image_id_2_caption_idx[im_id] < len(image_id_2_data[im_id]['captions']):
                random_pool.append((im_id, image_id_2_caption_idx[im_id]))
                image_id_2_caption_idx[im_id] += 1
    
    def _sample_from_bi_level_groups_without_duplication(v_group, inputs, batch_size, image_id_2_caption_idx, image_id_2_data, random_pool):
        assert batch_size == len(v_group) 
        while True:
            # check if there are more valid batches
            valid = True
            for t in range(batch_size):
                if v_group[t] == []:
                    valid = False
                    break
            if not valid:
                flat_v_group_remaining = [item for sublist in v_group for item in sublist]
                _add_remaining_to_random_pool(flat_v_group_remaining,image_id_2_caption_idx,image_id_2_data,random_pool)
                break
            else:
                batch = []
                for t in range(batch_size):
                    rand_idx = np.random.randint(low=0,high=len(v_group[t]))
                    im_id = v_group[t][rand_idx]
                    caption_idx = image_id_2_caption_idx[im_id]
                    batch.append((im_id,caption_idx))
                    # check if all captions has been used up for the picked image_id in this t_group
                    image_id_2_caption_idx[im_id] += 1
                    if image_id_2_caption_idx[im_id] == len(image_id_2_data[im_id]['captions']):
                        v_group[t].pop(rand_idx)
                inputs.append(batch)

    def _sample_from_bi_level_groups_with_duplication(v_group, inputs, batch_size, image_id_2_data):
        assert batch_size == len(v_group)
        # get image_text pairs for each t_group
        image_textidx_pairs = []
        max_t_group_len = 0
        for t_group in v_group:
            flattened_t_group = []
            for im_id in t_group:
                flattened_t_group += [(im_id,caption_idx) for caption_idx in range(len(image_id_2_data[im_id]['captions']))]
            random.shuffle(flattened_t_group)
            image_textidx_pairs.append(flattened_t_group)
            if max_t_group_len < len(flattened_t_group):
                max_t_group_len = len(flattened_t_group)
        # duplicate and padding to the same length
        for t in range(len(image_textidx_pairs)):
            while len(image_textidx_pairs[t]) < max_t_group_len:
                image_textidx_pairs[t] += image_textidx_pairs[t]
            image_textidx_pairs[t] = image_textidx_pairs[t][:max_t_group_len]
            assert len(image_textidx_pairs[t]) == max_t_group_len
            random.shuffle(image_textidx_pairs[t])
        # sample batches
        assert batch_size == len(image_textidx_pairs)
        for j in range(max_t_group_len):
            inputs.append([image_textidx_pairs[t][j] for t in range(batch_size)])
    ####################
    
    """ prepare image_id_2_data """
    # load ann json
    ann = json.load(open(ann_json))
    # get image_id_2_data dict
    image_id_2_data = {} # {"<image_id>":{"image_path":str,"captions":list(str)}...}
    for i_t_pair in ann:
        image_id = i_t_pair["image_id"]
        caption = i_t_pair["caption"]
        image_path = os.path.join(image_root, i_t_pair["image"])
        if image_id not in image_id_2_data:
            image_id_2_data[image_id] = {
                'image_path': image_path,
                'captions':[caption]
            }
        else:
            image_id_2_data[image_id]['captions'].append(caption)

    
    """ get batches according to grouping_function """
    batch_size = batching_config['batch_size']    
    model_ckpt =  batching_config['model_ckpt']
    
    if batching_config['grouping_function'] == "random":
        inputs = get_batches_coco_train_random(image_id_2_data, batch_size)
    
    elif batching_config['grouping_function'] == "clip":
        k1,k2 = batching_config['k1'],batching_config['k2']
        mode = batching_config['mode']
        if 'embedding_mode' in batching_config:
            embedding_mode = batching_config['embedding_mode']
        else:
            embedding_mode = 'original'
        # set up model
        model = CLIPModel.from_pretrained(model_ckpt)
        model.to(device)
        processor = CLIPProcessor.from_pretrained(model_ckpt)

        if batching_config['order'] == 'v_t':
            image_id_groups, left_out_ids = get_groups_coco_train_v_sim_t_dissim_clip(model, processor, image_id_2_data, k1, k2, device, embedding_mode)
        elif batching_config['order'] == 't_v':
            image_id_groups, left_out_ids = get_groups_coco_train_t_sim_v_dissim_clip(model, processor, image_id_2_data, k1, k2, device, embedding_mode)
        else:
            raise NotImplementedError("unkown order")

        # start sampling
        grouped_inputs = []
        image_id_2_caption_idx = defaultdict(int)
        random_pool = []
        for v_group in image_id_groups:
            if mode == 'all_duplicate':
                _sample_from_bi_level_groups_with_duplication(v_group, grouped_inputs, batch_size, image_id_2_data)
            elif mode in ["grouped_only","grouped_only_shuffle","all_add_as_random"]:
                _sample_from_bi_level_groups_without_duplication(v_group, grouped_inputs, batch_size, image_id_2_caption_idx, image_id_2_data, random_pool)
            else:
                raise NotImplementedError("unkown mode")

        # add left out ids into random pool
        print('random pool size before adding left out:', len(random_pool))
        _add_remaining_to_random_pool(left_out_ids, image_id_2_caption_idx, image_id_2_data, random_pool)
        print('random pool size after adding left out:', len(random_pool))

        print('num grouped inputs:', len(grouped_inputs))
        # randomly batch datapoints in random pool
        random.shuffle(random_pool)
        random_pool_batches = []
        for i in range(len(random_pool)//batch_size + 1):
            random_pool_batches.append(random_pool[i*batch_size:(i+1)*batch_size])
        print('num randomly inputs:',len(random_pool_batches))
        
        total_num_pairs = 0
        for key,item in image_id_2_data.items():
            total_num_pairs += len(item['captions'])
        print('total num unique image-text pairs:',total_num_pairs)

        if mode == 'grouped_only':
            inputs = grouped_inputs
        elif mode == 'grouped_only_shuffle':
            flattened = [pair for batch in grouped_inputs for pair in batch]
            random.shuffle(flattened)
            inputs = []
            for i in range(len(flattened)//batch_size + 1):
                inputs.append(flattened[i*batch_size:(i+1)*batch_size])
        elif mode == 'all_add_as_random':
            inputs = grouped_inputs + random_pool_batches
        elif mode == 'all_duplicate':
            inputs = grouped_inputs + random_pool_batches # here the random_pool_batches only contains the batches drawn from left_out_ids

    elif batching_config['grouping_function'] == "blip":
        raise NotImplementedError("TODO: implement grouping function using blip")
    else:
        raise NotImplementedError("unkown grouping function")
    
    print("num of batches:",len(inputs))
    
    batching_config['total_num_batches'] = len(inputs)
    batching_config['total_num_image_text_pairs'] = sum([len(b) for b in inputs])

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir,'batches.json'), 'w') as out:
            json.dump(inputs, out, indent = 4)
        with open(os.path.join(output_dir,'image_id_2_data.json'), 'w') as out:
            json.dump(image_id_2_data, out, indent = 4)
        with open(os.path.join(output_dir,'batching_config.json'), 'w') as out:
            json.dump(batching_config, out, indent = 4)

    return inputs, image_id_2_data
    

from textwrap import wrap

def visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization', n = 'all', order = 'v_t'):
    '''order: choose from ['v_t','t_v']'''
    ### helper function
    def get_max_t_group_len(t_groups):
        max_len = 1
        for t_group in t_groups:
            max_len = max(max_len, len(t_group))
        return max_len
    ###################

    k1 = len(image_id_groups)
    k2 = len(image_id_groups[0])

    if order == 'v_t':
        output_dir = os.path.join(output_root,f'v_{k1}_t_{k2}')
    elif order == 't_v':
        output_dir = os.path.join(output_root,f't_{k1}_v_{k2}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if n == 'all':
        n = k1-1

    for i in range(k1):
        v_group = image_id_groups[i]

        num_row = len(v_group)
        num_col = get_max_t_group_len(v_group)
        fig, axs = plt.subplots(num_row, num_col, figsize=(num_col*1.5,num_row*2))
        if order == 'v_t':
            fig.suptitle(f'v_group {i}')
        elif order == 't_v':
            fig.suptitle(f't_group {i}')

        print('num_row:',num_row,'num_col:',num_col)
        for t in range(num_row):
            t_group = v_group[t]
            if num_col > 1:
                for j in range(num_col):
                    axs[t][j].set_axis_off()
                    if j < len(t_group):
                        image_id = t_group[j]
                        captions = image_id_2_data[image_id]['captions']
                        image_path = image_id_2_data[image_id]['image_path']
                        axs[t][j].imshow(Image.open(image_path))
                        axs[t][j].set_title("\n".join(wrap(f"{captions[0]}", 30)), size=6)
            else:
                axs[t].set_axis_off()
                image_id = t_group[0]
                captions = image_id_2_data[image_id]['captions']
                image_path = image_id_2_data[image_id]['image_path']
                axs[t].imshow(Image.open(image_path))
                axs[t].set_title("\n".join(wrap(f"{captions[0]}", 30)), size=8)
        fig.tight_layout()
        if order == 'v_t':
            fig.savefig(os.path.join(output_dir, f'visual_group_{i}.png'))
        elif order == 't_v':
            fig.savefig(os.path.join(output_dir, f'textual_group_{i}.png'))

        if i == n:
            break
