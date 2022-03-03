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
import re 

import matplotlib.pyplot as plt


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def get_groups_coco_train_v_sim_t_dissim_clip(image_root, ann_json, k1, k2, device):
    '''grouping image-text by: first group into visually similar groups, then group into textually similar groups'''

    # set up clip model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ann = json.load(open(ann_json))

    image_id_2_data = {}
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
    print(len(image_id_2_data))

    image_ids = []
    image_embeddings = []
    text_embeddings = []

    for key in tqdm(image_id_2_data.keys()):
        image = Image.open(image_id_2_data[key]['image_path']).convert('RGB')
        texts = image_id_2_data[key]['captions']
        # print(key, image.size, image.mode)
        inputs = processor(
            text=texts, images=image, return_tensors="pt", padding=True
        ).to(device)
        outputs = model(**inputs)
        im_emb = outputs.image_embeds # (1,512)
        txt_emb = outputs.text_embeds # (m,512) m is the number of captions corresponding to this image
        # print(im_emb.size())
        # print(txt_emb.size())
        image_embeddings.append(im_emb[0].detach().cpu().numpy())
        text_embeddings.append(txt_emb.detach().cpu().numpy())
        image_ids.append(key)

    assert len(image_embeddings) > k1
    image_embeddings = np.array(image_embeddings)
    # step1: cluster images 
    kmeans_k1 = KMeans(n_clusters=k1, random_state=0).fit(image_embeddings)
    labels_k1 = kmeans_k1.labels_
    v_groups = []
    for i in range(k1):
        indices_group_i = np.where(labels_k1 == i)[0]
        v_groups.append(indices_group_i)
    print('num of v_group:',len(v_groups))
    print('--------------')
    v_t_groups = []
    for v_group in v_groups:
        text_embeddings_group_i = [text_embeddings[idx][0] for idx in v_group] # FIXME: use the first caption for each image as its text representation 
        if len(text_embeddings_group_i) < k2:
            continue
        kmeans_k2 = KMeans(n_clusters=k2, random_state=0).fit(text_embeddings_group_i) # TODO: k2 can be larger than len(text_embeddings_group_i)
        labels_k2 = kmeans_k2.labels_
        t_groups = []
        for j in range(k2):
            indices_group_j = np.where(labels_k2 == j)[0]
            t_groups.append([v_group[jj] for jj in indices_group_j])
        # print(t_groups)
        v_t_groups.append(t_groups)
    print('--------------')
    # print(v_t_groups)
    ret_groups = v_t_groups.copy()
    is_visited = set()
    for i in range(len(v_t_groups)):
        for j in range(k2):
            for m in range(len(v_t_groups[i][j])):
                idx = v_t_groups[i][j][m]
                assert idx not in is_visited
                is_visited.add(idx) 
                ret_groups[i][j][m] = image_ids[idx]
    print('num v_t_groups:', len(ret_groups))
    return ret_groups, image_id_2_data

def get_groups_coco_train_t_sim_v_dissim_clip(image_root, ann_json, k1, k2, device):
    '''grouping image-text by: first group into visually similar groups, then group into textually similar groups'''

    # set up clip model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ann = json.load(open(ann_json))

    image_id_2_data = {}
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
    print(len(image_id_2_data))

    image_ids = []
    image_embeddings = []
    text_embeddings = []

    for key in tqdm(image_id_2_data.keys()):
        image = Image.open(image_id_2_data[key]['image_path']).convert('RGB')
        texts = image_id_2_data[key]['captions']
        # print(key, image.size, image.mode)
        inputs = processor(
            text=texts, images=image, return_tensors="pt", padding=True
        ).to(device)
        outputs = model(**inputs)
        im_emb = outputs.image_embeds # (1,512)
        txt_emb = outputs.text_embeds # (m,512) m is the number of captions corresponding to this image
        # print(im_emb.size())
        # print(txt_emb.size())
        image_embeddings.append(im_emb[0].detach().cpu().numpy())
        text_embeddings.append(txt_emb.detach().cpu().numpy())
        image_ids.append(key)

    assert len(image_embeddings) > k1
    assert len(image_embeddings) == len(text_embeddings)

    # step1: cluster textual embedding
    text_embeddings_first_caption = np.array([t[0] for t in text_embeddings]) # use the first caption as text representation
    # image_embeddings = np.array(image_embeddings)
    kmeans_k1 = KMeans(n_clusters=k1, random_state=0).fit(text_embeddings_first_caption)
    labels_k1 = kmeans_k1.labels_
    t_groups = []
    for i in range(k1):
        indices_group_i = np.where(labels_k1 == i)[0]
        t_groups.append(indices_group_i)
    print('num of t_group:',len(t_groups))
    print('--------------')
    # step 2: cluster image embedding
    t_v_groups = []
    for t_group in t_groups:
        visual_embeddings_group_i = [image_embeddings[idx] for idx in t_group] 
        if len(visual_embeddings_group_i) < k2:
            continue
        kmeans_k2 = KMeans(n_clusters=k2, random_state=0).fit(visual_embeddings_group_i)
        labels_k2 = kmeans_k2.labels_
        v_groups = []
        for j in range(k2):
            indices_group_j = np.where(labels_k2 == j)[0]
            v_groups.append([t_group[jj] for jj in indices_group_j])
        # print(t_groups)
        t_v_groups.append(v_groups)

    ret_groups = t_v_groups.copy()
    is_visited = set()
    for i in range(len(t_v_groups)):
        for j in range(k2):
            for m in range(len(t_v_groups[i][j])):
                idx = t_v_groups[i][j][m]
                assert idx not in is_visited
                is_visited.add(idx) 
                ret_groups[i][j][m] = image_ids[idx]
    print('num t_v_groups:', len(ret_groups))
    print()
    return ret_groups, image_id_2_data


def get_subset_coco_train_random(ann_json, n, k2, output_dir = '.'):
    ann = json.load(open(ann_json))
    random.shuffle(ann)
    ann = ann[:n*k2]
    print('num of samples:',len(ann))
    with open(os.path.join(output_dir,f'random_subset_{n}-{k2}.json'), 'w') as out:
        json.dump(ann, out, indent=4)

def get_max_v_t_group_len(image_id_groups):
    max_len = 1
    for i in range(len(image_id_groups)):
        for j in range(len(image_id_groups[0])):
            max_len = max(max_len, len(image_id_groups[i][j]))
    return max_len

def get_max_t_group_len(t_groups):
    max_len = 1
    for t_group in t_groups:
        max_len = max(max_len, len(t_group))
    return max_len


from textwrap import wrap

def visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization', n = 'all', order = 'v_t'):
    # order: choose from ['v_t','t_v']
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

# custom dataset
class coco_karpathy_train_with_grouping(Dataset):
    def __init__(self, transform, image_root, ann_json, image_id_groups = None, image_id_2_data = None, k1 = -1, k2 = -1, max_words=30, prompt='', if_use_all = True):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_json (string): path to store the annotation file
        '''        

        # get groups        
        if image_id_groups is None or image_id_2_data is None:
            assert k1 != -1 and k2 != -1
            # TODO: call:
            # self.image_id_groups, self.image_id_2_data = get_groups_coco_train_v_sim_t_dissim(image_root, ann_json, k1, k2) 
        else:
            self.image_id_groups = json.load(open(image_id_groups))
            self.image_id_2_data = json.load(open(image_id_2_data))
            
        self.batch_size = len(image_id_groups[0])

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt

        self.img_ids = {}  
        n = 0
        for img_id, data in self.image_id_2_data.items():
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        # get inputs
        grouped_inputs, random_inputs = self._get_inputs()
        if if_use_all:
            self.inputs = grouped_inputs + random_inputs
        else:
            self.inputs = grouped_inputs
        print('total batch num:', len(self.inputs))
        print('batch size:',self.batch_size)

    def _get_inputs(self):

        def add_remaining_to_random_pool(v_group_remaining,image_id_2_caption_idx,random_pool):
            flat_v_group_remaining = [item for sublist in v_group_remaining for item in sublist]
            for im_id in flat_v_group_remaining:
                while image_id_2_caption_idx[im_id] < len(self.image_id_2_data[im_id]['captions']):
                    random_pool.append((im_id, image_id_2_caption_idx[im_id]))
                    image_id_2_caption_idx[im_id] += 1

        inputs = []
        image_id_2_caption_idx = defaultdict(int)
        random_pool = []
        image_id_groups_copy = self.image_id_groups.copy()
        for v_group in image_id_groups_copy:
            assert self.batch_size == len(v_group) 
            while True:
                # check if there are more valid batches
                valid = True
                for t in range(self.batch_size):
                    if v_group[t] == []:
                        valid = False
                        break
                if not valid:
                    add_remaining_to_random_pool(v_group,image_id_2_caption_idx,random_pool)
                    break
                else:
                    batch = []
                    for t in range(self.batch_size):
                        rand_idx = np.random.randint(low=0,high=len(v_group[t]))
                        im_id = v_group[t][rand_idx]
                        caption_idx = image_id_2_caption_idx[im_id]
                        batch.append((im_id,caption_idx))
                        # check if all captions has been used up for the picked image_id in this t_group
                        image_id_2_caption_idx[im_id] += 1
                        if image_id_2_caption_idx[im_id] == len(self.image_id_2_data[im_id]['captions']):
                            v_group[t].pop(rand_idx)
                    inputs.append(batch)

        print('num grouped inputs:', len(inputs))
        random.shuffle(random_pool)
        random_pool_batches = []
        for i in range(len(random_pool)//self.batch_size):
            random_pool_batches.append(random_pool[i*self.batch_size:(i+1)*self.batch_size])
        print('num randomly inputs:',len(random_pool_batches))
        total_num_pairs = 0
        for key,item in self.image_id_2_data.items():
            total_num_pairs += len(item['captions'])
        print('num total inputs:',total_num_pairs)
        return inputs, random_pool_batches

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):    
        batch = self.inputs[index]
        print(batch)
        images = []
        captions = []
        img_idxs = []
        for pair in batch:
            im_id = pair[0]
            caption_idx = pair[1]
            image_path = os.path.join(self.image_root,self.image_id_2_data[im_id]['image_path'])        
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:   
                image = self.transform(image)
            images.append(image)

            caption = self.prompt+pre_caption(self.image_id_2_data[im_id]['captions'][caption_idx], self.max_words)
            captions.append(caption)
            img_idxs.append(self.img_ids[im_id])

        ## visualize
        # self._vis_batch(images, captions, index)

        return images, captions, img_idxs
    
    def _vis_batch(self, images, captions, index):
        fig, axs = plt.subplots(1, self.batch_size, figsize=(20,8))
        fig.suptitle(f'vis batch')
        for i in range(self.batch_size):
            axs[i].set_axis_off()
            axs[i].imshow(images[i])
            axs[i].set_title("\n".join(wrap(f"{captions[i]}", 30)), size=6)
        fig.tight_layout()
        fig.savefig(f'vis_batch_{index}.png')


if __name__ == '__main__':
    
    random.seed(42)
    np.random.seed(42)

    
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    '''Test custom subset v-sim-t-dissim'''
        # image_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
        # ann_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
        
        # k1 = 4 # number of visually similar groups
        # k2 = 4 # number of textually similar groups within each visually similar groups
        # # for k1,k2 in [(4,4),(3,5),(6,4),(5,4),(2,6)]:
        # image_id_groups, image_id_2_data = get_groups_coco_train_v_sim_t_dissim(image_root, ann_json, k1, k2)
        # # visualize
        # visualize_groups(image_id_groups, image_id_2_data)

        # output_grouping_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping'
        # output_grouping_dir = os.path.join(output_grouping_root,f'coco_retrieval_subset_custom-v_{k1}_t_{k2}')
        # if not os.path.exists(output_grouping_dir):
        #     os.makedirs(output_grouping_dir)
        # with open(os.path.join(output_grouping_dir,'image_id_groups.json'), 'w') as out:
        #     json.dump(image_id_groups, out, indent = 4)
        # with open(os.path.join(output_grouping_dir,'image_id_2_data.json'), 'w') as out:
        #     json.dump(image_id_2_data, out, indent = 4)
    
    '''Test custom subset t-sim-v-dissim'''
        # image_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
        # ann_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
        
        # k1 = 4 # number of visually similar groups
        # k2 = 4 # number of textually similar groups within each visually similar groups
        # # for k1,k2 in [(4,4),(3,5),(6,4),(5,4),(2,6)]:
        # image_id_groups, image_id_2_data = get_groups_coco_train_t_sim_v_dissim(image_root, ann_json, k1, k2)
        # # visualize
        # print('saving visualization...')
        # visualize_groups(image_id_groups, image_id_2_data, output_root='./visualization/coco_subset_custom_t-sim-v-dissim', order = 't_v')
        # print('saving grouping...')
        # output_grouping_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping'
        # output_grouping_dir = os.path.join(output_grouping_root,f'coco_retrieval_subset_custom-t_{k1}_v_{k2}')
        # if not os.path.exists(output_grouping_dir):
        #     os.makedirs(output_grouping_dir)
        # with open(os.path.join(output_grouping_dir,'image_id_groups.json'), 'w') as out:
        #     json.dump(image_id_groups, out, indent = 4)
        # with open(os.path.join(output_grouping_dir,'image_id_2_data.json'), 'w') as out:
        #     json.dump(image_id_2_data, out, indent = 4)
    
    '''Test custom subset v-t-concat'''
        # image_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
        # ann_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
        
        # k = 10 # number of groups
        # image_id_groups, image_id_2_data = get_groups_coco_train_v_t_concat(image_root, ann_json, k)
        # # visualize
        # visualize_groups_v_t_concat(image_id_groups, image_id_2_data, output_root='./visualization/coco_subset_custom-v_t_concat')

        # output_grouping_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping'
        # output_grouping_dir = os.path.join(output_grouping_root,f'coco_retrieval_subset_custom-v_t_concat-k_{k}')
        # if not os.path.exists(output_grouping_dir):
        #     os.makedirs(output_grouping_dir)
        # with open(os.path.join(output_grouping_dir,'image_id_groups.json'), 'w') as out:
        #     json.dump(image_id_groups, out, indent = 4)
        # with open(os.path.join(output_grouping_dir,'image_id_2_data.json'), 'w') as out:
        #     json.dump(image_id_2_data, out, indent = 4)
        
    '''coco retrieval train set v -> t'''
        # image_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/images'
        # ann_json = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/annotations/coco_karpathy_train.json'
        
        # k1 = 1000 # number of visually similar groups
        # k2 = 8 # number of textually similar groups within each visually similar groups
        
        # image_id_groups, image_id_2_data = get_groups_coco_train_v_sim_t_dissim(image_root, ann_json, k1, k2)
        # # visualize_groups(image_id_groups, image_id_2_data)

        # output_grouping_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping'
        # output_grouping_dir = os.path.join(output_grouping_root,f'coco_retrieval_trainset-v_{k1}_t_{k2}')
        # if not os.path.exists(output_grouping_dir):
        #     os.makedirs(output_grouping_dir)
        # with open(os.path.join(output_grouping_dir,'image_id_groups.json'), 'w') as out:
        #     json.dump(image_id_groups, out, indent = 4)
        # with open(os.path.join(output_grouping_dir,'image_id_2_data.json'), 'w') as out:
        #     json.dump(image_id_2_data, out, indent = 4)

    '''coco retrieval train set t -> v'''
    image_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/images'
    ann_json = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/annotations/coco_karpathy_train.json'
    
    k1 = 500 # number of visually similar groups
    k2 = 8 # number of textually similar groups within each visually similar groups
    
    image_id_groups, image_id_2_data = get_groups_coco_train_t_sim_v_dissim(image_root, ann_json, k1, k2)
    
    visualize_groups(image_id_groups, image_id_2_data, output_root='./visualization/coco_full_trainset', n=10, order = 't_v')

    output_grouping_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping'
    output_grouping_dir = os.path.join(output_grouping_root,f'coco_retrieval_trainset-t_{k1}_v_{k2}')
    if not os.path.exists(output_grouping_dir):
        os.makedirs(output_grouping_dir)
    with open(os.path.join(output_grouping_dir,'image_id_groups.json'), 'w') as out:
        json.dump(image_id_groups, out, indent = 4)
    with open(os.path.join(output_grouping_dir,'image_id_2_data.json'), 'w') as out:
        json.dump(image_id_2_data, out, indent = 4)

    '''visualize only'''
        # image_id_groups_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_500_t_8/image_id_groups.json'
        # image_id_2_data_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_500_t_8/image_id_2_data.json'
        # image_id_groups = json.load(open(image_id_groups_json))
        # image_id_2_data = json.load(open(image_id_2_data_json))
        
        # visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization/coco_full_trainset', n=10)
    
        # image_id_groups_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_1000_t_8/image_id_groups.json'
        # image_id_2_data_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_1000_t_8/image_id_2_data.json'
        # image_id_groups = json.load(open(image_id_groups_json))
        # image_id_2_data = json.load(open(image_id_2_data_json))
        
        # visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization/coco_full_trainset', n=10)
    
    '''test dataset'''
        ## subset
        # image_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
        # ann_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
        
        # image_id_groups_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_subset_custom-v_4_t_4/image_id_groups.json'
        # image_id_2_data_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_subset_custom-v_4_t_4/image_id_2_data.json'
        # image_id_groups = json.load(open(image_id_groups_json))
        # image_id_2_data = json.load(open(image_id_2_data_json))

        ## full train set
        # image_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/images'
        # ann_json = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/annotations/coco_karpathy_train.json'
    
        # image_id_groups_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_500_t_8/image_id_groups.json'
        # image_id_2_data_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/grouping/coco_retrieval_trainset-v_500_t_8/image_id_2_data.json'
        # image_id_groups = json.load(open(image_id_groups_json))
        # image_id_2_data = json.load(open(image_id_2_data_json))

        # dataset = coco_karpathy_train_with_grouping(
        #     None, 
        #     image_root, 
        #     ann_json, 
        #     image_id_groups = image_id_groups, 
        #     image_id_2_data = image_id_2_data, 
        #     max_words=30, 
        #     prompt='',
        #     if_use_all = False
        # )
        # print(dataset[0])
        # print(dataset[100])
        # print(dataset[200])
        # print(dataset[300])
        # print(dataset[400])

    '''get random batches'''
        # image_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/images'
        # ann_json = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/annotations/coco_karpathy_train.json'
        # n = 26815
        # get_subset_coco_train_random(ann_json, n, 8)








































########################
## Archive #############
########################
# def get_groups_coco_train_v_t_concat(image_root, ann_json, k):
#     '''grouping image-text by: first group into visually similar groups, then group into textually similar groups'''
    
#     ''' set up device '''
#     # use cuda
#     if torch.cuda.is_available():  
#         dev = "cuda:3" 
#     else:  
#         dev = "cpu"
#     device = torch.device(dev)

#     # set up clip model
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     model.to(device)
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     ann = json.load(open(ann_json))

#     image_id_2_data = {}
#     for i_t_pair in ann:
#         image_id = i_t_pair["image_id"]
#         caption = i_t_pair["caption"]
#         image_path = os.path.join(image_root, i_t_pair["image"])
#         if image_id not in image_id_2_data:
#             image_id_2_data[image_id] = {
#                 'image_path': image_path,
#                 'captions':[caption]
#             }
#         else:
#             image_id_2_data[image_id]['captions'].append(caption)
#     print(len(image_id_2_data))

#     image_ids = []
#     image_embeddings = []
#     text_embeddings = []
#     im_txt_concat_embeddings = []

#     for key in tqdm(image_id_2_data.keys()):
#         image = Image.open(image_id_2_data[key]['image_path']).convert('RGB')
#         texts = image_id_2_data[key]['captions']
#         # print(key, image.size, image.mode)
#         inputs = processor(
#             text=texts, images=image, return_tensors="pt", padding=True
#         ).to(device)
#         outputs = model(**inputs)
#         im_emb = outputs.image_embeds # (1,512)
#         txt_emb = outputs.text_embeds # (m,512) m is the number of captions corresponding to this image
#         # print(im_emb.size())
#         # print(txt_emb.size())
#         image_embeddings.append(im_emb[0].detach().cpu().numpy())
#         text_embeddings.append(txt_emb.detach().cpu().numpy())
#         image_ids.append(key)
#         im_txt_emb = np.concatenate([image_embeddings[-1],text_embeddings[-1][0]]) 
#         # print(im_txt_emb.shape)
#         im_txt_concat_embeddings.append(im_txt_emb)

#     im_txt_concat_embeddings = np.array(im_txt_concat_embeddings)
#     # cluster image-text 
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(im_txt_concat_embeddings)
#     labels = kmeans.labels_
#     groups = []
#     for i in range(k):
#         indices_group_i = np.where(labels == i)[0]
#         groups.append(indices_group_i)
#     print(groups)
#     print('--------------')
#     ret_groups = []
#     is_visited = set()
#     for i in range(k):
#         g = []
#         for j in range(len(groups[i])):
#             idx = groups[i][j]
#             assert idx not in is_visited
#             is_visited.add(idx)
#             g.append(image_ids[idx])
#         ret_groups.append(g)
#     print(ret_groups)
#     return ret_groups, image_id_2_data


# def visualize_groups_v_t_concat(image_id_groups, image_id_2_data, output_root = './visualization', n = 'all'):
#     k = len(image_id_groups)

#     output_dir = os.path.join(output_root,f'k_{k}')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     if n == 'all':
#         n = k-1

#     for i in range(k):
#         v_t_concat_group = image_id_groups[i]

#         num_col = 10
#         num_row = (len(v_t_concat_group) // 10) + 1 
#         fig, axs = plt.subplots(num_row, num_col, figsize=(num_col*1.5,num_row*2))
#         fig.suptitle(f'v_t_concat_group {i}')
#         print('row:',num_row,'col:',num_col)
#         if num_row > 1:
#             for r in range(num_row):
#                 for c in range(num_col):
#                     axs[r][c].set_axis_off()
#                     idx = r*num_col + c
#                     if idx <= len(v_t_concat_group) - 1:
#                         image_id = v_t_concat_group[idx]
#                         captions = image_id_2_data[image_id]['captions']
#                         image_path = image_id_2_data[image_id]['image_path']
#                         axs[r][c].imshow(Image.open(image_path))
#                         axs[r][c].set_title("\n".join(wrap(f"{captions[0]}", 30)), size=6)
#         else:
#             for c in range(num_col):
#                 axs[c].set_axis_off()
#                 idx = c
#                 if idx <= len(v_t_concat_group) - 1:
#                     image_id = v_t_concat_group[idx]
#                     captions = image_id_2_data[image_id]['captions']
#                     image_path = image_id_2_data[image_id]['image_path']
#                     axs[c].imshow(Image.open(image_path))
#                     axs[c].set_title("\n".join(wrap(f"{captions[0]}", 30)), size=6)

#         fig.tight_layout()
#         fig.savefig(os.path.join(output_dir, f'visual_group_{i}.png'))
#         if i == n:
#             quit()


