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

class coco_karpathy_train_grouping(Dataset):
    # TODO
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    

    
def get_groups_coco_train(image_root, ann_json, k1, k2):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:1" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

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
    
    image_ids = []
    image_embeddings = []
    text_embeddings = []

    for key in tqdm(image_id_2_data.keys()):
        image = Image.open(image_id_2_data[key]['image_path'])
        texts = image_id_2_data[key]['captions']
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

    image_embeddings = np.array(image_embeddings)
    # step1: cluster images 
    kmeans_k1 = KMeans(n_clusters=k1, random_state=0).fit(image_embeddings)
    labels_k1 = kmeans_k1.labels_
    v_groups = []
    for i in range(k1):
        indices_group_i = np.where(labels_k1 == i)[0]
        v_groups.append(indices_group_i)
    print(v_groups)
    print('--------------')
    v_t_groups = []
    for v_group in v_groups:
        text_embeddings_group_i = [text_embeddings[idx][0] for idx in v_group] # FIXME: use the first caption for each image as its text representation 
        kmeans_k2 = KMeans(n_clusters=k2, random_state=0).fit(text_embeddings_group_i) # TODO: k2 can be larger than len(text_embeddings_group_i)
        labels_k2 = kmeans_k2.labels_
        t_groups = []
        for j in range(k2):
            indices_group_j = np.where(labels_k2 == j)[0]
            t_groups.append([v_group[jj] for jj in indices_group_j])
        print(t_groups)
        v_t_groups.append(t_groups)
    print('--------------')
    print(v_t_groups)
    ret_groups = v_t_groups.copy()
    is_visited = set()
    for i in range(k1):
        for j in range(k2):
            for m in range(len(v_t_groups[i][j])):
                idx = v_t_groups[i][j][m]
                assert idx not in is_visited
                is_visited.add(idx) 
                ret_groups[i][j][m] = image_ids[idx]
    print(ret_groups)
    return ret_groups, image_id_2_data

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

def visualize_groups(image_id_groups, image_id_2_data, output_root = './visualization'):
    k1 = len(image_id_groups)
    k2 = len(image_id_groups[0])

    output_dir = os.path.join(output_root,f'v_{k1}_t_{k2}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(k1):
        v_group = image_id_groups[i]

        num_row = len(v_group)
        num_col = get_max_t_group_len(v_group)
        fig, axs = plt.subplots(num_row, num_col, figsize=(30,10))
        fig.suptitle(f'v_group {i}')
        # print(num_row,num_col)
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
        fig.savefig(os.path.join(output_dir, f'visual_group_{i}.png'))

if __name__ == '__main__':
    image_root = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
    ann_json = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
    
    k1 = 4 # number of visually similar groups
    k2 = 4 # number of textually similar groups within each visually similar groups
    for k1,k2 in [(4,4),(3,5),(6,4),(5,4),(2,6)]:
        image_id_groups, image_id_2_data = get_groups_coco_train(image_root, ann_json, k1, k2)
        visualize_groups(image_id_groups, image_id_2_data)