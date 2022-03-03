import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

from collections import defaultdict
import numpy as np
import random
import torch


class coco_karpathy_train(Dataset):
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

# custom dataset
class coco_karpathy_train_subset_custom(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'coco_train_random_subset_26815-8.json'
        print('using training ann:', filename)
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        print('total number of samples:', len(self.annotation))
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

class coco_karpathy_train_with_grouping(Dataset):
    def __init__(self, transform, image_root, ann_root, image_id_groups = None, image_id_2_data = None, k1 = -1, k2 = -1, max_words=30, prompt='', if_use_all = True):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): dir to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        download_url(url,ann_root)
        ann_json = os.path.join(ann_root, filename)
        
        # get groups        
        if image_id_groups is None or image_id_2_data is None:
            assert k1 != -1 and k2 != -1
            # TODO: call:
            # self.image_id_groups, self.image_id_2_data = get_groups_coco_train_v_sim_t_dissim(image_root, ann_json, k1, k2) 
        else:
            self.image_id_groups = json.load(open(image_id_groups))
            self.image_id_2_data = json.load(open(image_id_2_data))
            
        self.batch_size = len(self.image_id_groups[0])
        print('batch size:',self.batch_size)

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
            print('including random batches')
            self.inputs = grouped_inputs + random_inputs
        else:
            print('excluding random batches')
            self.inputs = grouped_inputs
        print('total batch num:', len(self.inputs))

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
        images = torch.stack(images)
        img_idxs = torch.LongTensor(img_idxs)
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

class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index

class coco_karpathy_retrieval_eval_subset_custom(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=64):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        print('ann root:', ann_root)
        filenames = {'val':'coco_karpathy_val_subset_custom.json','test':'coco_karpathy_test_subset_custom.json'}
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index