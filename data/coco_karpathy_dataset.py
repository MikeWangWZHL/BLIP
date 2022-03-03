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

from data.coco_batching import batching


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
    def __init__(self, transform, image_root, ann_json = None, ann_root = None, batching_config = None, device = torch.device('cpu'), ratio = 1, max_words=30, prompt=''):        
        '''
            image_root (string): Root directory of images (e.g. coco/images/)
            ann_json (string): path to the annotation file
            ann_root (string): dir to store the official annotation file to be downloaded
            batching_config (dict): input to batching function, example of using clip for bi-level clustering: {
                "embedding_root" (string): path to the precomputed grouping jsons, if not exist, will mkdir and call grouping function on the fly and store jsons at this root
                "batching_root" (string): path to the precomputed grouping jsons, if not exist, will mkdir and call grouping function on the fly and store jsons at this root
                "grouping_function":'clip', 
                "mode":"all_duplicate", 
                "k1":500,
                "k2":8,
                "batch_size":8,
                "order":"v_t",
                "model_ckpt": "openai/clip-vit-base-patch32"
            } # batching_config field descriptions:
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
            ratio (int): if not 1, return a portion of all batched samples
        '''        
        if ann_json is None and ann_root is None:
            raise ValueError('please specify either a specific "ann_json" path or an "ann_root" path that will be used to store official coco train ann json to be downloaded')    
        if ann_json is None:
            url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
            filename = 'coco_karpathy_train.json'
            download_url(url,ann_root)
            ann_json = os.path.join(ann_root, filename)

        # get groups
        batching_root = batching_config['batching_root']
        if not os.path.exists(os.path.join(batching_root,'batches.json')):
            assert batching_config is not None
            self.inputs, self.image_id_2_data = batching(image_root, ann_json, batching_config, device, output_dir = batching_root)
        else:
            print(f'loading precomputed batches from: {batching_root}')
            batching_config = json.load(open(os.path.join(batching_root,'batching_config.json')))
            print(f'batching config:')
            print(batching_config)
            self.inputs = json.load(open(os.path.join(batching_root,'batches.json')))
            self.image_id_2_data = json.load(open(os.path.join(batching_root,'image_id_2_data.json')))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        if ratio != 1:
            print(f"using part of all batches: {ratio}")
            self.inputs = self.inputs[:ratio*len(self.inputs)]
        print()
        print('total batch num:', len(self.inputs))
        print('batch size:',len(self.inputs[0]))

        self.img_ids = {}  
        n = 0
        for batch in self.inputs:
            for pair in batch:
                img_id = pair[0]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

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

class coco_karpathy_retrieval_eval_subset_custom(Dataset):
    def __init__(self, transform, image_root, ann_json, max_words=64):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        self.annotation = json.load(open(ann_json,'r'))
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