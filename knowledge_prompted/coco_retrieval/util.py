import json
import shutil
import os



def load_coco_annotation_json(json_path, num = 'all'):
    anns = json.load(open(json_path))
    if num == 'all':
        num = len(anns)
    return anns[:num]

def copy_images(json_path, image_root, output_dir):
    anns = load_coco_annotation_json(json_path)
    # print(len(anns))
    for item in anns:
        image_path_src = os.path.join(image_root, item['image'])
        image_path_dest = os.path.join(output_dir, os.path.basename(item['image']))
        shutil.copyfile(image_path_src, image_path_dest)
        

def get_subset(num, phases = ['train','val','test']):
    # num: how many samples to sample (start from beginning)
    train_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/original_annotation/coco_karpathy_train.json'
    val_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/original_annotation/coco_karpathy_val.json'
    test_json_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/original_annotation/coco_karpathy_test.json'
    
    train_output_path = './subset_custom_annotation/coco_karpathy_train_subset_custom_500.json'
    val_output_path = './subset_custom_annotation/coco_karpathy_val_subset_custom.json'
    test_output_path = './subset_custom_annotation/coco_karpathy_test_subset_custom.json'

    train_anns = load_coco_annotation_json(train_json_path, num)
    val_anns = load_coco_annotation_json(val_json_path, num)
    test_anns = load_coco_annotation_json(test_json_path, num)

    if 'train' in phases:
        with open(train_output_path, 'w') as out:
            json.dump(train_anns, out, indent = 4)
    
    if 'val' in phases:
        with open(val_output_path, 'w') as out:
            json.dump(val_anns, out, indent = 4)
    
    if 'test' in phases:
        with open(test_output_path, 'w') as out:
            json.dump(test_anns, out, indent = 4)

if __name__ == '__main__':

    # copy images
    train = './subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
    val = './subset_custom_annotation/coco_karpathy_val_subset_custom.json'
    test = './subset_custom_annotation/coco_karpathy_test_subset_custom.json'

    image_root = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/images'
    output_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'

    print('copying train')
    copy_images(train, image_root, output_dir)
    print()
    print('copying val')
    copy_images(val, image_root, output_dir)
    print()
    print('copying test')
    copy_images(test, image_root, output_dir)


    # # get subset
    # get_subset(501, phases = ['train'])