from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json
import os
from tqdm import tqdm

def unit_test():
    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    predictor = Predictor.from_path("/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/data_engineering/allen_nlp_pretrained_models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    '''predict'''
    # ret = predictor.predict(
    #     sentence="Did Uriah honestly think he could beat the game in under three hours?."
    # )

    '''predict batch json'''
    ret = predictor.predict_batch_json(
        inputs=[
            {
                "sentence":"A narrow kitchen filled with appliances and cooking utensils."
            },
            {
                "sentence":"A child holding a flowered umbrella and petting a yak."
            }
        ]
    )

    print(json.dumps(ret,indent=4, sort_keys=True))

def srl_coco_retrieval(predictor, ann_json, phase, output_dir = None):
    if phase in ['val','test']:
        ann = json.load(open(ann_json))
        ret = []
        for item in tqdm(ann):
            input_batch = []
            for cap_text in item['caption']:
                input_batch.append({"sentence":cap_text})
            batch_result = predictor.predict_batch_json(input_batch)
            for i in range(len(item['caption'])):
                batch_result[i]['text'] = item['caption'][i]
            ret.append({
                "image":item["image"],
                "caption":batch_result
            })
    else:
        ann = json.load(open(ann_json))

        batch_size = 1000
        batches = []
        for i in range(len(ann)//batch_size + 1):
            batches.append(ann[i*batch_size:(i+1)*batch_size])
        print(len(batches))
        all_results = []
        for batch in tqdm(batches):
            input_batch = []
            for item in batch:
                input_batch.append({'sentence':item['caption']})
            batch_result = predictor.predict_batch_json(input_batch)
            all_results += batch_result

        ret = []
        for i in range(len(ann)):
            item = ann[i]
            srl_result = all_results[i]
            srl_result['text'] = item['caption']
            ret.append({
                "image":item["image"],
                "image_id":item["image_id"],
                "caption":srl_result
            })
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ann_name = os.path.basename(ann_json)
        with open(os.path.join(output_dir,f"srl__{ann_name}"), 'w') as out:
            json.dump(ret, out, indent=4)

    return ret

if __name__ == '__main__':
    predictor = Predictor.from_path(
        archive_path = "./allen_nlp_pretrained_models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device = 3
    )

    # ann_val_custom_subset = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_val_subset_custom.json'
    # ann_test_custom_subset = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_test_subset_custom.json'
    # ann_train_custom_subset = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_annotation/coco_karpathy_train_subset_custom_100.json'
    ann_train_coco_full = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/COCO/annotations/coco_karpathy_train.json'

    output_dir = "./srl_results/tmp"
    # srl_coco_retrieval(predictor, ann_val_custom_subset, 'val', output_dir=output_dir)
    # srl_coco_retrieval(predictor, ann_test_custom_subset, 'test', output_dir=output_dir)
    # srl_coco_retrieval(predictor, ann_train_custom_subset, 'train', output_dir=output_dir)
    srl_coco_retrieval(predictor, ann_train_coco_full, 'train', output_dir=output_dir)