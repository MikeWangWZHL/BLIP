from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
from glob import glob
from datetime import date
import json

def get_image_description(local_image_path, client):
    print("===== Describe an Image - local =====")
    # Open local image file
    local_image = open(local_image_path, "rb")

    # Call API
    description_result = computervision_client.describe_image_in_stream(local_image)

    # Get the captions (descriptions) from the response, with confidence level
    print("Description of local image: ")
    if (len(description_result.captions) == 0):
        print("No description detected.")
    else:
        for caption in description_result.captions:
            print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
    print()

def get_image_tags(local_image_path, client):
    print("===== Tag an Image - local =====")
    # Open local image file
    local_image = open(local_image_path, "rb")
    # Call API local image
    tags_result_local = computervision_client.tag_image_in_stream(local_image)

    # Print results with confidence score
    print("Tags in the local image: ")
    ret = []
    if (len(tags_result_local.tags) == 0):
        print("No tags detected.")
    else:
        for tag in tags_result_local.tags:
            print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))
            ret.append((tag.name, tag.confidence))
        print()
    return ret

if __name__ == "__main__":

    subscription_key = "dd77902d70f548408b5203cece5f9466"
    endpoint = "https://image-tagging-wangz3.cognitiveservices.azure.com/"

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # image dir
    image_dir = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP/knowledge_prompted/coco_retrieval/subset_custom_images'
    image_paths = glob(os.path.join(image_dir,'*.jpg'))
    # local_image_path = "/shared/nas/data/m1/wangz3/video_language_pretraining_project/ALPRO/src/error_analysis/msrvtt_ret_test_frames/video7027.mp4/video7027.mp4_frame-185.jpg"
    # get unique output path 
    today = date.today()
    d = today.strftime("%m-%d-%y")
    t = time.strftime("%H-%M-%S", time.localtime())
    timestamp = d + '_' + t
    input_dir_name = os.path.basename(image_dir)
    output_path = f'./{input_dir_name}_tags_{timestamp}.json'
    
    # get tags
    image_2_tag = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_2_tag[image_name] = get_image_tags(image_path, computervision_client)
        
        # 20 calls per minute
        time.sleep(3.2)

        # save results
        with open(output_path, 'w') as out:
            json.dump(image_2_tag, out, indent= 4)
    


