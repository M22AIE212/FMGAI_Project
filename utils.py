from dataset import EcommerceDataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer
from collections import OrderedDict
import pickle
from args import Arguments
from clip_classifier import CLIPClassifier
import torch
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Process fusion argument.')
    parser.add_argument('--fusion', type=str, default='attention_m', help='Type of fusion method')
    return parser

def load_dataset(args, root_folder,image_folder,split):
    dataset = EcommerceDataset(root_folder=root_folder, image_folder=image_folder, split=split, image_size=args.image_size)
    return dataset

# Function to preprocess image
def preprocess_image(image_path):
    image_size = 224
    image = Image.open(f"{image_path}").convert('RGB').resize((image_size, image_size))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0) # Add batch dimension
    return image

def load_model(model_path,args):
    
    model_obj = CLIPClassifier(args)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model_obj.load_state_dict(new_state_dict)
    return model_obj
