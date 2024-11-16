from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer
from utils import preprocess_image,load_model
from args import Arguments
import pickle
import torch

# Inference function
def inference(model_path,fusion,image_path,product_name,label_encoder_path="label_encoder.pkl"):
    
    with open(label_encoder_path, 'rb') as f:
        loaded_label_encoder = pickle.load(f)
    
    args = Arguments(fusion)
    model = load_model(model_path,args)
    
    image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
    text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    # Ensure model is in eval mode
    model.eval()

    # Preprocess image
    image = preprocess_image(image_path)
    pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values']

    # Prepare text input
    text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)
    text_output = text_processor(product_name, padding=True, return_tensors="pt", truncation=True)
    
    # Prepare batch
    batch = {
        'pixel_values': [pixel_values],
        'input_ids': text_output['input_ids'],
        'attention_mask': text_output['attention_mask']
    }

    # Pass batch through model
    with torch.no_grad():
        logits = model(batch)

    # Get predicted class
    predicted_class = loaded_label_encoder.inverse_transform(torch.argmax(logits, dim=1).cpu().numpy())
    
    return predicted_class

if __name__ == "__main__" :
    fusion = 'attention_m' # 'concat' , 'cross'
    model_path = f'best_model_{fusion}.pth'
    product_name='Amazon Brand – Stone & Beam Tisbury Nailhead Trim King Bed, 84"W, Curious Pearl'
    image_dir = "/home/jupyter/sagar/clip_training/codes/abo-images-small/images/small/"
    image_path = f"{image_dir}/{'e5/e5c6217b.jpg'}"
    inference(model_path,fusion,image_path,product_name)
