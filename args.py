from dataclasses import dataclass
@dataclass
class Arguments :
  clip_pretrained_model = 'openai/clip-vit-base-patch32'
  image_size = 224
  batch_size = 32
  num_cpus = os.cpu_count()
  use_pretrained_map = False
  num_mapping_layers = 1
  map_dim = 256
  fusion = 'cross'
  num_pre_output_layers = 3
  lr = 0.001
  weight_decay = 1e-3
  weight_image_loss = 0
  weight_text_loss = 0
  weight_super_loss = 0
  drop_probs = [0.1 ,0.1 ,0.1]
  freeze_image_encoder = True
  freeze_text_encoder = True
  num_class = 34