import os
class Arguments :
    def __init__(self, fusion=None,drop_probs = [0.1 ,0.1 ,0.1]):
        self.clip_pretrained_model = 'openai/clip-vit-base-patch32'
        self.image_size = 224
        self.batch_size = 32
        self.num_cpus = os.cpu_count()
        self.use_pretrained_map = False
        self.num_mapping_layers = 1
        self.map_dim = 1024
        self.fusion = fusion
        self.num_pre_output_layers = 3
        self.lr = 0.001
        self.weight_decay = 1e-3
        self.weight_image_loss = 0
        self.weight_text_loss = 0
        self.weight_super_loss = 0
        self.drop_probs = drop_probs
        self.freeze_image_encoder = True
        self.freeze_text_encoder = True
        self.num_class = 34
        self.gpus = [0,1,2,3]
        self.epochs = 10
