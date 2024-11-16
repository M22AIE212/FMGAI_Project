import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_dataset,create_parser
from args import Arguments
import os
from collator_fn import CustomCollator
from train_eval import train_and_validate
from clip_classifier import CLIPClassifier
import argparse

if __name__ == "__main__" :
    
    parser = create_parser()
    args = parser.parse_args()
    print('Fusion method:', args.fusion)
    args = Arguments(fusion = args.fusion)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device : {device} with fusion {args.fusion.upper()}")
    collator = CustomCollator(args = Arguments())
    root_folder = os.getcwd()
    image_folder = "abo-images-small/images/small/"

    dataset_train = load_dataset(args,root_folder,image_folder, split='train')
    dataset_validation = load_dataset(args,root_folder,image_folder, split='validation')
    dataset_test = load_dataset(args,root_folder,image_folder, split='test')

    ## Initializing Dataloader
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_cpus, collate_fn=collator)
    val_dataloader = DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpus, collate_fn=collator)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpus, collate_fn=collator)

    ## Initialize model
    model = CLIPClassifier(args)
    model = torch.nn.DataParallel(model, device_ids=args.gpus)

    # Print trainable layers
    trainable_layers = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    for name, param in trainable_layers:
        print(f"Layer: {name}, Size: {param.size()}")

    # Define optimizer and optional scheduler
    loss_fn = nn.CrossEntropyLoss()

    param_dicts = [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
                ]
    optimizer = AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = None

    # Train the model
    train_and_validate(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        device=device,
        loss_fn = loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        fusion = args.fusion
    )
