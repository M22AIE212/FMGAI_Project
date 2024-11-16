from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from copy import deepcopy
import pandas as pd

def train_and_validate(model, train_dataloader, val_dataloader, epochs, device, loss_fn,optimizer, scheduler=None, print_every=100,fusion = None):
    """
    Train and validate the model.

    :param model: The CLIPClassifier model.
    :param train_dataloader: DataLoader for the training dataset.
    :param val_dataloader: DataLoader for the validation dataset.
    :param epochs: Number of training epochs.
    :param device: The device to use for training (e.g., 'cuda' or 'cpu').
    :param loss_fn: Loss function for training.
    :param optimizer: Optimizer for model training.
    :param scheduler: Learning rate scheduler (optional).
    :param print_every: How frequently to print training status.
    """

    model.to(device)

    # Trackers for loss and accuracy
    best_val_accuracy = 0.0
    val_loss_list = []
    train_loss_list = []
    # Loop over epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        total_train_loss = 0.0
        train_preds = []
        train_labels = []

        # Progress bar for training
        train_progress = tqdm(train_dataloader, desc='Training', leave=False)

        for batch_idx, batch in enumerate(train_progress):
            # Move batch to device
            pixel_values = batch['pixel_values'][0].to(device)
            pixel_values = [pixel_values]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model({
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })

            # Compute loss
            loss = loss_fn(preds, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Track loss and accuracy
            total_train_loss += loss.item()
            train_preds.extend(torch.argmax(preds,dim=1).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            # Print progress
            if batch_idx % print_every == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                print(f"Batch {batch_idx}, Training Loss: {avg_loss:.4f}")

        # Calculate train accuracy
        train_accuracy = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch + 1} Training Loss: {total_train_loss / len(train_dataloader):.4f}")
        train_loss_list.append(total_train_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc='Validating', leave=False)

            for batch in val_progress:
                # Move batch to device
                pixel_values = batch['pixel_values'][0].to(device)
                pixel_values = [pixel_values]
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                preds = model({
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })

                # Compute loss
                loss = loss_fn(preds, labels)
                total_val_loss += loss.item()

                # Track predictions and labels
                val_preds.extend(torch.argmax(preds,dim=1).detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1} Validation Loss: {total_val_loss / len(val_dataloader):.4f}")
        val_loss_list.append(total_val_loss / len(val_dataloader))
        print(f"Epoch {epoch + 1} Validation Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            # best_val_accuracy = val_accuracy
            # torch.save(model.state_dict(), "best_model.pth")
            # print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
            best_val_accuracy = val_accuracy
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, f"./best_model_{fusion}.pth")
            print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
    df_train_loss = pd.DataFrame(train_loss_list,columns = ['loss'])
    df_val_loss = pd.DataFrame(val_loss_list,columns = ['loss'])
    
    df_train_loss.to_csv(f"loss_{fusion}.csv")
    df_val_loss.to_csv(f"loss_{fusion}.csv")
    print(f"Training complete. Best Validation Accuracy: {best_val_accuracy:.4f}")
