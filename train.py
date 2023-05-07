from models import get_model
from dataset import ImageDataset
from utils import save_model

import torch
import torch.nn as nn

import torch.optim as optim

import numpy as np

DEVICE = 'cuda'
MODEL_NAME = 'conv-mix'   # 'conv-mix' for ConvMixer and 'res-net' for ResNet-50
BATCH_SIZE = 16
EPOCH = 15
LEARNING_RATE = 3 * 1e-4
BETAS = (0.5, 0.9)

PATH = './models/conv_mix.pt' if MODEL_NAME == 'conv-mix' else './models/resnet.pt'

print(PATH)

# Define Model
model = get_model(MODEL_NAME=MODEL_NAME)
model.to(torch.device(DEVICE))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)

# Loss
criterion = nn.CrossEntropyLoss()

# Let's define dataset
train_dataset = ImageDataset(status='train')
test_dataset = ImageDataset(status='test')

# Dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loss = []
train_acc = []

val_loss = []
val_acc = []

for epoch in range(EPOCH):
    
    total_acc_train = 0
    total_loss_train = 0
    
    # for accuracy calculation
    total_samples = 0
    total_correct_predictions = 0
    
    model.train()
    
    for image, label in train_dataloader:
        image = image.to(torch.device('cuda'))
        label = label.to(torch.device('cuda'))
        
        output = model(image)
        
        batch_loss = criterion(output, label)
        total_loss_train += batch_loss.item()
        
        _, predicted_labels = torch.max(output.data, 1)
        total_samples += label.size(0)
        total_correct_predictions += (predicted_labels == label).sum().item()
        
        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
    print(
            f'Epochs: {epoch + 26} | Train Loss: {total_loss_train / total_samples: .3f} \
            | Train Accuracy: {total_correct_predictions / total_samples: .3f}')
    
    train_acc.append(total_correct_predictions / total_samples)
    train_loss.append(total_loss_train / total_samples)

    total_acc_val = 0
    total_loss_val = 0

    # For accuracy calculation
    total_samples = 0
    total_correct_predictions = 0

    # Save the model
    save_model(model, optimizer)
    
    model.eval()
    
    with torch.no_grad():
        for image, label in test_dataloader:
            
            image = image.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))

            output = model(image)

            batch_loss = criterion(output, label)
            total_loss_val += batch_loss.item()
            
            acc = (output.argmax(dim=1) == label).sum().item()
            total_acc_val += acc
            val_acc.append(total_acc_val)

            _, predicted_labels = torch.max(output.data, 1)
            total_samples += label.size(0)
            total_correct_predictions += (predicted_labels == label).sum().item()

        print(
                f'Epochs: {epoch + 26} | Validation Loss: {total_loss_train / total_samples: .3f} \
                | Validation Accuracy: {total_correct_predictions / total_samples: .3f}')
    
    val_acc.append(total_correct_predictions / total_samples)
    val_loss.append(total_loss_train / total_samples)
