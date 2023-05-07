import torch

def save_model(model, optimizer, PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH)