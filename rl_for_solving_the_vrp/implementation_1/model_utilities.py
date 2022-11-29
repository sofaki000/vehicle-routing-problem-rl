import torch


def load_model(model, model_path):
    #path = os.path.join(checkpoint, 'actor.pt')
    model.load_state_dict(torch.load(model_path))
    return model
    # path = os.path.join(checkpoint, 'critic.pt')
    # critic.load_state_dict(torch.load(path, device))