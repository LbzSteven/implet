import numpy as np
import torch
from tqdm import tqdm
from typing import cast, Dict
from torch.nn import functional as F, ReLU
from tslearn.datasets import UCR_UEA_datasets
from tqdm.notebook import tqdm, trange
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tsai.models.InceptionTime import InceptionTime
from tsai.models.MLP import MLP
from tsai.models.FCN import FCN
from tsai.models.ResNet import ResNet

def model_init(model_name, in_channels, n_pred_classes, seq_len=None):
    # n_pred_classes = train_y.shape[1]
    # Those models don't have softmax as nn.CrossEntropyLoss alreadyhas SoftMax insie
    if model_name == 'ResNet':
        model = ResNet(c_in=in_channels, c_out=n_pred_classes)
    elif model_name == 'FCN':
        model = FCN(c_in=in_channels, c_out=n_pred_classes)
    elif model_name == 'InceptionTime':
        model = InceptionTime(in_channels, n_pred_classes)
    elif model_name == 'MLP':
        model = MLP(c_in=in_channels, c_out=n_pred_classes, seq_len=seq_len, layers=[500, 500, 500], ps=[0.1, 0.2, 0.2],
                    act=ReLU(inplace=True))
    else:
        raise 'Unspecified model'
    return model

def fit(model, train_loader, device, num_epochs: int = 10,
        learning_rate: float = 0.001,
        patience: int = 1500, ) -> None:  # patience was 100
    # pbar = trange(num_epochs, desc='Process', unit='epoch', initial=0, disable=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_train_loss = np.inf
    patience_counter = 0
    best_state_dict = None

    model.to(device)
    model.train()
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:

            epoch_train_loss = 0
            for x_t, y_t in train_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                optimizer.zero_grad()
                output = model(x_t.float())
                if len(y_t.shape) == 1:
                    train_loss = F.binary_cross_entropy_with_logits(
                        output, y_t.unsqueeze(-1).float(), reduction='mean'
                    )
                else:
                    train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')

                epoch_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()

            model.eval()
            tqdm.update
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_state_dict = model.state_dict()
                patience_counter = 0

            else:
                patience_counter += 1
                if patience_counter == patience:
                    if best_state_dict is not None:
                        model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print(f'Early stopping! at {epoch + 1}, using state at {epoch + 1 - patience}')
                    return None
def get_all_preds(model, loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch in loader:
            item, label = batch
            item = item.to(device)
            preds = model(item.float())
            all_preds = all_preds + preds.cpu().argmax(dim=1).tolist()
            labels = labels + label.tolist()
    return all_preds, labels


def get_all_preds_prob(model, loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch in loader:
            item, label = batch
            item = item.to(device)
            preds = model(item.float())

            all_preds.append(preds.cpu().numpy())
            labels = labels + label.tolist()

    return np.concatenate(all_preds, axis=0), labels


def get_hidden_layers(model, hook_block, data, device='cuda'):
    latent_representation = {}
    if hook_block is None:
        if isinstance(model, FCN) or isinstance(model, InceptionTime):
            hook_block = model.gap.gap
        elif isinstance(model, ResNet):
            hook_block = model.gap
        elif isinstance(model, MLP):
            hook_block = model.mlp[2][2]
        else:
            raise "Unspecified model"

    def forward_hook(name):
        def hook(model, input, output):
            latent_representation[name] = output.detach().cpu()

        return hook

    model.to(device)
    features = data.shape[-2]
    length = data.shape[-1]
    for i in range(len(data)):
        handle = hook_block.register_forward_hook(forward_hook(i))
        input_data = torch.from_numpy(data[i].reshape(1, features, length)).float().to(device)
        output = model(input_data)
        handle.remove()

    return np.vstack(list(latent_representation.values()))