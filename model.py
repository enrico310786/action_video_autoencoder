import torch.nn as nn
from torchvision import models
import os
import torch
from torchvision.models.video import R2Plus1D_18_Weights
from transformers import TimesformerModel


class Encoder(nn.Module):
    def __init__(self,  init_dim, num_autoencoder_layers, dim_autoencoder_layers, dropout):
        super(Encoder, self).__init__()
        self.init_dim = init_dim
        self.num_autoencoder_layers = num_autoencoder_layers
        self.dim_autoencoder_layers = dim_autoencoder_layers
        self.layer_1 = nn.Linear(self.init_dim, dim_autoencoder_layers[0])
        self.layer_2 = nn.Linear(dim_autoencoder_layers[0], dim_autoencoder_layers[1])
        self.layer_3 = None
        if num_autoencoder_layers == 3:
            self.layer_3 = nn.Linear(dim_autoencoder_layers[1], dim_autoencoder_layers[2])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(init_dim)

    def forward(self, x):
        #x = x.float()
        # normalize the input tensor between 0 and 1 before to pass it through the encoder
        #x -= x.min(1, keepdim=True)[0]
        #x /= x.max(1, keepdim=True)[0]
        x = self.batchNorm1(x)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        if self.num_autoencoder_layers == 3:
            x = self.relu(x)
            x = self.dropout(x)
            x = self.layer_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self,  init_dim, num_autoencoder_layers, dim_autoencoder_layers, dropout):
        super(Decoder, self).__init__()
        self.init_dim = init_dim
        self.num_autoencoder_layers = num_autoencoder_layers
        self.dim_autoencoder_layers = dim_autoencoder_layers
        self.layer_3 = None
        if num_autoencoder_layers == 3:
            self.layer_3 = nn.Linear(dim_autoencoder_layers[2], dim_autoencoder_layers[1])
        self.layer_2 = nn.Linear(dim_autoencoder_layers[1], dim_autoencoder_layers[0])
        self.layer_1 = nn.Linear(dim_autoencoder_layers[0], self.init_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.num_autoencoder_layers == 3:
            x = self.layer_3(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_1(x)
        #x = self.sigmoid(x)

        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TimeSformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

    def forward(self, x):
        x = self.base_model(x)
        x = x.last_hidden_state
        # the output of the timsformer is [B, T, E], with B the batch size, T the number of the sequence tokens and E the embedding dimensions
        # take just the first component of the second dimension, namely the embedding tensor relative to the classification token for all the tensors in the batch
        x = x[:, 0, :]
        return x


class R2plus1d_18(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.base_model.fc = Identity()

    def forward(self, x):
        x = self.base_model(x)
        return x


class R3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        self.base_model.blocks[5].proj = Identity()

    def forward(self, x):
        x = self.base_model(x)
        return x


class TimeAutoencoder(nn.Module):
    def __init__(self,  model_config):
        super().__init__()
        self.init_dim = model_config['init_dim']
        self.name_time_model = model_config['name_time_model']
        self.freeze_layers = model_config.get("freeze_layers", 1.0) > 0.0
        self.num_autoencoder_layers = model_config['num_autoencoder_layers']
        dim_autoencoder_layers = model_config['dim_autoencoder_layers'].split(",")
        self.dim_autoencoder_layers = [int(i) for i in dim_autoencoder_layers]
        self.dropout = model_config['dropout']
        self.base_model = None

        if self.name_time_model == "timesformer":
            self.base_model = TimeSformer()
        elif self.name_time_model == "r2plus1d_18":
            self.base_model = R2plus1d_18()
        elif self.name_time_model == "r3d":
            self.base_model = R3D()

        if self.freeze_layers:
            print("Freeze layers of base model")
            self.freeze_layers_base_model()

        self.encoder = Encoder(self.init_dim, self.num_autoencoder_layers, self.dim_autoencoder_layers, self.dropout)
        self.decoder = Decoder(self.init_dim, self.num_autoencoder_layers, self.dim_autoencoder_layers, self.dropout)

    def freeze_layers_base_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.base_model(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def find_last_checkpoint_file(checkpoint_dir, use_best_checkpoint=False):
    '''
    Cerco nella directory checkpoint_dir il file .pth.
    Se use_best_checkpoint = True prendo il best checkpoint
    Se use_best_checkpoint = False prendo quello con l'epoca maggiore tra i checkpoint ordinari
    :param checkpoint_dir:
    :param use_best_checkpoint:
    :return:
    '''
    print("Cerco il file .pth in checkpoint_dir: {} ".format(checkpoint_dir))
    list_file_paths = []

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pth"):
            path_file = os.path.join(checkpoint_dir, file)
            list_file_paths.append(path_file)
            print("Find: {}".format(path_file))

    print("Number of files .pth: {}".format(int(len(list_file_paths))))
    path_checkpoint = None

    if len(list_file_paths) > 0:

        if use_best_checkpoint:
            if os.path.isfile(os.path.join(checkpoint_dir, 'best.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        else:
            if os.path.isfile(os.path.join(checkpoint_dir, 'latest.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')

    return path_checkpoint