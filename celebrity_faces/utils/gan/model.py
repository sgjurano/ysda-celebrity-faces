import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


EMB_COUNT = 202599
EMB_SIZE = 128
IMG_SHAPE = (160, 160)


class LinearBnRelu(nn.Module):
    """[FC => BN => ReLU]"""

    def __init__(self, in_features, out_features):
        super(LinearBnRelu, self).__init__()
        self.inner_module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.inner_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        self.embeddings = None

        super(Decoder, self).__init__()
        self.linear1 = LinearBnRelu(in_features, 500)
        self.linear4 = LinearBnRelu(500, 4000)
        self.linear_out = nn.Linear(4000, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear4(x)
        x = self.linear_out(x)
        out = F.sigmoid(x)
        return out


def load_model(model_data):
    model_path = os.path.join(model_data, 'decoder_model')
    if not os.path.exists(model_path):
        raise FileNotFoundError('There are no decoder model at {}!'.format(model_path))

    decoder = torch.load(model_path)
    _ = decoder.train(False)

    decoder.embeddings = read_embeddings(os.path.join(model_data, 'embeddings_float.txt'))
    return decoder


def read_embeddings(embeddings_path):
    embeddings = np.ndarray(shape=(EMB_COUNT, EMB_SIZE))

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError('There are no embeddings at {}!'.format(embeddings_path))

    with open(embeddings_path) as f:
        for i, line in enumerate(f):
            embeddings[i, :] = np.array(list(map(np.float64, line.strip()[4:].split())))

    return embeddings


def decode_pairs(decoder, pairs):
    batch = np.ndarray(shape=(len(pairs), EMB_SIZE))
    for idx, (user_img_emb, img_from_dataset_idx) in enumerate(pairs):
        img_from_dataset_emb = decoder.embeddings[img_from_dataset_idx - 1]
        batch[idx, :] = (np.array(user_img_emb, dtype=np.float64) + img_from_dataset_emb) / 2

    decoded_imgs = decoder(Variable(torch.FloatTensor(batch))).data.numpy()
    return decoded_imgs.reshape(len(pairs), *IMG_SHAPE, 3)


