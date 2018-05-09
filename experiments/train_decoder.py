
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
import imageio
import numpy as np


from decoder import Decoder, LinearBnRelu

EMB_SIZE = 128
EMB_COUNT = 202599

def get_batch(indices, embeddings, batch_size=1, img_dir='img_align_celeba_160', shuffle=True):
    if shuffle:
        indices_shufled = np.random.permutation(indices)
    else:
        indices_shufled = indices
     
    for start in range(0, len(indices), batch_size):
        batch_indices = indices_shufled[start: start + batch_size]
        
        emb_batch = np.ndarray((batch_size, EMB_SIZE))
        img_batch = np.ndarray((len(batch_indices), 160*160*3))
        
        for i, img_id in enumerate(batch_indices):
            img_path = 'img_align_celeba_160/{0:06d}.jpg'.format(img_id + 1)
            img_batch[i] = np.array(imageio.imread(img_path) / 255).reshape(1, -1)
            emb_batch[i] = embeddings[img_id]
        yield img_batch, emb_batch


decoder = Decoder(in_features=EMB_SIZE, out_features=160*160*3)

embeddings = np.ndarray(shape=(EMB_COUNT,EMB_SIZE))

with open('embeddings_float.txt') as f:
    for i, line in enumerate(f):
        embeddings[i, :] = np.array(list(map(np.float64, line.strip()[4:].split())))

train_indices = np.random.choice(range(0, EMB_COUNT), size=int(EMB_COUNT * 0.9), replace=False)
train_indices.sort()
test_indices = np.array(list(set(range(0, EMB_COUNT)) - set(train_indices)))
test_indices.sort()


optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)



n_epochs = 100
batch_size = 5
iteration = 0
try:
    for epoch in range(n_epochs):
        for img_batch, emb_batch in get_batch(train_indices, embeddings, batch_size):
            decoder.train(True)
            iteration += 1

            img_batch = Variable(torch.FloatTensor(img_batch))
            emb_batch = Variable(torch.FloatTensor(emb_batch))

            decoded = decoder(emb_batch)
            loss = F.binary_cross_entropy(decoded, img_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_iterations.append(loss.data.numpy()[0])

except KeyboardInterrupt:
    torch.save(decoder.state_dict(), 'decoder_state')
    pass
