#import torch
#import torch.nn.functional as F

#def tumor_pad_collate_fn(batch):
 #   data = [item[0] for item in batch]
  #  labels = torch.tensor([item[1] for item in batch])
   # max_len = max([x.shape[0] for x in data])
    #padded_data = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in data])
    #return padded_data, labels

import torch
import torch.nn.functional as F

def tumor_pad_collate_fn(batch):
    data = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    max_len = max(x.shape[0] for x in data)
    padded_data = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in data])

    # Create mask
    mask = torch.tensor([[1]*x.shape[0] + [0]*(max_len - x.shape[0]) for x in data], dtype=torch.bool)

    return padded_data, labels, mask

