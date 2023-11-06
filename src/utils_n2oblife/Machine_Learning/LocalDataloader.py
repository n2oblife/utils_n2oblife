import os, glob, torch
import torch.utils.data as data


# class ImageDataLoader():
#     def __init__(self, dataset):
#         self.dataset = dataset
        

#     def __getitem__(self, index):
#             img_path = self.img_files[index]
#             mask_path = self.mask_files[index]
#             data = use opencv or pil read image using img_path
#             label =use opencv or pil read label  using mask_path
#             return torch.from_numpy(data).float(), torch.from_numpy(label).float()

#     def __len__(self):
#         return len(self.img_files)