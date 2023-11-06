# test loading models

import torch

model_name = 'resnet50_2x_low_res.pth'
model = torch.load('C:/Users/ZaccarieKanit/SIMULANDS/git_projects/prototype/others/models/'+model_name, 
                   map_location='cpu')
breakpoint()