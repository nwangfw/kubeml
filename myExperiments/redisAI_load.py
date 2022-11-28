#https://developer.redis.com/howtos/redisai/getting-started/

import json
import time
import redisai as rai
import ml2rt
import os
import torch
from torchvision import models


con = rai.Client(host="localhost", port=6379)


pt_model_path = '/home/ning/Repo/kubeml/myExperiments/model_0.pt'


#   print(dir(model))
# con.modelset("model", 'torch', 'cpu', model, tag='v1.0')

# print(model)
# pt_model2 = models.resnet34(pretrained= False)
# pt_model2.load_state_dict(model)

# print(pt_model2)

# st = pt_model2.state_dict()


# con.modelget('model', meta_only=True)
# # for key in st:
# #     con.tensorset(key, st[key], dtype='float')

# # print(con.tensorget('b'))

# for key in st:
#     classes = con.tensorget(key).to_numpy()
#     print(len(classes))

model = models.resnet34(pretrained= False)

model.load_state_dict(torch.load(f'/home/ning/Repo/kubeml/myExperiments/model_0.pt'))
st = model.state_dict()
for name, param in model.named_parameters():
    con.tensorset(f'{name}',param.data.numpy())

for name, param in model.named_parameters():
    print(name)
    classes = con.tensorget(f'{name}')
    print(type(classes))