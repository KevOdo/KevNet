import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as numpy
from DataLoad import UCF11Dataset

class KevNet(nn.Module):
  def __init__(self):
    super(KevNet, self).__init__()
    self.features = nn.Sequential(
      *list(alexNet.features.children())[:-3]
    )
    self.avgpool = nn.Sequential(*list(alexNet.avgpool.children()))
    for param in self.features.parameters():
      param.requires_grad_ = False
    for param in self.avgpool.parameters():
      param.requires_grad_ = False
    self.reduce_features = nn.Linear(12544, 8)
    self.lstm = nn.LSTM(8,64)
    self.reduce_lstm = nn.Linear(64,11)
    self.h = None


  def get_alexnet_features(self, inputs):
    out = self.features(inputs)
    out = self.avgpool(out)
    out = torch.flatten(out,1)
    out = self.reduce_features(out)
    return out

  def init_hidden(self):
    hidden = (torch.randn(1,batch_size,64), torch.randn(1,batch_size,64))
    return hidden

  def forward(self, x):
    outputs = []
    for i in x:
      i = i.permute(0,3,1,2,)
      i = i.float()
      outputs.append(self.get_alexnet_features(i))

    outputs = torch.stack(outputs)
    #print (outputs.size())
    self.h = self.init_hidden()
    #inputs = outputs.view(len(outputs),1,-1)
    output, self.h = self.lstm(outputs, self.h)

    ht = self.h[0]
    out = self.reduce_lstm(ht[-1])
    return out

# class LSTM(nn.Module):
#   def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#     super(LSTM, self).__init__()
#     self.hidden_dim = hidden_dim
#     self.layer_dim = layer_dim
#     self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#     self.fc = nn.Linear(hidden_dim, output_dim)

#   def forward(self, x):
#     h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#     c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#     out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#     out = self.fc(out[:, -1, :])
#     return out

# def joinModels(mod1, mod2):
#   for param in mod1.parameters():
#     param.requires_grad_ = False
#   return nn.Sequential(mod1, mod2)

batch_size = 5
trans = transforms.Compose([torchvision.transforms.Resize(128, 128)])
frame_dataset = UCF11Dataset(csv_file='./frames/frames.csv', root_dir='./frames', transform=trans)
trainLoader = torch.utils.data.DataLoader(frame_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = { 'basketball' : 0,
            'biking' : 1,
            'diving' : 2,
            'golf' : 3,
            'riding' : 4,
            'juggle' : 5,
            'swing' : 6,
            'tennis' : 7,
            'jumping' : 8,
            'spiking' : 9,
            'walk' : 10,
}

alexNet = models.alexnet(pretrained=True)
kevNet = KevNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(kevNet.parameters(), lr=0.001, momentum=0.9)

print ("Starting Training")
for epoch in range(10):
  print("Training Epoch: {}".format(epoch))
  epoch_loss = 0
  for i , data in enumerate(trainLoader, 0):
    inputs = data.get('frame')
    out = kevNet(inputs)
    labels = data.get('category')

    cat = []
    for item in labels:
      cat.append(classes.get(item))
    cat = torch.LongTensor(cat)

    optimizer.zero_grad()
    loss = criterion(out, cat)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    #print(loss.item())
  print("Epoch {} loss: {}".format(epoch, epoch_loss))