import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import csv
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
        self.lstm = nn.LSTM(8, 64, num_layers=num_layers)
        self.reduce_lstm = nn.Linear(64, 11)
        self.h = None

    def get_alexnet_features(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.reduce_features(out)
        return out

    def init_hidden(self):
        hidden = (torch.randn(num_layers, batch_size, 64), torch.randn(num_layers, batch_size, 64))
        return hidden

    def forward(self, x):
        outputs = []
        for i in x:
            i = i.permute(0, 3, 1, 2, )
            i = i.float()
            outputs.append(self.get_alexnet_features(i))

        outputs = torch.stack(outputs)
        self.h = self.init_hidden()
        output, self.h = self.lstm(outputs, self.h)

        ht = self.h[0]
        out = self.reduce_lstm(ht[-1])
        return out


def train(net, dataLoader):
    print("Starting Training")
    with open(PATH_csv, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Epoch', 'Loss', 'Elapsed Time'])
        for epoch in range(epochs):
            print("Training Epoch: {}".format(epoch))
            epoch_loss = 0
            start_time = time.time()
            for i, data in enumerate(dataLoader, 0):
                inputs = data.get('frame')
                out = net(inputs)
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
            elapsed_time = time.time() - start_time
            norm_loss = epoch_loss / 54
            writer.writerow([epoch, norm_loss, elapsed_time])
            print("Epoch {} loss: {} elapsed time: {}".format(epoch, norm_loss, elapsed_time))

    print('Finished Training')
    torch.save(kevNet.state_dict(), PATH_net)


def test(net, path):
    print('Testing Model')
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testLoader:
            frames = data.get('frame')
            output = net(frames)
            labels = data.get('category')
            cat = []
            for item in labels:
                cat.append(classes.get(item))
            cat = torch.LongTensor(cat)
            _, predicted = torch.max(output.data, 1)
            c = (predicted == cat).squeeze()
            for i in range(batch_size):
                label = cat[i]
                class_correct[label.abs()] += c[i].item()
                class_total[label.abs()] += 1
            total += cat.size(0)
            correct += (predicted == cat).sum().item()
    for i in range(len(classes_list)):
        print('Accuracy of {}: {}'.format(classes_list[i], 100 * class_correct[i] / class_total[i]))
    print('Accuracy: {}'.format(100 * correct / total))


batch_size = 5
num_layers = 1
epochs = 100
lr = 0.001
csv_train = './frames_3/frames_3.csv'
root_train = './frames_3'
csv_test = './frames_test_3/frames_test_3.csv'
root_test = './frames_test_3'
trans = transforms.Compose([torchvision.transforms.Resize(128, 128)])

train_dataset = UCF11Dataset(csv_file=csv_train, root_dir=root_train, transform=trans)
test_dataset = UCF11Dataset(csv_file=csv_test, root_dir=root_test, transform=trans)

trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = {'basketball': 0,
           'biking': 1,
           'diving': 2,
           'golf': 3,
           'riding': 4,
           'juggle': 5,
           'swing': 6,
           'tennis': 7,
           'jumping': 8,
           'spiking': 9,
           'walk': 10,
           }

classes_list = ['basketball',
                'biking',
                'diving',
                'golf',
                'riding',
                'juggle',
                'swing',
                'tennis',
                'jumping',
                'spiking',
                'walk']

alexNet = models.alexnet(pretrained=True)
kevNet = KevNet()
PATH_net = './Models/kev_net_14.pth'
PATH_csv = './Loss Data/loss_60_14.csv'


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(kevNet.parameters(), lr=lr, momentum=0.9)

train(kevNet, trainLoader)
test(kevNet, PATH_net)
