# %% 
#라이브러리
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets


# %% 
##Parameters
#1. learning late (0.001)
#너무 크면 발산 / 너무 작으면 학습 느림 
lr = 1e-3 
#2.한번 학습하는 데이터 수 
batch_size = 64
#3. 복습 횟수
num_epoch = 10

ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
## Networks (MNIST 숫자 분류기 모델 정의)
#Net이라는 이름의 CNN 생성/딥러닝 모델 정의한 클래스 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #[Conv + Pool + ReLU] - 첫 번째 블록
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
            # 1채널(흑백) → 10채널, 커널 5x
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        #[Conv + Dropout + Pool + ReLU] - 두 번째 블록
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        #Fully Connected (완전 연결층)
        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)
            #50%확률로 끔

        #출력층 (클래스 예측): 최종 출력 10개개
        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x


# %% 
## 학습 오래걸릴때 중간 저장/불러오기 
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim


# %% 
## MNIST 데이터 불러오기
# 1. 이미지 전처리 NumPy 배열을 PyTorch 텐서로 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=0)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data / batch_size)


# %% 
## 네트워크 설정 및 필요한 손실함수 구현하기
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)


# %% 
## 트레이닝 시작하기
for epoch in range(1, num_epoch + 1):
    net.train()

    loss_arr = []
    acc_arr = []

    for  batch, (input, label) in enumerate(loader, 1):
        input = input.to(device)
        label = label.to(device)

        output = net(input)
        pred = fn_pred(output)

        optim.zero_grad()

        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        loss.backward()

        optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
              (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

    writer.add_scalar('loss', np.mean(loss_arr), epoch)
    writer.add_scalar('acc', np.mean(acc_arr), epoch)

    save(ckpt_dir = ckpt_dir, net=net, optim=optim, epoch=epoch)

writer.close()
