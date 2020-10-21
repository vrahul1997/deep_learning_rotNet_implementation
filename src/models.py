import torch
import torch.nn as nn
import torch.nn.functional as F


class RotNetclassifier(nn.Module):
    def __init__(self, num_classes):
        super(RotNetclassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(35*35*64, 64)
        self.drop2 = nn.Dropout(0.1)
        self.output = nn.Linear(64, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, target=None):
        bs, ch, ht, width = images.shape
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())

        x = F.relu(self.conv2(x))
        # print(x.size())
        x = self.max_pool2(x)
        # print(x.shape)
        x = self.drop1(x)
        x = x.view(-1, 35*35*64)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = self.drop2(x)
        x = self.output(x)
        # print(x.shape)

        if target is not None:
            loss = self.loss(x, target)
            return x, loss

        return x, None


# images = torch.rand(1, 3, 65, 300)
# target = torch.randint(1, 36, (1, 36))
# print(target)
# h = RotNetclassifier(36)

# x = h(images)
