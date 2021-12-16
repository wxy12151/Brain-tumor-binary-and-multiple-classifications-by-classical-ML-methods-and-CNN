import torch
from torch import nn

# MLP with single hidden layer
class MLP_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_3, self).__init__()
        self.model1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# CNN: plus simple Convolutional layer and Pooling layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model2 = nn.Sequential(
            # 1 x 512 x 512
            nn.Conv2d(1, 8, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4), # 8 x 128 x 128
            nn.Conv2d(8, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4), # 16 x 32 x 32
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4), # 32 x 8 x 8

            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.model2(x)
        return x

### test the model structure
if __name__ == '__main__':
    input_size = 512 * 512
    hidden_size = 2000
    num_classes = 4
    # model = MLP_3(input_size, hidden_size, num_classes)
    model = CNN()
    input = torch.ones((64, 1, 512, 512))
    output = model(input)
    print(output.shape)