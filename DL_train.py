import time
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from model import *
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


### Hyperparameter Tuning
batch_size = 30
every_n_step = 20 # print training records every n steps
learning_rate = 0.001
epoch = 400
## Parameters for MLP
input_size = 512 * 512
hidden_size = 256
num_classes = 4
### Load the model structure
## Model_1: MLP with single hidden layer
# model = MLP_3(input_size, hidden_size, num_classes)
# model = model.to(device)
## Model_2: CNN
model = CNN()
model = model.to(device)


### Define Dataset, transform and load data.
class MyData(Dataset):

    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.img_path = os.listdir(self.img_dir)
        self.label_dir = label_dir
        self.label_path = pd.read_csv(self.label_dir)
        self.Y_N = self.label_path['label']
        self.Y_N[self.Y_N == 'no_tumor'] = 0
        self.Y_N[self.Y_N == 'meningioma_tumor'] = 1
        self.Y_N[self.Y_N == 'glioma_tumor'] = 2
        self.Y_N[self.Y_N == 'pituitary_tumor'] = 3

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.img_dir, img_name)
        img_read = Image.open(img_item_path)
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        img = data_transform(img_read)
        label = self.Y_N[idx]
        return img, label

    def __len__(self):
        return len(self.img_path)
img_dir = "AMLS-2021_dataset/dataset/image"
label_dir = "AMLS-2021_dataset/dataset/label.csv"
dataset_all = MyData(img_dir, label_dir)

### split the dataset into train_dataset and valid_dataset: 8:2
train_size = int(len(dataset_all) * 0.8)
valid_size = len(dataset_all) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset_all, [train_size, valid_size])
# print(len(train_dataset)) # 2400
# print(len(valid_dataset)) # 600

train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)
print("The length of training set is：{}".format(train_data_size))
print("The length of validation set is：{}".format(valid_data_size))

### Load the data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)



### Define loss function as cross entropy loss
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

### Define optimizer as SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

### Record training times
total_train_step = 0
### Record validation times
total_valid_step = 0
### Add tensorboard
writer = SummaryWriter("logs_mlp")

start_time = time.time()

for i in range(epoch):
    print("-------Training round {} begins-------".format(i+1))

    ### The training step begins
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        ### Optimizer model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % every_n_step ==0:
            end_time = time.time()
            print(end_time - start_time)
            print("Training Times on batch size：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            start_time = time.time()

    ### The validation step begins
    model.eval()
    total_valid_loss = 0
    total_accuracy = 0
    ### No grad optimization
    with torch.no_grad():
        for data in valid_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_valid_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("loss on entire validation set：{}".format((total_valid_loss)))
    print("Accuracy on entire validation set：{}".format(total_accuracy/valid_data_size))
    writer.add_scalar("valid_loss", total_valid_loss, total_valid_step)
    writer.add_scalar("valid_accuracy", total_accuracy / valid_data_size, total_valid_step)
    total_valid_step += 1

    torch.save(model, "./model_save/mlp/model_epoch{}_gpu.pth".format(i + 1))
    # torch.save(model, "./model_save/cnn/model_epoch{}_gpu.pth".format(i + 1))
    print("Model saved successfully")

writer.close()


