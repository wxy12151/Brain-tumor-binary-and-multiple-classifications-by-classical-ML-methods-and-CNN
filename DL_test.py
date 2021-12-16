import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


### Load the model structure
from model import *


### Load the model parameters
# model = torch.load("model_save/mlp/mlp_lr0.01_nodrop/model_epoch200_gpu.pth", map_location=torch.device('cpu')) #53%
# model = torch.load("model_save/mlp/mlp_lr0.001_withdrop/model_epoch400_gpu.pth", map_location=torch.device('cpu')) #83.50%

# model = torch.load("model_save/cnn/cnn-lr0.01-withdrop/model_epoch110_gpu.pth", map_location=torch.device('cpu')) #91%
# model = torch.load("model_save/cnn/cnn-lr0.001-withdrop-epoch400/model_epoch400_gpu.pth", map_location=torch.device('cpu')) #93%



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
        # trans_tensor = transforms.ToTensor()
        # label = trans_tensor(label_read)
        return img, label

    def __len__(self):
        return len(self.img_path)
img_dir = "AMLS-2021_test/test/image"
label_dir = "AMLS-2021_test/test/label.csv"
dataset_test = MyData(img_dir, label_dir)

test_data_size = len(dataset_test)
print("The length of testing set is：{}".format(test_data_size))

### Load the data
test_dataloader = DataLoader(dataset_test, batch_size=len(dataset_test))

### The testing step begins
model.eval()
# total_test_loss = 0
total_accuracy = 0
### no grad optimation
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        # loss = loss_fn(outputs, targets)
        # total_test_loss += loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
# print("loss on entire testing set：{}".format((total_test_loss)))
print("Accuracy on entire testing set：{}".format(total_accuracy/test_data_size))
