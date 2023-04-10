
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from glob import glob
import os
from os.path import basename
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import torch
from torch.utils import data 
from os.path import join 
from PIL import Image
from torchvision import transforms 
from torch.utils.data import DataLoader



import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image

#elenchiamo le classi
classes = glob('./drive/MyDrive/medical_images/*')
#estraiamo il nome della classe dal path completo 
classes = [basename(c) for c in classes] 
print(classes)

class_dict = {c : i for i , c in enumerate(classes)} 
print(class_dict)
class_dict_reverse = {i : c for i , c in enumerate(classes)} 
print(class_dict_reverse)

image_paths = glob('./drive/MyDrive/medical_images/*/*') 
print(image_paths[:10])


def class_from_path(path):
    _, _, _, _, cl, _ = path.split('/')
    return class_dict[cl]



labels = [class_from_path(im) for im in image_paths] 
print(labels)

dataset = pd.DataFrame({'path':image_paths, 'label':labels}) 
print(dataset.head())



def split_train_val_test(dataset, perc=[0.6, 0.1, 0.3]):
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2])) 
    return train, val, test



random.seed(1395)
np.random.seed(1359)
train, val, test = split_train_val_test(dataset) 
print(len(train))
print(len(val))
print(len(test))

train.to_csv('./drive/MyDrive/dataset_csv/train.csv', index=None) 
val.to_csv('./drive/MyDrive/dataset_csv/valid.csv', index=None) 
test.to_csv('./drive/MyDrive/dataset_csv/test.csv', index=None)


classes, ids = zip(*class_dict.items())
classes = pd.DataFrame({'id':ids, 'class':classes}).set_index('id') 
classes.to_csv('./drive/MyDrive/dataset_csv/classes.csv')

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils import data 
import pandas as pd

class CSVImageDataset(data.Dataset):
    def __init__(self, data_root, csv, transform = None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        #il dataset contiene alcune immagini in scala di grigi
        #convertiamo tutto in RGB per avere delle immagini consistenti
        im = Image.open(join(self.data_root,im_path)) #.convert('RGB')
        
        if self.transform is not None: 
            im = self.transform(im)
        return im, im_label

class ImageClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        #self.resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet18 = resnet18(pretrained=True)
        
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 6) # 6 classes
        #self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.resnet18(x)
        return F.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc_m', acc)
        self.log('train_acc', self.accuracy(y_hat, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y))
        self.log('val_acc_m', acc)

        #return {
        #    'predictions': y_hat.cpu().topk(1).indices,
        #    'labels': y.cpu()
        #}

    #def on_validation_epoch_end(self, outputs):
    #    #concateniamo tutte le predizioni 
    #    predictions = np.concatenate([o['predictions'] for o in outputs])
    #    #concateniamo tutte le etichette
    #    labels = np.concatenate([o['labels'] for o in outputs])
        
    #    acc = accuracy_score(labels, predictions)
        
    #    self.log('val/accuracy', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(y_hat, y))
        self.log('test_acc_m', acc)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def accuracy(self, y_hat, y):
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / len(preds)
        return acc

def gray_to_rgb(image):
    # assume l'immagine in input è un tensore di dimensione [1, H, W]
    # gray_image = image.repeat(3, 1, 1)
    #rgb_image = torch.cat([image, image, image], dim=0)
    rgb_image = Image.merge('RGB', [image]*3)
    return rgb_image

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(gray_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize((0.3580, 0.3580, 0.3580), (0.2824, 0.2824, 0.2824))
])

#train_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.3580,), (0.2824,)) #da valutare
#])


classes = pd.read_csv('drive/MyDrive/dataset_csv/classes.csv').to_dict()['class']
dataset_train = CSVImageDataset('./','drive/MyDrive/dataset_csv/train.csv',transform=train_transform)
dataset_valid = CSVImageDataset('./','drive/MyDrive/dataset_csv/valid.csv',transform=train_transform) 
dataset_test = CSVImageDataset('./','drive/MyDrive/dataset_csv/test.csv',transform=train_transform)

# im, lab = dataset_train[0]
# print('Class id:',lab, 'Class name:',classes[lab]) 
# print(im)
# from matplotlib import pyplot as plt
# print(im.shape)
# plt.imshow(im.squeeze(), cmap='gray') 
# plt.title("Classe: "+str(lab)) 
# plt.show()

dataset_train_loader = DataLoader(dataset_train, batch_size=256, num_workers=4, shuffle =True)
dataset_valid_loader = DataLoader(dataset_valid, batch_size=256, num_workers=4)
dataset_test_loader = DataLoader(dataset_test, batch_size=256, num_workers=4)


model = ImageClassifier()


# checkpoint callback to save best model during training
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./drive/MyDrive/weights',
    filename='best-resnet18-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
)


logger = TensorBoardLogger("./drive/MyDrive/tb_logs", name="my_model")
#CPU
#trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback], logger=logger) 
#GPU
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, callbacks=[checkpoint_callback], logger=logger) 

# Without resume_checkpoint
trainer.fit(model, dataset_train_loader, dataset_test_loader)
# effettuiamo il fit
#trainer.fit(model, dataset_train_loader, dataset_test_loader, ckpt_path="./drive/MyDrive/weights/best-resnet18-epoch=20-val_loss=1.04.ckpt")

# effettuiamo il passaggio di validation per visualizzare le performance finali di validation
print(trainer.validate(model, dataset_valid_loader))



# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir=tb_logs

#elenchiamo le classi
classes = glob('./drive/MyDrive/medical_images/*')
#estraiamo il nome della classe dal path completo 
classes = [basename(c) for c in classes] 
print(classes)

class_dict = {i : c for i , c in enumerate(classes)} 
print(class_dict)



model = ImageClassifier.load_from_checkpoint("./drive/MyDrive/weights/best-resnet18.ckpt")
model.eval()


def gray_to_rgb(image):
    rgb_image = Image.merge('RGB', [image]*3)
    return rgb_image

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(gray_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize((0.3580, 0.3580, 0.3580), (0.2824, 0.2824, 0.2824))
])




# definisci la cartella da cui prendere le immagini
folder_path = "./drive/MyDrive/test"

# ottieni la lista di tutti i file nella cartella
files = os.listdir(folder_path)

# filtra solo i file con estensione immagine
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

# crea la lista dei path completi delle immagini
image_paths = [os.path.join(folder_path, f) for f in image_files]

print(image_paths)
predictions = []
for path in image_paths:
  print(path)
  image = Image.open(path)
  transformed_image = transform(image)
  output = model(transformed_image.unsqueeze(0))
  prediction = torch.argmax(output)
  predictions.append(prediction.item())
    
print(classes[predictions[0]])