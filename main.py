from CNN import CNN
import torch as T
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
from pathlib import Path
from dataset import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

data_path = Path("data/")
train_dir = data_path / "train" 
test_dir = data_path / "test"

batch_size = 32
epochs = 100

train_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_data_augmented = ImageDataset(targ_dir=train_dir, transform=train_transforms)
train_data = ImageDataset(targ_dir=train_dir, transform=test_transforms)
test_data = ImageDataset(targ_dir=test_dir, transform=test_transforms)

shuffle_data = True
validation_split = 0.2
dataset_size = train_data.__len__()
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_data:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)

# Create dataloaders
train_dataloader = DataLoader(train_data_augmented, batch_size=batch_size, shuffle=False, sampler=train_sampler)
validation_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False, sampler=validation_sampler)


model = CNN(dim=(256,256))

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

min_valid_loss = np.inf

device=T.device("cuda:0" if T.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

'''
TRAINING
'''

train_accu = []
train_losses = []

def train(epochs):
    print('\nEpoch : %d'%epochs)

    model.train()

    running_loss=0
    correct=0
    total=0

    for data in tqdm(train_dataloader):
        inputs,labels=data[0].to(device),data[1].to(device)
        optimizer.zero_grad()
        outputs=T.flatten(model(inputs),0)
        loss=criterion(outputs,labels.type(T.float32))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += sum((outputs.round() == (labels)))
        
    train_loss=running_loss/len(train_dataloader)
    accu=100.*correct.cpu().detach().numpy()/total

    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

'''
VALIDATION
'''

eval_losses=[]
eval_accu=[]


def validation():
    model.eval()

    running_loss=0
    correct=0
    total=0

    with T.no_grad():
        for data in tqdm(validation_dataloader):
            images,labels=data[0].to(device),data[1].to(device)
        
            outputs=T.flatten(model(images),0)

            loss= criterion(outputs,labels.type(T.float32))
           
            running_loss+=loss.item()
            
            total += labels.size(0)
            correct += sum((outputs.round() == (labels)))
    
        val_loss=running_loss/len(validation_dataloader)
        accu=correct.cpu().detach().numpy()/total*100

    eval_losses.append(val_loss)
    eval_accu.append(accu)

    print('Validation Loss: %.3f | Accuracy: %.3f'%(val_loss,accu))
    return val_loss

min_valid_loss = np.inf

for epoch in range(1,epochs+1): 
    train(epoch)
    val_loss = validation()
    if min_valid_loss > val_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        min_valid_loss = val_loss
        T.save(model.state_dict(), f'saved_NNet_dataset_final.pth')
    # if (epoch % 50 == 0):
    #     T.save(model.state_dict(), f'saved_NNet_dataset_final_epoch{epoch}.pth') 

'''
Loss- and Accuracy curve
'''

plot1 = plt.figure(1)
plt.plot(train_accu,'-o')
plt.plot(eval_accu,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')

plot2 = plt.figure(2)
plt.plot(train_losses,'-o')
plt.plot(eval_losses,'-o')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Loss')

predictions_roc = np.array([])
predictions_cm = np.array([])
true_labels = np.array([])

model.eval()

with T.no_grad():
  for data in tqdm(validation_dataloader):
    images,labels=data[0].to(device),data[1].to(device)

    outputs=T.flatten(model(images),0)
    predictions_cm = np.append(predictions_cm, outputs.round().cpu().numpy())
    predictions_roc = np.append(predictions_roc, outputs.cpu().numpy())
    true_labels = np.append(true_labels, labels.type(T.float32).cpu().numpy())

### ROC-AUC CURVE ###
fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions_roc, pos_label=1.0)
roc_auc = metrics.auc(fpr, tpr)

fig = plt.figure(4)
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
plt.plot([], [], ' ', label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'\n NNet ROC Curve\ndataset')
ax.legend()


### CONFUSION MATRIX PLOT ###
plot3 = plt.figure(3)
cm = confusion_matrix(true_labels, predictions_cm)
ax = sns.heatmap(cm/np.sum(cm), annot=True, cmap='Blues', fmt='.2%')
ax.set_title(f'\n NNet Confusion Matrix\ndataset')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()