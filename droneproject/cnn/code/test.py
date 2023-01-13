from CNN import CNN
import torch as T
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from dataset import ImageDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics


data_path = Path("data/")
# test data
# test_dir = data_path / "test"

# drone images
# test_dir = data_path / "drone_image"

# real images
test_dir = data_path / "real"

batch_size = 16
epochs = 100

test_transforms = transforms.Compose([
    transforms.ToTensor()
])
test_data = ImageDataset(targ_dir=test_dir, transform=test_transforms)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = CNN(dim=(256, 256))
criterion = nn.BCELoss()
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

model.load_state_dict(
    T.load('saved_NNet_dataset_final.pth', map_location=T.device(device)))

model.eval()

model.to(device)

test_losses = []
test_accu = []


def test():

    running_loss = 0
    correct = 0
    total = 0

    with T.no_grad():
        for data in tqdm(test_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = T.flatten(model(images), 0)

            loss = criterion(outputs, labels.type(T.float32))

            running_loss += loss.item()

            total += labels.size(0)
            correct += sum((outputs.round() == (labels)))

        test_loss = running_loss/len(test_dataloader)
        accu = correct.cpu().detach().numpy()/total*100

    test_losses.append(test_loss)
    test_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))
    return test_loss


test_loss = test()
predictions_roc = np.array([])
predictions_cm = np.array([])
true_labels = np.array([])

with T.no_grad():
    for data in tqdm(test_dataloader):
        images, labels = data[0].to(device), data[1].to(device)

        outputs = T.flatten(model(images), 0)
        predictions_cm = np.append(
            predictions_cm, outputs.round().cpu().numpy())
        predictions_roc = np.append(predictions_roc, outputs.cpu().numpy())
        true_labels = np.append(
            true_labels, labels.type(T.float32).cpu().numpy())

### ROC-AUC CURVE ###
fpr, tpr, thresholds = metrics.roc_curve(
    true_labels, predictions_roc, pos_label=1.0)
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
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

plt.show()
