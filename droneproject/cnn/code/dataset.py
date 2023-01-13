from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image


def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx


class ImageDataset(Dataset):
    def __init__(self, targ_dir, transform=None) -> None:
        self.paths = list(Path(targ_dir).glob("*/*"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index):
        image_path = self.paths[index]
        image = Image.open(image_path)
        return image

    def __len__(self):
        size = len(self.paths)
        return size

    def __getitem__(self, index):
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx
