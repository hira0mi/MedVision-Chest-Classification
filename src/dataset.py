from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file , transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = row['File Path']

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f'Ошибка загрузки {full_path}: {e}')
            new_index = random.randint(0, len(self.df) - 1)
            return self.__getitem__(new_index)
    
        labels = row[2:].values.astype('float32')

        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
     