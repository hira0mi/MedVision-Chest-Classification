import kagglehub
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
import torch
TARGET_LABELS = [
    "Consolidation",
    "Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Infiltration"
]
#TO-DO: исправить неравномернное распределение лейблов

class DataPreprocessor:
    def __init__(self, target_labels):
        self.target_labels = target_labels
        self.image_path_dict = dict()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.current_dir, '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_and_load(self):
        try:
            print('Начинаю загрузку датасета...')
            self.path = kagglehub.dataset_download('nih-chest-xrays/data')  
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f'Ошибка при загрузке файлов: {e}')
    
    def build_index(self):
        for root, dirs, files in os.walk(self.path):
            if 'images' in root:
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_path_dict[file] = os.path.join(root, file)
        self.df['File Path'] = self.df['Image Index'].map(self.image_path_dict)
        self.df = self.df.dropna(subset=['File Path'])
    
    def process_labels(self):
        for label in self.target_labels:
            self.df[label] = self.df['Finding Labels'].str.contains(label, regex=False).astype(int)
        
        keep_columns = ['Image Index', 'Patient ID', 'File Path'] + self.target_labels
        self.df = self.df[keep_columns]
    
    def split_and_save(self):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(self.df, groups=self.df['Patient ID']))
        
        train_df = self.df.iloc[train_idx]
        val_df = self.df.iloc[val_idx]

        train_df.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val.csv'), index=False)
        
        print(f"Готово. Обучение: {len(train_df)} снимков, Валидация: {len(val_df)} снимков.")
        pos_weights = []
        for label in self.target_labels:
            num_positives = train_df[label].sum()
            num_negatives = len(train_df) - num_positives
            weight = num_negatives / (num_positives + 1e-5)
            pos_weights.append(weight)
        
        torch.save(torch.tensor(pos_weights, dtype=torch.float32), os.path.join(self.data_dir, 'pos_weights.pt'))
        print("Веса для дисбаланса классов сохранены.")

    def run(self):
        self.download_and_load()
        self.build_index()
        self.process_labels()
        self.split_and_save()

if __name__ == '__main__':
    preprocessor = DataPreprocessor(target_labels=TARGET_LABELS)
    preprocessor.run()


