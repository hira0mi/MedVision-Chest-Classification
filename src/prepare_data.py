import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
TARGET_LABELS = [
    "Consolidation",
    "Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Infiltration"
]


class DataPreprocessor:
    def __init__(self, target_labels):
        self.target_labels = target_labels
        self.image_path_dict = dict()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.current_dir, '..', 'data')
        self.output_path = os.path.join(self.data_dir, 'labels_all.csv')
    
    def download_and_load(self):
        try:
            print('Начинаю загрузку датасета...')
            self.path = kagglehub.dataset_download('nih-chest-xrays/data')
            self.csv_path = os.path.join(self.path, 'Data_Entry_2017.csv')
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f'Ошибка при загрузке файлов: {e}')
    
    def build_index(self):
        for folder in os.listdir(self.path):
            folder_path = os.path.join(self.path, folder)
            if os.path.isdir(folder_path) and folder.startswith('images_'):
                subfolder_path = os.path.join(folder_path, 'images')
                if os.path.exists(subfolder_path):
                    for image_name in os.listdir(subfolder_path):
                        self.image_path_dict[image_name] = os.path.join(subfolder_path, image_name)
    
    def process_labels(self):
        for label in TARGET_LABELS:
            self.df[label] = self.df[label].str.contains(label, regex=False).astype(int)
        keep_columns = ['Image Index'] + TARGET_LABELS
        self.df = self.df[keep_columns]
    
    def split_and_save(self):
        train_df, val_df = train_test_split(
            self.df, 
            test_size=0.2, 
            random_state=42,
            shuffle=True
        )
        os.makedirs(self.data_dir, exist_ok=True)

        train_df.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val.csv'), index=False)
        print(f"Готово. Обучающий датафрейм: {len(train_df)}, валидационный: {len(val_df)}")
        

    def run(self):
        self.download_and_load()
        self.build_index()
        self.process_labels()
        self.split_and_save()

if __name__ == '__main__':
    preprocessor = DataPreprocessor(target_labels=TARGET_LABELS)
    preprocessor.run()


