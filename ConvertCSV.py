import os
import pandas as pd

class ConvertDataset:
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        self.rooms = []

    def convert(self, name_file):
        class_names = os.listdir(self.dataset)
        for class_name in class_names:
            class_path = os.path.join(self.dataset, class_name)
            if os.path.isdir(class_path):  # Ensure it's a directory
                video_files = os.listdir(class_path)
                
                for video in video_files:
                    video_path = os.path.join(class_path, video)
                    if os.path.isfile(video_path):  # Ensure it's a file
                        self.rooms.append((video_path, class_name))

        train_df = pd.DataFrame(data=self.rooms, columns=['video_name', 'tag'])
        train_df.to_csv(f'{name_file}.csv', index=True)


if __name__ == "__main__":
  dataset_path = r"C:\Users\hoang\Desktop\Folder_lam_viec\Video-Classifier-Using-CNN-and-RNN\Custom_dataset\test"  # Use raw string for Windows paths
  converter = ConvertDataset(dataset_path)
  converter.convert('test')