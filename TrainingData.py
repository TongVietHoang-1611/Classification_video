from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os
import time
from SequenceModel import Model
import tensorflow as tf



class ProcessingData(Model):
  
    def __init__(self, img_size, batch_size, max_seq_length, num_features, train, test, epochs):
        super().__init__(img_size, batch_size, max_seq_length, num_features, train, test, epochs)
        self.train_df = pd.read_csv(train)
        self.test_df = pd.read_csv(test)
        self.epochs = epochs
  
    def load_video(self, path, max_frames=0):
        cap = cv2.VideoCapture(path)
        resize = (self.img_size, self.img_size)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [0, 1, 2]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break

                time.sleep(0.05)  # Wait for 0.05 seconds before capturing the next frame
        finally:
            cap.release()
        return np.array(frames)
  
    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    def build_feature_extractor(self):
        feature_extractor = keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(self.img_size, self.img_size, 3),
        )
        preprocess_input = keras.applications.mobilenet_v2.preprocess_input

        inputs = keras.Input((self.img_size, self.img_size, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")

    def label_processor(self, labels):
        label_encoder = keras.preprocessing.LabelEncoder()
        return label_encoder.fit_transform(labels)

    def prepare_all_videos(self, df, root_dir):
        feature_extractor = self.build_feature_extractor()
        num_samples = len(df)
        video_paths = df["video_name"].values.tolist()
        labels = df["tag"].values

        # Convert class labels to label encoding
        labels = self.label_processor(labels[..., None]).numpy()

        frame_masks = np.zeros(shape=(num_samples, self.max_seq_length), dtype="bool")
        frame_features = np.zeros(shape=(num_samples, self.max_seq_length, self.num_features), dtype="float32")

        # For each video
        for idx, path in enumerate(video_paths):
            frames = self.load_video(os.path.join(root_dir, path))
            frames = frames[None, ...]

            temp_frame_mask = np.zeros(shape=(1, self.max_seq_length), dtype="bool")
            temp_frame_features = np.zeros(shape=(1, self.max_seq_length, self.num_features), dtype="float32")

            # Extract features from the frames of the current video
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(self.max_seq_length, video_length)
                for j in range(length):
                    temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
                temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

            frame_features[idx,] = temp_frame_features.squeeze()
            frame_masks[idx,] = temp_frame_mask.squeeze()

        return (frame_features, frame_masks), labels
  
    def run_experiment(self):
        filepath = "model"+ time.strftime("%Y%m%d-%H%M%S")+".h5"
        log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        seq_model = self.get_sequence_model()
        train_data, train_labels = self.prepare_all_videos(self.train_df, "train")
        test_data, test_labels = self.prepare_all_videos(self.test_df, "test")
        
        history = seq_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            validation_split=0.3,
            epochs=self.epochs,
            callbacks=[checkpoint, tensorboard_callback],
        )
        seq_model.save(filepath)
        seq_model.load_weights(filepath)
        _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        return history, seq_model
  
if __name__ == "__main__":
    EPOCHS = 100
    IMG_SIZE = 96
    BATCH_SIZE = 64
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 1280

    _, sequence_model = ProcessingData(
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        max_seq_length=MAX_SEQ_LENGTH, 
        num_features=NUM_FEATURES, 
        train='train.csv', 
        test='test.csv', 
        epochs=EPOCHS
    ).run_experiment()
