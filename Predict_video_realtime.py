from tensorflow import keras
import cv2
import numpy as np
import time
from TrainingData import ProcessingData

class predict:
  
    def __init__(self, max_seq_length, img_size, num_features):
        self.processingdata = ProcessingData(
            img_size=img_size, 
            batch_size=64, 
            max_seq_length=max_seq_length, 
            num_features=num_features, 
            train='train.csv', 
            test='test.csv', 
            epochs=100
        )
        self.max_seq_length = max_seq_length
        self.img_size = img_size
        self.num_features = num_features
        self.feature_extractor = self.processingdata.build_feature_extractor()
    
    def prepare_video_for_prediction(self, frames):
        frame_masks = np.zeros(shape=(1, self.max_seq_length), dtype="bool")
        frame_features = np.zeros(shape=(1, self.max_seq_length, self.num_features), dtype="float32")

        frames = frames[None, ...]
        video_length = frames.shape[1]
        length = min(self.max_seq_length, video_length)

        for i in range(length):
            frame_features[0, i, :] = self.feature_extractor.predict(frames[:, i, :])
        frame_masks[0, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_masks

    def predict_from_webcam_realtime(self, model_path):
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam

        # Load the saved sequence model
        sequence_model = keras.models.load_model(model_path)

        class_name = ["Normal", "Unnormal"]

        while True:
            frames = []
            start_time = time.time()
            
            for _ in range(self.max_seq_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.processingdata.crop_center_square(frame)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame[:, :, [0, 1, 2]]  # Convert BGR to RGB
                frames.append(frame)

                # Display the frame
                cv2.imshow('Webcam Feed', frame)

                # Wait for 1 second before capturing the next frame
                while time.time() - start_time < 0.5:
                    if cv2.waitKey(1) & 0xFF != 255:
                        break

                start_time = time.time()

            if len(frames) < self.max_seq_length:
                continue

            frames = np.array(frames)
            frame_features, frame_masks = self.prepare_video_for_prediction(frames)

            # Predict
            predictions = sequence_model.predict([frame_features, frame_masks])
            predicted_class = np.argmax(predictions, axis=1)
            index = predicted_class[0]
            print(class_name[index])

            # Wait for 1 millisecond for user input (stop if any key is pressed)
            if cv2.waitKey(1) & 0xFF != 255:
                break

        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
  
    IMG_SIZE = 96
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 1280
  
    predict(max_seq_length=MAX_SEQ_LENGTH, 
            img_size=IMG_SIZE, 
            num_features=NUM_FEATURES).predict_from_webcam_realtime(r"models\model2.h5")
