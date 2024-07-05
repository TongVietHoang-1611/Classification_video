from tensorflow import keras
import pandas as pd
import numpy as np

class Model:
  
  def __init__(self, img_size, batch_size, max_seq_length, num_features,train, test,epochs ):
    
    self.img_size=img_size
    self.batch_size=batch_size
    self.max_seq_length=max_seq_length
    self.num_features=num_features
    train_csv=train
    test_csv=test
    self.epochs=epochs
    self.train_df=pd.read_csv(train_csv)
    self.test_df=pd.read_csv(test_csv)
    labels = self.train_df["tag"].values
    self.label_processor = keras.layers.StringLookup(num_oov_indices=0, 
                                                vocabulary=np.unique(self.train_df["tag"]))
    self.labels = self.label_processor(labels[..., None]).numpy()
     
  def get_sequence_model(self):
    class_vocab = self.label_processor.get_vocabulary()

    frame_features_input = keras.Input((self.max_seq_length, self.num_features))
    mask_input = keras.Input((self.max_seq_length,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model
