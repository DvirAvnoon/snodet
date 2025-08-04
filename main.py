# from scipy.io import wavfile


# # Load the wav file
# sample_rate, data = wavfile.read(wav_path)

# print(f"Sample rate: {sample_rate}")
# print(f"Data shape: {data.shape}")



import os
import librosa
import sounddevice as sd
import tensorflow as tf
import numpy as np
import time



def load_and_clip_wav(wav_path, start_time, end_time, sr=16000):
    """
    Load a wav file and clip it to the specified start and end times.
    """
    y, sr = librosa.load(wav_path, sr=sr)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return y[start_sample:end_sample], sr

def preprocess():
    wav_path = '../ICSD/demo/Real_Snoring.wav'
    wav_path2 = '../ICSD/demo/Real_Infantcry.wav'
    wav_array = [wav_path, wav_path2]
    start_array = [4.3, 0.3]  # Start times for snore and non-snore
    end_array = [5.3, 1.3]  # End times for snore and non-snore
    # File paths for MFCCs
    mfcc_snore_path = 'mfcc_snore.npy'
    mfcc_non_snore_path = 'mfcc_non_snore.npy'
    mfcc_path_list = [mfcc_snore_path, mfcc_non_snore_path]
    mfcc_list = []
    for mfcc_path, wav, start, end in zip(mfcc_path_list, wav_array, start_array, end_array):
        if os.path.exists(mfcc_path):
            curr_mfcc = np.load(mfcc_path)
            print(f"Loaded MFCC from {mfcc_path}")
            mfcc_list.append(curr_mfcc)
        else:
            y, sr = load_and_clip_wav(wav, start, end)  # Snore clip
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            np.save(mfcc_path, mfcc)
            print(f"Computed and saved MFCC to {mfcc_path}")
            mfcc_list.append(mfcc)

    return mfcc_list

def train_model(mfcc_list):
    """
    Train a simple model using the provided MFCC features.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13, 32)),  # MFCC shape (e.g., 13x32)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    X_train = np.stack([mfcc_list[0], mfcc_list[1]])
    y_train = np.array([0, 1])
    X_train = X_train[:, :, :32]  # Ensure both have the same time dimension
    model.fit(X_train, y_train, epochs=10, batch_size=1)
    
    return model


if __name__ == "__main__":
    # Path to the wav file
    # Load or compute MFCC for snore
    mfcc_list = preprocess()



    y_train = np.array([0, 1])


    model_path = 'snore_detection_model.h5'
    start_time = time.time()
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"Model not found at {model_path}. Training a new model.")
        loaded_model = train_model(mfcc_list)
        loaded_model.save(model_path)
        print(f"Model saved to {model_path} in {time.time() - start_time:.2f} seconds.")

    # Predict using the loaded model
    pred_start = time.time()
    predictions = loaded_model.predict(mfcc_list[0].reshape(1, 13, 32))  # Reshape to match input shape
    print(f"Prediction time: {time.time() - pred_start:.2f} seconds.")
    print("Predictions:", predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted classes:", predicted_classes)
