# from scipy.io import wavfile

# Path to the wav file
wav_path = '../ICSD/demo/Real_Snoring.wav'
wav_path2 = '../ICSD/demo/Real_Infantcry.wav'

# # Load the wav file
# sample_rate, data = wavfile.read(wav_path)

# print(f"Sample rate: {sample_rate}")
# print(f"Data shape: {data.shape}")

import librosa
import sounddevice as sd

y, sr = librosa.load(wav_path, sr=16000)
y2, sr2 = librosa.load(wav_path2, sr=16000)

snore_start = int(4.3 * sr)
snore_end = int(5.3 * sr)  # 1-second clip
snore_clip = y[snore_start:snore_end]
non_snore_start = int(0.3 * sr2)
non_snore_end = int(1.3 * sr2)  # 1-second clip
non_snore_clip = y2[non_snore_start:non_snore_end]
print(f"Clip shape: {snore_clip.shape}")
print(f"y shape: {y.shape}")
# print("Clip and y are identical:" , (clip.shape == y.shape and (clip == y).all()))
# sd.play(snore_clip, sr)
# sd.wait()
# print('now non snore clip')
# sd.play(non_snore_clip, sr)
# sd.wait()
mfcc = librosa.feature.mfcc(y=snore_clip, sr=sr, n_mfcc=13)

print(f"MFCC shape: {mfcc.shape}")
# print(f"MFCC: {mfcc}")

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13, 32)),  # MFCC shape (e.g., 13x32)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# ...existing code...

import numpy as np

# Example: create dummy labels (0 for snore, 1 for non-snore)
mfcc_no_snore = librosa.feature.mfcc(y=non_snore_clip, sr=sr, n_mfcc=13)
X_train = np.stack([mfcc, librosa.feature.mfcc(y=non_snore_clip, sr=sr, n_mfcc=13)])

y_train = np.array([0, 1])

# Reshape X_train to (samples, 13, 32)
X_train = X_train[:, :, :32]  # Ensure both have the same time dimension

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)
# Save the model
model.save('snore_detection_model.h5')
print("Model trained and saved as 'snore_detection_model.h5'.")
# look at the model
model.summary()
# Load the model
loaded_model = tf.keras.models.load_model('snore_detection_model.h5')
# Predict using the loaded model
predictions = loaded_model.predict(mfcc_no_snore.reshape(1, 13, 32))  # Reshape to match input shape
print("Predictions:", predictions)
predicted_classes = np.argmax(predictions, axis=1)  
print("Predicted classes:", predicted_classes)
# Evaluate the model