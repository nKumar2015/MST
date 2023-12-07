import librosa
import numpy as np

# Load the audio file
audio_file_path = 'FeatureExtraction/StarWars60.wav'
y, sr = librosa.load(audio_file_path)

# Compute the Chroma Short-Time Fourier Transform (chroma_stft)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Calculate the mean chroma feature across time
mean_chroma = np.mean(chromagram, axis=1)

# Define the mapping of chroma features to keys
chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Find the key by selecting the maximum chroma feature
estimated_key_index = np.argmax(mean_chroma)
estimated_key = chroma_to_key[estimated_key_index]

# Print the detected key
print("Detected Key:", estimated_key)

# Ignore detected key code above, just ues the chromagram and do a frame-wise cosine similarity
# A frame-wise cosine similarity is for every frame, so do not use the mean_chroma value.
# You calculate this value by sklearn.metrics.pairwise.cosine_similarity