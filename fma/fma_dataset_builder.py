"""fma dataset."""

import tensorflow_datasets as tfds
import numpy as np
import librosa
import tensorflow as tf

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for fma dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial Release',
    }

    def scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def load_spectrogram(self, samples, sr):
        sgram = librosa.stft(samples)

        sgram_mag, _ = librosa.magphase(sgram)

        mels = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
        mels = np.log(mels + 1e-9)
        mels = mels.astype(np.uint8)
        return mels, sr

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(fma): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'sgram': tfds.features.Image(shape=(128, 1293, 3)),
                'sample_rate': tfds.features.Scalar(dtype=tf.float32),
                'chromagram': tfds.features.Tensor(shape=(12, None), dtype=tf.float32),
                'path': tfds.features.Text()
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            # Set to `None` to disable
            supervised_keys=('sgram', 'sample_rate',),
            homepage='https://github.com/mdeff/fma',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(fma): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            'https://zenodo.org/records/10364398/files/fma_rock_and_noise_rock.zip?download=1')

        # TODO(fma): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train_rock': self._generate_examples(path / 'fma/train_rock'),
            'train_noise_rock': self._generate_examples(path / 'fma/train_noise_rock'),
            'test_rock': self._generate_examples(path / 'fma/test_rock'),
            'test_noise_rock': self._generate_examples(path / 'fma/test_noise_rock'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(fma): Yields (key, example) tuples from the dataset

        for f in path.glob('*.mp3'):
            pathstring = str(f)
            id = pathstring[-10:]
            key = id[:-3]
            y, sr = librosa.load(pathstring)
            sgram, sample_rate = self.load_spectrogram(y, sr)
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            sgram = np.pad(sgram, [(0, 0), (0, 1293-sgram.shape[1])],
                           mode='constant', constant_values=0)
            sgram = np.dstack((sgram, sgram, sgram))

            yield (key, {
                'sgram': sgram,
                'sample_rate': sample_rate,
                'chromagram': chromagram,
                'path': pathstring,
            })
