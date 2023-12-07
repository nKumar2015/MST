"""fma dataset."""

import tensorflow_datasets as tfds
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for fma dataset."""

    skiplist = ["011298.mp3",
                "021657.mp3",
                "029245.mp3",
                "054568.mp3",
                "054576.mp3",
                "098565.mp3",
                "098567.mp3",
                "098569.mp3",
                "099234.mp3",
                "099134.mp3",
                "108925.mp3",
                "133297.mp3",
                "155066.mp3"]

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial Release',
    }

    def scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def load_spectrogram(self, path):
        samples, sr = librosa.load(path, sr=None)
        sgram = librosa.stft(samples)

        sgram_mag, _ = librosa.magphase(sgram)

        mels = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
        mels = librosa.amplitude_to_db(mels, ref=np.min)
        mels = np.log(mels + 1e-9)
        mels = self.scale_minmax(mels, 0, 255).astype(np.uint8)
        mels = np.dstack((mels, mels, mels))
        return mels, sr

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(fma): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'sgram_image': tfds.features.Image(),
                'sample_rate': tfds.features.Scalar(dtype=tf.float32),
                'genre': tfds.features.ClassLabel(names=['Hip-Hop', 'Pop', 'Folk', 'Avant-Garde', 'Lo-Fi', 'Rock', 'Metal',
                                                         'Post-Punk', 'Krautrock', 'Punk', 'Electroacoustic', 'Reggae - Dub',
                                                         'Latin America', 'International', 'Free-Folk', 'Noise', 'Noise-Rock',
                                                         'Audio Collage', 'Drone', 'Electronic', 'Psych-Folk', 'Field Recordings',
                                                         'Psych-Rock', 'Experimental', 'Progressive', 'Experimental Pop', 'Balkan',
                                                         'Hardcore', 'Singer-Songwriter', 'Polka', 'African', 'French', 'Middle East',
                                                         'Sound Collage', 'Freak-Folk', 'Death-Metal', 'Garage', 'Electro-Punk',
                                                         'Alternative Hip-Hop', 'Compilation', 'Unclassifiable', 'Industrial', 'IDM',
                                                         'New Wave', 'Drum & Bass', 'Holiday', 'Trip-Hop', 'North African', 'Chip Music',
                                                         'Breakcore - Hard', 'Soundtrack', 'Indie-Rock', 'Synth Pop', 'Indian', 'Asia-Far East',
                                                         'Afrobeat', 'Black-Metal', 'Chiptune', 'Rock Opera', 'Space-Rock', 'Brazilian', 'Techno',
                                                         'Ambient Electronic', 'Dubstep', 'Europe', 'Tango', 'House', 'Improv', 'Sound Poetry',
                                                         'Minimal Electronic', 'Breakbeat', 'Novelty', 'No Wave', 'Ambient', 'British Folk', 'Downtempo',
                                                         'Power-Pop', 'Minimalism', 'Surf', 'Celtic', 'Sound Art', 'Post-Rock', 'Klezmer', 'Romany (Gypsy)',
                                                         'Kid-Friendly', 'Salsa', 'Chill-out', 'Shoegaze', 'Hip-Hop Beats', 'Rap', 'Dance', 'Goth', 'Grindcore',
                                                         'Instrumental', 'New Age', 'Glitch', 'Latin', 'Jungle', 'Reggae - Dancehall', 'Turkish', 'Cumbia', 'Nerdcore', 'Loud-Rock']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            # Set to `None` to disable
            supervised_keys=('sgram', 'sample_rate', 'genre'),
            homepage='https://github.com/mdeff/fma',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(fma): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            'https://zenodo.org/records/10223581/files/fma.zip?download=1')

        # TODO(fma): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train_style': self._generate_examples(path / 'fma/train_style', path/"fma/genres.csv"),
            'train_content': self._generate_examples(path / 'fma/train_content', path/"fma/genres.csv"),
            'test_style': self._generate_examples(path / 'fma/test_style', path/"fma/genres.csv"),
            'test_content': self._generate_examples(path / 'fma/test_content', path/"fma/genres.csv"),
        }

    def _generate_examples(self, path, genres):
        """Yields examples."""
        # TODO(fma): Yields (key, example) tuples from the dataset
        data = np.loadtxt(genres, delimiter=",", dtype=str, skiprows=1)
        for f in path.glob('*.mp3'):
            pathstring = str(f)
            id = pathstring[-10:]
            genre = data[data[:, 0] == id][0, 1][3:-1]
            key = id[:-3]
            sgram, sample_rate = self.load_spectrogram(pathstring)

            y, sr = librosa.load(path)
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            if f not in self.skiplist:
                yield (key, {
                    'sgram_image': sgram,
                    'sample_rate': sample_rate,
                    'genre': genre,
                    'chromagram': chromagram,
                })
