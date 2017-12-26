### [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/)

![](images/tensorflow.png)

[Simple Audio Recognition using Tensorflow](https://www.tensorflow.org/versions/master/tutorials/audio_recognition)

---
note:
reedit photo that day
---

### About the Speech Commands Datasets

- Size

        test.7z --> 2.46 GB
        train.7z --> 1.04 GB

- Contents

    + Includes 65,000 one-second long utterances

    + Each containing a single spoken English word（thirty different words）

            yes，no，up，down，left，right，on，off，stop，go，silence ...

    + From thousands of different people

---

### Target

Use the Speech Commands Dataset

to

build an algorithm

that

understands simple spoken commands

---

### Algorithms that may be used

- Audio Data Conversion to Images

- Single WAV + FFT + Entropy

- Light-Weight CNN

---

### Speech representation and data exploration

- Visualization of the recordings - input features - 1/2

    + Wave and spectrogram
    + MFCC
    + Sprectrogram in 3d
    + Silence removal
    + Resampling - dimensionality reductions
    + Features extraction steps

---

- Dataset investigation - 2/2

    + Number of files
    + Mean spectrograms and fft
    + Deeper into recordings
    + Length of recordings
    + Note on Gaussian Mixtures modeling
    + Frequency components across the words
    + Anomaly detection
