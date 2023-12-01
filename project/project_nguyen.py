# import libraries
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import fft,ifft
from scipy.signal.windows import hamming
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,Input,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics

max_sample_size = 192 # max size of each audio sample
sample_rate = 44100 
dt = 0.032 # time interval
win_size = int(sample_rate * dt) # window size
n_fft = win_size # num of sample in fft
hop_length = int(0.5 * win_size) # hop_length => 50% overlap
n_mels = 192 # number of Mel bin - as number of features
num_sample_with_hop = max_sample_size * hop_length
window=hamming(win_size)

train_data_dir = "data/train_data"
test_data_dir = "data/test_data"
labels = ["cars", "trams"]
train_car_audio_list = os.listdir(os.path.join(train_data_dir, labels[0]))
train_car_audio_list.sort()
train_tram_audio_list = os.listdir(os.path.join(train_data_dir, labels[1]))
train_tram_audio_list.sort()
test_car_audio_list = os.listdir(os.path.join(test_data_dir, labels[0]))
test_car_audio_list.sort()
test_tram_audio_list = os.listdir(os.path.join(test_data_dir, labels[1]))
test_tram_audio_list.sort()

def get_mel_feature(audio_file):
    sample, _ = librosa.load(audio_file, sr=sample_rate)
    if (sample.shape[0] < num_sample_with_hop):
        print(audio_file)
        return None
        
    sample = sample[:num_sample_with_hop]
    # calculate mel spectrogram
    S = librosa.feature.melspectrogram(y=sample, sr=sample_rate, n_fft=n_fft, 
                                       hop_length=hop_length, n_mels=n_mels,
                                       window=hamming)
    S_DB = librosa.power_to_db(S, ref=np.max)
    # print(S_DB.shape)
    
    # scale
    max_db = np.max(S_DB)
    min_db = np.min(S_DB)
    S_DB = (S_DB - min_db) / (max_db - min_db)

    return S_DB

# Load data from directory
# load_mode: 1-load all, 2-train only, 3-test_only, 4-specific test sample index
# load_sample: the index used in loading specfic test sample for both classes
def load_data(load_mode=4,load_sample=5):
    X_train, y_train, X_test, y_test = None, None, None, None
    if (load_mode == 1 or load_mode == 2):
        X_train, y_train = np.empty((0, n_mels, max_sample_size, 1)), np.empty((0,1))
        for i, car_audio in enumerate(train_car_audio_list):
            audio_file = os.path.join(train_data_dir, labels[0], car_audio)
            feature = get_mel_feature(audio_file)
            if (feature is not None):
                feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
                X_train = np.append(X_train, feature, axis=0)
                y_train = np.append(y_train, np.array([[0]]), axis=0)
            
        for i, tram_audio in enumerate(train_tram_audio_list):
            audio_file = os.path.join(train_data_dir, labels[1], tram_audio)
            feature = get_mel_feature(audio_file)
            if (feature is not None):
                feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
                X_train = np.append(X_train, feature, axis=0)
                y_train = np.append(y_train, np.array([[1]]), axis=0)
        X_train, y_train = shuffle(X_train, y_train, random_state=1112)

    # LOAD TEST DATA
    if (load_mode == 1 or load_mode == 3 or load_mode == 4):
        X_test, y_test = np.empty((0, n_mels, max_sample_size, 1)), np.empty((0,1))
        if (load_mode != 4):
            for i, car_audio in enumerate(test_car_audio_list):
                audio_file = os.path.join(test_data_dir, labels[0], car_audio)
                feature = get_mel_feature(audio_file)
                if (feature is not None):
                    feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
                    X_test = np.append(X_test, feature, axis=0)
                    y_test = np.append(y_test, np.array([[0]]), axis=0)
            
            for i, tram_audio in enumerate(test_tram_audio_list):
                audio_file = os.path.join(test_data_dir, labels[1], tram_audio)
                feature = get_mel_feature(audio_file)
                if (feature is not None):
                    feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
                    X_test = np.append(X_test, feature, axis=0)
                    y_test = np.append(y_test, np.array([[1]]), axis=0)
        else:
            audio_file = os.path.join(test_data_dir, labels[0], test_car_audio_list[load_sample])
            feature = get_mel_feature(audio_file)
            feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
            X_test = np.append(X_test, feature, axis=0)
            y_test = np.append(y_test, np.array([[0]]), axis=0)
            audio_file = os.path.join(test_data_dir, labels[1], test_tram_audio_list[load_sample])
            feature = get_mel_feature(audio_file)
            feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=-1)
            X_test = np.append(X_test, feature, axis=0)
            y_test = np.append(y_test, np.array([[1]]), axis=0)
    
    return X_train, y_train, X_test, y_test

def visualize_sound_features(car_audio_path=None, tram_audio_path=None):
    if (car_audio_path is None or tram_audio_path is None):
        print("Need path to both classes")
        return
    
    audio_file_car = os.path.join(test_data_dir, labels[0], car_audio_path)
    audio_file_tram = os.path.join(test_data_dir, labels[1], tram_audio_path)
    sample_car, _  = librosa.load(audio_file_car,  sr=sample_rate)
    sample_tram, _ = librosa.load(audio_file_tram, sr=sample_rate)

    # Plot signal in time domain
    fig, ax = plt.subplots(2,figsize=(8, 5))
    fig.tight_layout(h_pad=2)
    fig.suptitle("CAR AND TRAM SIGNALS IN TIME DOMAIN", y=0.025, fontsize=15)
    ax[0].plot(sample_car[5000:15000])
    ax[0].set_xlabel("Time (samples)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Car sample sound")
    ax[1].plot(sample_tram[5000:15000])
    ax[1].set_xlabel("Time (samples)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Tram sample sound")

    # One window FFT spectrum of car and tram
    fft_sample_car = np.abs(fft(sample_car[5000:5000+win_size]*window, n_fft))
    fft_sample_tram = np.abs(fft(sample_tram[5000:5000+win_size]*window, n_fft))
    fig, ax = plt.subplots(2,figsize=(8, 5))
    fig.tight_layout(h_pad=3)
    fig.suptitle("CAR AND TRAM SEGMENTS IN FREQUENCY DOMAIN", y=0.025, fontsize=15)
    ax[0].plot(fft_sample_car[:500])
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Spectrum")
    ax[0].set_title("Car sample FFT")
    ax[1].plot(fft_sample_tram[:500])
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Spectrum")
    ax[1].set_title("Tram sample FFT")

    # Car and Tram FFT spectrogram, log-scaled spectrogram and mel-scaled spectrogram
    fig, ax = plt.subplots(2,3,figsize=(12, 5))
    fig.tight_layout(h_pad=3, w_pad=5)
    fig.suptitle("CAR AND TRAM MULTI-SCALES SPECTROGRAMS", y=0.025, fontsize=15)
    # CAR SPECTROGRAM
    D = np.abs(librosa.stft(sample_car, n_fft=n_fft,  hop_length=hop_length, window=hamming))
    im_car = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='linear', ax=ax[0][0])
    ax[0][0].set_xlabel("Time (s)")
    ax[0][0].set_ylabel("Frequency (Hz)")
    ax[0][0].set_title("Car spectrogram")
    plt.colorbar(im_car, ax=ax[0][0])

    # CAR LOG SCALED SPECTROGRAM
    DB = librosa.amplitude_to_db(D, ref=np.max)
    im_car = librosa.display.specshow(DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0][1]);
    ax[0][1].set_xlabel("Time (s)")
    ax[0][1].set_ylabel("Frequency (Hz)")
    ax[0][1].set_title("Car spectrogram in log scale (dB)")
    plt.colorbar(im_car, ax=ax[0][1], format='%+2.0f dB')

    # CAR MEL SPECTROGRAM
    S = librosa.feature.melspectrogram(y=sample_car, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=hamming)
    S_DB = librosa.power_to_db(S, ref=np.max)
    im_car = librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax[0][2]);
    ax[0][2].set_xlabel("Time (s)")
    ax[0][2].set_ylabel("Frequency (Hz)")
    ax[0][2].set_title("Car spectrogram in mel scale (dB)")
    plt.colorbar(im_car, ax=ax[0][2], format='%+2.0f dB');

    # TRAM SPECTROGRAM
    D = np.abs(librosa.stft(sample_tram, n_fft=n_fft,  hop_length=hop_length, window=hamming))
    im_tram = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='linear', ax=ax[1][0])
    ax[1][0].set_xlabel("Time (s)")
    ax[1][0].set_ylabel("Frequency (Hz)")
    ax[1][0].set_title("Tram spectrogram")
    plt.colorbar(im_tram,ax=ax[1][0])

    # TRAM LOG SCALED SPECTROGRAM
    DB = librosa.amplitude_to_db(D, ref=np.max)
    im_tram = librosa.display.specshow(DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[1][1]);
    ax[1][1].set_xlabel("Time (s)")
    ax[1][1].set_ylabel("Frequency (Hz)")
    ax[1][1].set_title("Tram spectrogram in log scale (dB)")
    plt.colorbar(im_tram, ax=ax[1][1], format='%+2.0f dB')

    # CAR MEL SPECTROGRAM
    S = librosa.feature.melspectrogram(y=sample_tram, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=hamming)
    S_DB = librosa.power_to_db(S, ref=np.max)
    im_tram = librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax[1][2]);
    ax[1][2].set_xlabel("Time (s)")
    ax[1][2].set_ylabel("Frequency (Hz)")
    ax[1][2].set_title("Tram spectrogram in mel scale (dB)")
    plt.colorbar(im_tram, ax=ax[1][2], format='%+2.0f dB');
    plt.show()

def define_model(input_shape=(192,192,1), opt=Adam()):
    model_cnn = Sequential()
    model_cnn.add(Input(input_shape))
    model_cnn.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(32, activation="relu"))
    model_cnn.add(Dropout(0.1))
    model_cnn.add(Dense(16, activation="relu"))
    model_cnn.add(Dropout(0.1))
    model_cnn.add(Dense(1, activation="sigmoid"))
    model_cnn.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])
    model_cnn.summary()
    return model_cnn

def save_model(model_cnn, path):
    model_cnn.save(path)

def load_model(path):
    model_cnn = tf.keras.models.load_model(path)
    return model_cnn

def train_model(model_cnn, X_train, y_train):
    batch_size = 16
    callback = EarlyStopping(monitor='val_loss', patience=5)
    epochs = 10
    history = model_cnn.fit(X_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        # validation_data=(X_val, y_val),
                        validation_split=0.1,
                        callbacks=[callback])
    return history

def plot_history(history):
    fig, ax = plt.subplots(1,2,figsize=(16, 5))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy %')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Val'], loc='upper left')
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Val'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    # LOAD DATA
    X_train, y_train, X_test, y_test = load_data(load_mode=1)
    # _, _, X_test, y_test = load_data(load_mode=4, load_sample=5)

    # TRAINING
    # Init model
    model_cnn = define_model(input_shape=(192,192,1), opt=Adam(0.0001))

    # Fit model with training data
    history = train_model(model_cnn, X_train, y_train)

    # Plot training history
    plot_history(history)

    # Evaluation
    train_losses, train_acc = model_cnn.evaluate(X_train, y_train)
    test_losses, test_acc = model_cnn.evaluate(X_test, y_test)
    print("Training: Accuracy ", round(train_acc*100.0,2), " - Losses ", round(train_losses,2))
    print("Testing : Accuracy ", round(test_acc*100.0,2) , " - Losses ", round(test_losses,2))

    # # Save model
    # save_model(model_cnn, path="cars_trams_model_ver2.keras")

    # TESTING ON SAMPLES
    # Visualize data
    # load_sample=5
    # visualize_sound_features(car_audio_path=test_car_audio_list[load_sample],
    #                          tram_audio_path=test_tram_audio_list[load_sample])
    
    # # Load model
    # model_cnn = load_model(path="cars_trams_model.keras")
    
    # # Test model
    # y_test_pred = model_cnn.predict(X_test)
    # y_test_pred[y_test_pred>0.5] = 1
    # y_test_pred[y_test_pred<=0.5] = 0
    # test_labels = [labels[int(y_test[0])], labels[int(y_test[1])]]
    # pred_labels = [labels[int(y_test_pred[0])], labels[int(y_test_pred[1])]]
    # print("Groundtruth|    ", test_labels[0], "    ", test_labels[1])
    # print("Prediction |    ", pred_labels[0], "    ", pred_labels[1])



