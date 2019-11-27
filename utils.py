import librosa
import numpy as np

def get_mfccs(url):
    '''Extracts mfccs of wav file.
    
    Parameters
    ----------
    url : string
        The path of the wav file.

    Returns
    -------
    mfccs : array-like, shape (n_features, n_frames)
        The n_features nomally are 39.
    '''

    y, sr = librosa.load(url, sr=16000) # sr:采样率
    mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 39)

    return mfccs

def load_data_and_normalize(i, n_features = 39):
    '''Load 14 samples of number i.

    Parameters
    ----------
    i : int
        The number said in wav file.

    n_features : int, defaults to 39

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        The data array.
    '''
  
    X = np.zeros((0, n_features), dtype = float)

    root = "records/digit_" + str(i)
    for num in range(1, 15):
        url = root + "/" + str(num) + "_" + str(i) + ".wav"
        mfccs = get_mfccs(url)
        X = np.concatenate((X, mfccs.T), axis = 0)

    # Normalize the data array X
    # X += np.finfo(float).eps
    X_sum = X.sum(0)
    X_sum[X_sum == 0] = 1 # Make sure we don't divide by zero.
    Y = X / X_sum + np.finfo(float).eps
    return Y