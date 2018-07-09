from vad import VoiceActivityDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
import librosa
import pandas as pd
import numpy as np
import os

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def extract_mfcc(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    
    return mfccs

def parse_audio_files(data_dir_train, df):
    features, labels = np.empty((0,40)), np.empty(0)
    for index, row in df.iterrows():
        try:
            file_name = os.path.join(os.path.abspath(data_dir_train), str(row.filename))
            mfccs = extract_mfcc(file_name)
        except Exception as e:
            print(file_name + "\tError: extract feature error. %s" % (e))
            continue
        ext_features = np.hstack([mfccs])
        features = np.vstack([features,ext_features])
        # labels = np.append(labels, fn.split('/')[1])
        labels = np.append(labels, row.target)
    return np.array(features), np.array(labels, dtype = np.int)

def get_score(file_name):
    v = VoiceActivityDetector(file_name)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    
    start = 0
    end = 0
    if len(speech_labels) > 0:
        start = speech_labels[0]['speech_begin']
        end = speech_labels[-1]['speech_end']
    
    return end - start

def parse_audio_files_test(data_dir_test, fclose_task):
    features = np.empty((0,40))
    files = []
    scores = []
    for file in os.listdir(data_dir_test):
        if (fclose_task and 'unknown' in file) \
        or (not fclose_task and not 'unknown' in file):
            continue
        try:
            file_name = os.path.join(os.path.abspath(data_dir_test), file)    
            mfccs = extract_mfcc(file_name)
        except Exception as e:
            print(file_name + "\tError: extract feature error. %s" % (e))
            continue
        ext_features = np.hstack([mfccs])
        features = np.vstack([features,ext_features])
        scores.append(get_score(file_name)) 
        files.append(file)
        # labels = np.append(labels, fn.split('/')[1])
    return np.array(features), files, scores

def classify(X_train, y_train, X_test, files, scores, class_list):
    # Grid search for best parameters
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [10, 50, 100, 150, 200],'max_depth': [3, 5, 10, 20, 30, 40, 50]}]
    
    print("# Tuning hyper-parameters for accuracy")
    print('')

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy', verbose=2)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print('')
    print(clf.best_params_)
    print('')
    print("Grid scores on development set:")
    print('')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('')

    y_pred = clf.predict(X_test)

    with open('result.txt', 'a') as f:
        for i in range(len(files)):
            f.write('{0}\t{1}\t{2}'.format(files[i], str(scores[i]), class_list[y_pred[i] - 1]))

def classify_files(dir_train, file_train, dir_test, fclose_task):
    with open(file_train, 'r') as f:
        data = [row.split('\t') for row in f]
    if fclose_task:
        df = pd.DataFrame(data, columns=['filename', 'a', 'b', 'c', 'class'])
    else:
        df = pd.DataFrame(data, columns=['filename', 'class'])
    class_list = list(Counter(df['class']).keys())
    df['target'] = df['class'].map(lambda x: class_list.index(x) + 1)
    
    X_train, y_train = parse_audio_files(dir_train, df)
    X_test, files, scores = parse_audio_files_test(dir_test, fclose_task)
    
    classify(X_train, y_train, X_test, files, scores, class_list)

dir_train_close_task = 'audio/'
dir_train_open_task = 'audio2/'
dir_test = 'test/'
file_train_close_task = 'meta/meta.txt'
file_train_open_task = 'meta/meta_unknown_train.txt'

classify_files(dir_train_close_task, file_train_close_task, dir_test, True)
classify_files(dir_train_open_task, file_train_open_task, dir_test, False)
