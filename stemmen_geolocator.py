#!/usr/bin/python3
"""
File:       stemmen_geolocator.py
Author:     Martijn E.N.F.L. Schendstok
Date:       April 2020
"""

import sys
import argparse
import pickle
import random
import operator
import librosa
from numba import np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from statistics import mean


def parse_arguments():
    """
    Read arguments from a command line
    :return:    args
    """

    parser = argparse.ArgumentParser(description="Geo-lacation multi-lable classifier, "
                                                 "based on the Stemmen dataset. "
                                                 "Optional classifiers: RandomForest, LinearSVM, SVM, KNN")
    parser.add_argument("-develop",
                        metavar="CLASSIFIER",
                        help="Run classification development for given classifier "
                             "(RandomForest, LinearSVM, SVM, KNN)")
    parser.add_argument("-test",
                        metavar="CLASSIFIER",
                        help="Run classification test for given classifier "
                             "(RandomForest, LinearSVM, SVM, KNN)")
    args = parser.parse_args()

    # test compatibility of parameters
    if args.develop and args.test:
        raise RuntimeError("-develop and -test can not be called at the same time.")
    if args.develop and args.develop not in ["RandomForest", "LinearSVM", "SVM", "KNN", "ALL"]:
        raise RuntimeError("Classifier not implemented, choose RandomForest, LinearSVM, SVM or KNN.")
    if args.test and args.test not in ["RandomForest", "LinearSVM", "SVM", "KNN"]:
        raise RuntimeError("Classifier not implemented, choose RandomForest, LinearSVM, SVM or KNN.")

    return args


def load_pickle(file_name):
    """
    Loads and returns a dictionary with the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     string
                                                     MAUS_input_sampa:  string
                                                     MAUS_output_IPA:   string
                                                     audio_file:        x.wav}}
                 longitude:         float
                 latitude:          float
                 selected_location: string
                 province:          string
                 age:               string
                 gender:            string
                 municipality:      string}
    :param file_name:   The desired file to load (without .pickle)
    :return:            dict
    """

    with open("Data/" + file_name +'.pickle', 'rb') as file:
        data = pickle.load(file)
    return data


def get_encode_dict(data):
    """
    Makes encode dict for one hot encoding from all characters that occur.
    :param data:    dict
    :return:        dict
    """

    encode_list = list()
    for sessionID in data:
        for word in data[sessionID]['words']:
            outputIPA = data[sessionID]['words'][word]['MAUS_output_IPA']
            encode_list += list(outputIPA)

    encode_dict = dict.fromkeys(encode_list)
    for n, key in enumerate(encode_dict):
        encode_dict[key] = n

    return encode_dict


def one_hot_encode(word, encode_dict):
    """
    One hot encodes the characters in a word
    :param word:            string
    :param encode_dict:     dict
    :return:                list
    """

    word_encoded = [0] * len(encode_dict)
    for n, char in enumerate(word):
        word_encoded[encode_dict[char]] = 1

    return word_encoded


def add_encoded_word(data, encode_dict):
    """
    Adds one_hot_encoded word to data dictionary in the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     string
                                                     MAUS_input_sampa:  string
                                                     MAUS_output_IPA:   string
                                                     audio_file:        x.wav
                                                     encoded:           list(int)}}
                 longitude:         float
                 latitude:          float
                 selected_location: string
                 province:          string
                 age:               string
                 gender:            string
                 municipality:      string}
    :param data:        dict
    :param encode_dict: dict
    :return data:       dict
    """

    for sessionID in data:
        for word in data[sessionID]['words']:
            word_encoded = one_hot_encode(data[sessionID]['words'][word]['MAUS_output_IPA'],
                                          encode_dict)
            data[sessionID]['words'][word]['encoded'] = word_encoded

    return data


def add_spectral_features(data):
    """
    Adds spectral features to data dictionary in the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     string
                                                     MAUS_input_sampa:  string
                                                     MAUS_output_IPA:   string
                                                     audio_file:        x.wav
                                                     encoded:           list(int)
                                                     spectral_features: {spectral_centroid:     float
                                                                         spectral_bandwidth:    float
                                                                         rms:                   float
                                                                         spectral_rolloff:      float
                                                                         zero_crossing_rate:    float
                                                                         chroma_stft:           list(float)
                                                                         mfcc:                  list(float)
                                                                         tempo:                 float}}}
                 longitude:         float
                 latitude:          float
                 selected_location: string
                 province:          string
                 age:               string
                 gender:            string
                 municipality:      string}
    :param data:    dict
    :return data:   dict
    """

    for sessionID in data:
        spectral_features = {}
        for word in data[sessionID]['words']:
            y, sr = librosa.load("Data/BA Stemmen/Stemmen audio/" + data[sessionID]['words'][word]["audio_file"])

            spectral_features["spectral_centroid"] = mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            spectral_features["spectral_bandwidth"] = mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
            #spectral_features["rms"] = mean(librosa.feature.rms(y=y)[0])
            #spectral_features["spectral_rolloff"] = mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
            spectral_features["zero_crossing_rate"] = mean(librosa.feature.zero_crossing_rate(y=y)[0])

            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_features["chroma_stft"] = list()
            for e in chroma_stft:
                spectral_features["chroma_stft"].append(mean(e))

            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            spectral_features["mfcc"] = list()
            for e in mfcc:
                spectral_features["mfcc"].append(mean(e))

            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            spectral_features["tempo"] = tempo

            #print(spectral_features)

            data[sessionID]['words'][word]["spectral_features"] = spectral_features

    return data


def get_feats_train(data, seperate=True, encoded=True, spectral=True):
    """
    Returns features-list and labels-list for municipalities and provinces
    :param data:                                dict
    :param seperate:                            Boolean
    :param encoded:                             Boolean
    :param spectral:                            Boolean
    :return feats, municipalities, provinces:   list, list, list
    """

    feats = list()
    municipalities = list()
    provinces = list()

    for sessionID in data:
        words_array = list()
        for word in data[sessionID]['words']:
            if encoded:
                word_feat = data[sessionID]['words'][word]['encoded']
            else:
                word_feat = data[sessionID]['words'][word]['MAUS_output_IPA']

            spectral_features = list()
            if spectral:
                for sf in data[sessionID]['words'][word]["spectral_features"]:
                    if isinstance(data[sessionID]['words'][word]["spectral_features"][sf], list):
                        spectral_features += data[sessionID]['words'][word]["spectral_features"][sf]
                    else:
                        spectral_features.append(data[sessionID]['words'][word]["spectral_features"][sf])

            if seperate:
                feats.append(word_feat + spectral_features)
                municipalities.append(data[sessionID]['municipality'])
                provinces.append(data[sessionID]['province'])
            else:
                words_array += word_feat + spectral_features

        if not seperate:
            feats.append(words_array)
            municipalities.append(data[sessionID]['municipality'])
            provinces.append(data[sessionID]['province'])

    return feats, municipalities, provinces


def get_folds(data, folds=10):
    """
    Makes n-fold (standard set to 10) for n-fold cross validation.
    The data is randomised and then split in train and test sets n-times.
    Where the test-set is 100%/n and train is 100%-100%/n.
    :param data:        dict
    :param folds:       int
    :return n_folds:    list((train_dict, test_dict) x folds)
    """

    # Randomnize sessionID order:
    sessionID_list = list(data.keys())
    random.shuffle(sessionID_list)

    # Get folds:
    n_folds = list()
    for n in range(folds):
        # Get test feats:
        test_feat_ids = sessionID_list[int(n / folds * len(sessionID_list)):int((n + 1) / folds * len(sessionID_list))]
        test_dict = {key:data[key] for key in data if key in test_feat_ids}

        # Get train feats:
        train_dict = {key: data[key] for key in data if key not in test_feat_ids}

        n_folds.append((train_dict, test_dict))

    return n_folds


def location_count_dict(data, location_level):
    """
    Counts the occurrence per location and saves it in a dict.
    :param data:            Data dict
    :param location_level:  String with selected_location, municipality, or province
    :return:                Dict with count per location
    """

    dict = defaultdict(int)

    for key in data:
        dict[data[key][location_level]] += 1

    return dict


def train_classifier(train_data, feats, classifications, location_level, classifier="RandomForest"):
    """
    Trains the desired classifier. Possible classifiers are: RadomForest, LinearSVM, SVM, KNN.
    :param train_data:          dict
    :param feats:               list
    :param classifications:     list
    :param location_level:      string
    :param classifier:          string
    :return clf:                trained classifier
    """

    # class_count = location_count_dict(train_data, location_level)
    # maximum = max(class_count.values())
    # class_weight = {key:(maximum / class_count[key]) for key in class_count}

    if classifier == "RandomForest":
        clf = RandomForestClassifier()  # class_weight=class_weight)
    elif classifier == "LinearSVM":
        clf = LinearSVC(dual=False)  # class_weight=class_weight)
    elif classifier == "SVM":
        clf = SVC()  # class_weight=class_weight)
    elif classifier == "KNN":
        clf = KNeighborsClassifier()
    else:
        print("Error: train_classifier() is called for a classifier that is not implemented",
              file=sys.stderr)
        return None

    clf.fit(feats, classifications)
    return clf


def majority_vote(list):
    """
    Returns the majority vote in a list, the most occurring string.
    :param list:    list
    :return:        string
    """

    dict = defaultdict(int)
    for x in list:
        dict[x] += 1
    return max(dict.items(), key=operator.itemgetter(1))[0]


def classifier_predict(data, classifier, location_lvl, encoded=True, spectral=True):
    """
    Predicts location label for given location level (municipality or province)
    and appends the predicted label to the data dictionary in the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     string
                                                     MAUS_input_sampa:  string
                                                     MAUS_output_IPA:   string
                                                     audio_file:        x.wav
                                                     encoded:           list(int)
                                                     spectral_features: {spectral_centroid:     float
                                                                         spectral_bandwidth:    float
                                                                         rms:                   float
                                                                         spectral_rolloff:      float
                                                                         zero_crossing_rate:    float
                                                                         chroma_stft:           list(float)
                                                                         mfcc:                  list(float)
                                                                         tempo:                 float}}}
                 longitude:         float
                 latitude:          float
                 selected_location: string
                 province:          string
                 age:               string
                 gender:            string
                 municipality:      string
                 prediction_x:      string}
    :param data:            dict
    :param classifier:      trained_classifier
    :param location_lvl:    string
    :param encoded:         Boolean
    :param spectral:        Boolean
    :return data:           dict
    """

    for sessionID in data:
        x = list()
        for word in data[sessionID]['words']:
            if encoded:
                word_feat = data[sessionID]['words'][word]['encoded']
            else:
                word_feat = data[sessionID]['words'][word]['MAUS_output_IPA']

            spectral_features = list()
            if spectral:
                for sf in data[sessionID]['words'][word]["spectral_features"]:
                    if isinstance(data[sessionID]['words'][word]["spectral_features"][sf], list):
                        spectral_features += data[sessionID]['words'][word]["spectral_features"][sf]
                    else:
                        spectral_features.append(data[sessionID]['words'][word]["spectral_features"][sf])

            x.append(word_feat + spectral_features)
        predicted = classifier.predict(x)
        #print(predicted)
        data[sessionID]["prediction_{0}".format(location_lvl)] = majority_vote(predicted)

    return data


def get_accuracy(data, location_lvl):
    """
    Returns the accuracy of classification for the given location level (municipality or province).
    :param data:            dict
    :param location_lvl:    string
    :return:                float
    """

    total = 0
    true = 0
    for sessionID in data:
        if data[sessionID][location_lvl] == data[sessionID]["prediction_{0}".format(location_lvl)]:
            true += 1
        total += 1

    return float(true/total)


def develop(classifier, data=None, fold_info=True):
    print("### Classifier: {}".format(classifier))
    if not data:
        data = load_pickle("stemmen_train")
        print("## Amount of participants: {}".format(len(data)))
        encode_dict = get_encode_dict(data)
        data = add_encoded_word(data, encode_dict)
        print("# Get spectral features:", end=" ")
        data = add_spectral_features(data)
        print("Done")

    if fold_info: print("\n#### 10-fold cross-validation accuracies ####")
    accuracy_list_muni = list()
    accuracy_list_prov = list()
    for i, (train_dict, test_dict) in enumerate(get_folds(data)):
        if fold_info: print("## Fold {0}".format(i + 1))

        if fold_info: print("# Training classifier:", end=" ")
        train_feats, train_municipalities, train_provinces = get_feats_train(train_dict)
        if fold_info: print("Done")

        municipality_classifier = train_classifier(train_dict,
                                                   train_feats,
                                                   train_municipalities,
                                                   "municipality",
                                                   classifier=classifier)
        province_classifier = train_classifier(train_dict,
                                               train_feats,
                                               train_provinces,
                                               "province",
                                               classifier=classifier)

        test_dict = classifier_predict(test_dict, municipality_classifier, "municipality")
        test_dict = classifier_predict(test_dict, province_classifier, "province")

        accuracy_list_muni.append(get_accuracy(test_dict, "municipality"))
        if fold_info: print("Accuracy municipalities: {0}".format(accuracy_list_muni[-1]))
        accuracy_list_prov.append(get_accuracy(test_dict, "province"))
        if fold_info: print("Accuracy provinces: {0}\n".format(accuracy_list_prov[-1]))

    print("Mean accuracy municipalities: {0}".format(sum(accuracy_list_muni) / len(accuracy_list_muni)))
    print("Mean accuracy provinces: {0}\n".format(sum(accuracy_list_prov) / len(accuracy_list_prov)))


def test(classifier):
    train_data = load_pickle("stemmen_train")
    encode_dict = get_encode_dict(train_data)
    train_data = add_encoded_word(train_data, encode_dict)
    train_data = add_spectral_features(train_data)

    test_data = load_pickle("stemmen_test")
    test_data = add_encoded_word(test_data, encode_dict)
    test_data = add_spectral_features(test_data)

    print("# Training Classifier:", end=" ")
    train_feats, train_municipalities, train_provinces = get_feats_train(train_data)
    print("Done")

    municipality_classifier = train_classifier(train_data,
                                               train_feats,
                                               train_municipalities,
                                               "municipality",
                                               classifier=classifier)
    province_classifier = train_classifier(train_data,
                                           train_feats,
                                           train_provinces,
                                           "province",
                                           classifier=classifier)

    test_data = classifier_predict(test_data, municipality_classifier, "municipality")
    test_data = classifier_predict(test_data, province_classifier, "province")

    print("Accuracy municipalities: {0}".format(get_accuracy(test_data, "municipality")))
    print("Accuracy provinces: {0}\n".format(get_accuracy(test_data, "province")))


if __name__ == "__main__":
    args = parse_arguments()
    if args.develop:
        if args.develop == "ALL":
            print("### Running all classifiers")
            data = load_pickle("stemmen_train")
            print("## Amount of participants: {}".format(len(data)))
            encode_dict = get_encode_dict(data)
            data = add_encoded_word(data, encode_dict)
            print("# Get spectral features:", end=" ")
            data = add_spectral_features(data)
            print("Done\n")
            for classifier in ["RandomForest", "LinearSVM", "SVM", "KNN"]:
                develop(classifier, data=data, fold_info=False)
        else:
            develop(args.develop)
    elif args.test:
        test(args.test)
    else:
        print("No parameters given. Use '-h' for help.")
