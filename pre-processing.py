#!/usr/bin/python3
"""
File:       pre-processing.py
Author:     Martijn E.N.F.L. Schendstok
Date:       April 2020
"""

import sys
import glob
from collections import defaultdict
import random
import csv
import pickle


def get_stemmen_data(file):
    """
    Gets data from file and puts it in a dict with the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     x
                                                     MAUS_input_sampa:  x
                                                     MAUS_output_IPA:   x
                                                     audio_file:        x.wav}}
                 longitude:         x
                 latitude:          x
                 selected_location: x
                 province:          x
                 age:               x
                 gender:            x}
    :param file:    filepath for csv-file containing stemmen data
    :return data:   dict with data
    """

    stemmen_dict = {}
    with open(file) as f:
        for i, line in enumerate(f):
            if i == 0:
                keys = [key.strip().strip('"') for key in line.split(',')]
            else:
                item_dict = {'words':{}}
                word_dict = {}
                for j, item in enumerate(line.split(',')):
                    if j == 0:
                        sessionID = item.strip().strip('"')
                    elif j == 1:
                        word = item.strip().strip('"')
                    elif j <= 4:
                        word_dict[keys[j]] = item.strip().strip('"')
                    else:
                        item_dict[keys[j]] = item.strip().strip('"')

                word_dict["audio_file"] = "{0}_{1}-extracted.wav".format(sessionID, word.replace(" ", "-"))
                if sessionID in stemmen_dict:
                    stemmen_dict[sessionID]['words'][word] = word_dict
                else:
                    item_dict['words'][word] = word_dict
                    stemmen_dict[sessionID] = item_dict

    return stemmen_dict


def filter_Low_Saxon_Area(data_original, dir):
    """
    Remove the data items outside the Dutch Low Saxon area
    :param data_original:   dict
    :param dir:             dir path where placename text files are
    :return data:           filtered dict
    """

    # Get list of places within the Dutch Low Saxon area:
    provinces_Low_Saxon_Area = ["Groningen", "Drenthe", "Overijssel"]
    places_Low_Saxon_Area = []
    for file in glob.glob(dir + "/*.txt"):
        with open(file) as f:
            for line in f:
                places_Low_Saxon_Area.append(line.strip())

    # Remove data outside the Dutch Low Saxon area:
    data = dict(data_original)
    for key in data_original:
        if not data_original[key]['province'] in provinces_Low_Saxon_Area:
            if not data_original[key]['selected_location'] in places_Low_Saxon_Area or data_original[key]['province'] not in ["Gelderland",
                                                                                                                              "Fryslân",
                                                                                                                              "Utrecht",
                                                                                                                              "Flevoland"]:
                #print(key, data_original[key]['province'], data_original[key]['selected_location'])
                del data[key]

    return data


def get_municipalities(data, file):
    """
    Append the corresponding municipality to each data item,
    returns a dictionary with the following structure:
    {SessionID:  {words:            {10 * word_x:   {wordChoiceIPA:     x
                                                     MAUS_input_sampa:  x
                                                     MAUS_output_IPA:   x
                                                     audio_file:        x.wav}}
                 longitude:         x
                 latitude:          x
                 selected_location: x
                 province:          x
                 age:               x
                 gender:            x
                 municipality:      x}
    :param data:    dict
    :param file:    filepath for csv-file containing municipality per city info
    :return data:   dict
    """

    # Get city: municipality dictionary:
    city_municipalities_data = {}
    with open(file) as f:
        for i, line in enumerate(f):
            if i > 1:
                line_data = [x.strip('\ufeff').strip('"') for x in line.split(';')]
                city_municipalities_data[line_data[0]] = line_data[2]
    city_municipalities_data['Het Hogeland'] = 'HetHogeland'  # Someone put municipality as selected_location

    # Append corresponding municipality to data:
    for key in data:
        # print(data[key])
        if data[key]['selected_location']:
            data[key]['municipality'] = city_municipalities_data[data[key]['selected_location']]
        else:
            data[key]['municipality'] = None
            print("Could not find municipality for:", data[key],
                  file=sys.stderr)

    return data


def fix_missing_words(data_original):
    """
    Removes participants with less than 5 words.
    Appends donor words from participants with in the same municipality randomly.
    :param data_original:   dict
    :return: data:          dict
    """
    data = dict(data_original)
    fix_dict = {}
    for key in data_original:
        # Remove participants with less than 5 words:
        if len(data_original[key]["words"]) < 5:
            del data[key]
        # Add participants with less than 10 words, but more than 5, to a fix dict:
        elif len(data_original[key]["words"]) < 10:
            fix_dict[key] = data_original[key]["municipality"]
            # print(key, data[key]["municipality"], len(data[key]["words"]))

    for key in fix_dict:
        # Get possible donor word lists, saved in a dict, from participants
        # within the same municipality:
        donor_word_dict = defaultdict(list)
        for key2 in data:
            if data_original[key2]["municipality"] == fix_dict[key] and key != key2:
                for word in data_original[key2]['words']:
                    donor_word_dict[word].append(data_original[key2]['words'][word])

        # Remove participants with no other participants within the
        # same municipality to take donor words from:
        if len(donor_word_dict['twee']) == 0:
            del data[key]
        else:
            # Append random donor word:
            for word in donor_word_dict:
                if word not in data_original[key]['words']:
                    data[key]['words'][word] = random.choice(donor_word_dict[word])
            # Check if it has 10 words:
            if len(data[key]["words"]) < 10:
                del data[key]

    return data


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


def save_csv_file(data, file_name):
    """
    Saves data as a csv file with an entry per word like the original stemmen csv
    :param data:        dict
    :param file_name:   The desired name for the file (without .csv)
    """
    with open("Data/" + file_name + ".csv", "w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["SessionID",
                         "longitude",
                         "latitude",
                         "selected_location",
                         "province",
                         "age",
                         "gender",
                         "municipality",
                         "word",
                         "wordChoiceIPA",
                         "MAUS_input_sampa",
                         "MAUS_output_IPA",
                         "audio_file"
                         ])
        for key in data:
            row = [key]
            for item in data[key]:
                if item != "words":
                    row.append(data[key][item])
            for word in data[key]["words"]:
                word_list = [word]
                for word_item in data[key]["words"][word]:
                    word_list.append(data[key]["words"][word][word_item])
                writer.writerow(row + word_list)


def save_pickle_file(data, file_name):
    """
    Saves data as a pickle file
    :param data:        dict
    :param file_name:   The desired name for the file (without .pickle)
    """
    with open("Data/" + file_name +'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_std_dict(dict):
    """
    Prints a standard formatted dict with only one layer.
    :param dict:
    """
    for key in dict:
        print("{0:<27}{1:>6}".format(key, dict[key]))
    print()


def main(argv):
    # Pre-process data:
    data = get_stemmen_data(argv[1] + '/BA Stemmen/stemmen.csv')
    data = filter_Low_Saxon_Area(data, argv[1] + '/placenames')
    data = get_municipalities(data, argv[1] + '/CBS_woonplaatsen_in_Nederland_2020.csv')
    data = fix_missing_words(data)

    # Get information about data:
    cities = location_count_dict(data, "selected_location")
    municipalities = location_count_dict(data, "municipality")
    provinces = location_count_dict(data, "province")

    # Print information about data:
    print("#### Files Processed ####")
    print("## Amount of participants: {0:>6}".format(len(data)))
    print("## Amount of cities: {0:>12}".format(len(cities)))
    print("## Amount of municipalities: {0:>4}".format(len(municipalities)))
    print_std_dict(municipalities)
    print("## Amount of provinces: {0:>9}".format(len(provinces)))
    print_std_dict(provinces)

    # Save pre-processed data:
    save_csv_file(data, "stemmen_preprocessed")
    print("### Pre-processed data saved ###")

    # Split randomly into test and train data and save:
    sessionIDs = list(data.keys())
    random.shuffle(sessionIDs)

    test_data = {x:data[x] for x in data if x in sessionIDs[:int(len(data) / 10)]}  # random first 10% of data
    save_csv_file(test_data, "stemmen_test")
    save_pickle_file(test_data, "stemmen_test")
    print("### Test data saved ###")

    train_data = {x:data[x] for x in data if x in sessionIDs[int(len(data) / 10):]}  # random last 90% of data
    save_csv_file(train_data, "stemmen_train")
    save_pickle_file(train_data, "stemmen_train")
    print("### Train data saved ###")


def param_test(argv):
    """
    Test parameter input
    :param argv:
    :return: Boolean
    """

    if len(argv) == 1:
        print("Error: 1 parameter containing the data directory path needed",
              file=sys.stderr)
        return False
    elif len(argv) > 2:
        print("Error: Can only input 1 parameter containing the data directory path",
              file=sys.stderr)
        return False
    else:
        return True


if __name__ == "__main__":
    if param_test(sys.argv):
        main(sys.argv)
