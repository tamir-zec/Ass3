import pretty_midi as pm
import pandas as pd
import numpy as np
import os
from functools import reduce
import operator

from sklearn.decomposition import PCA

import gensim
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api

Normalized_Pitch = 127
data_path = 'C:/code/course/Deep_learning/Ass3/Data'
midi_dir = 'midi_files'


def build_jpg_dataset(jpg_dir_path):
    try:
        os.mkdir(jpg_dir_path)
    except FileExistsError:
        pass

    pictures_data = []
    bad_songs = []
    song_names = []
    midi_files_path = os.path.join(data_path, midi_dir)

    for path in os.listdir(midi_files_path):
        try:
            midi_file = pm.PrettyMIDI(os.path.join(midi_files_path, path))
            #toDo: implement some way to take pictures

        except:
            bad_songs.append(path)


    df = pd.DataFrame(pictures_data,
                      columns=['song_name',
                               'unknown'])
    df.to_csv('Data/midi__jpg_df.csv', sep='\t', index=False)
    with open('Data/badSongs.txt', 'w') as file:
        file.writelines('\n'.join(bad_songs))


def build_autoencoder():
    pass


def main():
    midi_jpg_dir = 'midi_jpgs'
    train_set_path = 'lyrics_train_set.csv'
    build_jpg_dataset(os.path.join(data_path, midi_jpg_dir))
    model = build_autoencoder()

    # lyrics_to_embedding(os.path.join(data_path, train_set_path))


if __name__ == '__main__':
    main()
