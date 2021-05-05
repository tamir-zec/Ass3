import pretty_midi as pm
import pandas as pd
import numpy as np
import os

import gensim
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api


Normalized_Pitch = 127
data_path = 'C:/code/course/Deep_learning/Ass3/Data'

# Pitch from C-1 to G-9 values 0-127
# velocity - indicates volume level, 1-127 as well

def reshape_lists(*args):
    output_data=[]
    if len(args) > 0:
        for row_index in range(len(args[0])):
            row = []
            for num_of_lists in range(len(args)):
                row.append(args[num_of_lists][row_index])
            output_data.append(row)
    return output_data


def midi_to_csv(midi_files_path):
    bad_songs = []
    song_names = []
    key_signatures = []
    time_signatures = []
    ticks = []
    is_drum = []
    notes_lists = []

    for path in os.listdir(midi_files_path):
        try:
            midi_file = pm.PrettyMIDI(os.path.join(midi_files_path, path))
            for instrument in midi_file.instruments:
                song_names.append(path)
                key_signatures.append(midi_file.key_signature_changes)
                time_signatures.append(midi_file.time_signature_changes)
                ticks.append(midi_file._tick_scales)
                is_drum.append(instrument.is_drum)
                notes_lists.append(instrument.notes)
        except:
            bad_songs.append(path)

    data = reshape_lists(song_names, key_signatures, time_signatures, ticks, is_drum, notes_lists)
    df = pd.DataFrame(data,
                      columns=['song_name',
                               'key_signatures',
                               'time_signatures',
                               'ticks',
                               'is_drum',
                               'notes_lists'])
    df.to_csv('Data/midi_df.csv', sep='\t', index=False)
    with open('Data/badSongs.txt', 'w') as file:
        file.writelines('\n'.join(bad_songs))


class words_generator:
    def __init__(self, lyrics):
        self.lyrics = lyrics

    def __iter__(self):
        for song in self.lyrics:
            yield song.split()


#   def lyrics_to_embedding(data_lyrics_path):
    # lyrics_df = pd.read_csv(data_lyrics_path, sep=',', names=['artist', 'song_name', 'lyrics'],
    #                         usecols=[0, 1, 2], header=None)
    # # nltk.download('stopwords')
    # # stopwords_list = stopwords.words("english")
    # EMBEDDING_DIM = 300
    # gen_lyrics = words_generator(lyrics_df['lyrics'])
    # glove_key_words = api.load("glove-wiki-gigaword-300")
    # w2v_model = gensim.models.Word2Vec(sentences=gen_lyrics, vector_size=EMBEDDING_DIM, window=8, min_count=1)
    # w2v_model.wv = glove_key_words
    # w2v_model.train(corpus_iterable=gen_lyrics, total_examples=w2v_model.corpus_count, epochs=2)
    #
    # #save model
    # model_path = os.path.join(data_path, 'w2v_model')
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # w2v_model.save(os.path.join(model_path, 'w2v_songs'))

def music_to_csv(midi_files_path):
    tick_window = 10
    try:
        os.mkdir(os.path.join(data_path, "song_representations"))
    except FileExistsError:
        pass


    for song_path in os.listdir(midi_files_path):
        try:
            midi_file = pm.PrettyMIDI(os.path.join(midi_files_path, song_path))
            song_encoding = []
            time_stamp = {}
            for time in range(0, len(midi_file._PrettyMIDI__tick_to_time), tick_window):
                start_time = midi_file.tick_to_time(time)
                end_time = midi_file.tick_to_time(tick_window-1)
                time_stamp['start'] = start_time
                time_stamp['end'] = end_time
                time_stamp['key signature'] = get_key_sig(midi_file.key_signature_changes, start_time)
                time_stamp['time signature'] = get_time_sig(midi_file.time_signature_changes, start_time)
                for instrument in midi_file.instruments:


            # df = pd.DataFrame(song_encoding,
            #      columns = ['start',
            #                 'end',
            #                 'pitch',
            #                 'velocity',
            #                 ])
            # df.to_csv(os.path.join(data_path, "song_representations/", song_path), sep='\t', index=False)
        except:
            pass


def get_key_sig(key_signatures, start_time):
    for key_sig in key_signatures:
        if start_time >= key_sig.time:
            return key_sig.key_number
    return 0


def get_time_sig(time_signatures, start_time):
    for time_sig in time_signatures:
        if start_time >= time_sig.time:
            return time_sig.numerator / time_sig.denominator
    return 0


def main():
    midi_dir = 'midi_files'
    train_set_path = 'lyrics_train_set.csv'
    model_path = 'Data/w2v_model'
    # midi_to_csv(os.path.join(data_path, midi_dir))
    music_to_csv(os.path.join(data_path, midi_dir))
    # lyrics_to_embedding(os.path.join(data_path, train_set_path))


if __name__ == '__main__':
    main()