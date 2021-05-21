import pretty_midi as pm


# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt

import os


from sklearn.decomposition import PCA

# import warnings
# warnings.filterwarnings("ignore")

Normalized_Pitch = 127
data_path = 'C:/code/course/Deep_learning/Ass3/Data'
midi_dir = 'midi_files'

def plot_piano_roll(midi_file, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(midi_file.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))


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
        if os.path.exists(os.path.join(jpg_dir_path, path + ".jpg")):
            continue
        try:
            midi_file = pm.PrettyMIDI(os.path.join(midi_files_path, path))
            plt.figure(figsize=(24, 8))
            plot_piano_roll(midi_file, 0, 127)
            beats = midi_file.get_beats()
            downbeats = midi_file.get_downbeats()
            ymin, ymax = plt.ylim()
            # Plot beats as grey lines, downbeats as white lines
            mir_eval.display.events(beats, base=ymin, height=ymax, color='#AAAAAA', lw=0.1)
            mir_eval.display.events(downbeats, base=ymin, height=ymax, color='#FFFFFF', lw=0.25)
            plt.savefig(os.path.join(jpg_dir_path, path + ".jpg"))
            plt.close()
        except Exception as e:
            bad_songs.append(path)
            print(e)

midi_jpg_dir = 'midi_jpgs'
train_set_path = 'lyrics_train_set.csv'
build_jpg_dataset(os.path.join(data_path, midi_jpg_dir))
