import pandas as pd
import glob
import os
import argparse
import soundfile as sf
import swifter
import fnmatch


def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def get_duration(audio_path):
    try:
        data, samplerate = sf.read(audio_path)
        duration = len(data) / samplerate
        return duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare audio data for training and validation.')
    parser.add_argument('--audio_dir', type=str, default='data/ArVoice', help='Directory containing audio files')
    parser.add_argument('--prematched_dir', type=str, default='data/ArVoice-16k/prematched', help='Directory containing prematched features')
    parser.add_argument('--output_dir', type=str, default='data_splits', help='Directory to save the output CSV files')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    prematched_dir = args.prematched_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(find_files(audio_dir, query="*.wav"))
    prematched_files = sorted(find_files(prematched_dir, query="*.pt"))

    assert len(audio_files) == len(prematched_files), "Number of audio files and prematched files must be the same"

    # create a dataframe of audio_path, prematched_path columns
    df = pd.DataFrame({'audio_path': audio_files, 'feat_path': prematched_files})
    # make the audio and feature paths relative to the audio_dir and prematched_dir
    df['audio_path'] = df['audio_path'].apply(lambda x: x.replace(audio_dir + '/', ''))
    df['feat_path'] = df['feat_path'].apply(lambda x: x.replace(prematched_dir + '/', ''))

    df['speaker'] = df['audio_path'].apply(lambda x: x.split('/')[-2])
    df['split'] = df['audio_path'].apply(lambda x: 'test' if 'test' in x else 'train')

    valid_speakers = ['ar-XA-Wavenet-D', 'ar-XA-Wavenet-B']
    df = df[df['speaker'].isin(valid_speakers)]
    # df_test = df[df['split'] == 'test'].copy()
    df_train = df[df['split'] == 'train'].copy()

    # split train set into train and valid
    df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle the train set
    n = len(df_train)
    n_train = int(n * 0.9)
    df_valid = df_train.iloc[n_train:].reset_index(drop=True)
    df_train = df_train.iloc[:n_train].reset_index(drop=True)

    label_to_save = ['audio_path', 'feat_path']

    df_train[label_to_save].to_csv(os.path.join(output_dir, 'train.csv'), index=None)
    df_valid[label_to_save].to_csv(os.path.join(output_dir, 'valid.csv'), index=None)
    # df_test[label_to_save].to_csv(os.path.join(output_dir, 'test.csv'), index=None)












