import pandas as pd
import glob
import os
import argparse
import soundfile as sf
import swifter


def find_audio(path, ext='.wav'):
    audio_files = []
    for root, dirs, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(ext):
                audio_files.append(os.path.join(root, file))
    return audio_files

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
    parser.add_argument('--output_dir', type=str, default='data_splits', help='Directory to save the output CSV files')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    audio_files = find_audio(audio_dir)
    df_audio = pd.DataFrame(audio_files, columns=['audio_path'])

    df_audio['feat_path'] = df_audio['audio_path'].apply(lambda x: x.replace('.wav', '.pt').replace('data', 'outputs'))
    df_audio['speaker'] = df_audio['audio_path'].apply(lambda x: x.split('/')[-2])
    df_audio['split'] = df_audio['audio_path'].apply(lambda x: 'test' if 'test' in x else 'train')
    
    # Use swifter for parallel processing of the duration calculation
    df_audio['duration'] = df_audio['audio_path'].swifter.apply(get_duration)
    print(f"Total audio duration: {df_audio['duration'].sum():.2f} seconds")
    print(f"Average file duration: {df_audio['duration'].mean():.2f} seconds")
    df_audio.drop(columns=['feat_path'], errors='ignore').to_csv(os.path.join(output_dir, 'stats.csv'), index=False)

    df_test = df_audio[df_audio['split'] == 'test'].copy()
    df_train = df_audio[df_audio['split'] == 'train'].copy()

    # split train set into train and valid
    df_train = df_train.sample(frac=1).reset_index(drop=True)  # shuffle the train set
    n = len(df_train)
    n_train = int(n * 0.9)
    df_valid = df_train.iloc[n_train:].reset_index(drop=True)
    df_train = df_train.iloc[:n_train].reset_index(drop=True)

    label_to_save = ['audio_path', 'feat_path']

    df_train[label_to_save].to_csv(os.path.join(output_dir, 'train.csv'), index=None)
    df_valid[label_to_save].to_csv(os.path.join(output_dir, 'valid.csv'), index=None)
    df_test[label_to_save].to_csv(os.path.join(output_dir, 'test.csv'), index=None)












