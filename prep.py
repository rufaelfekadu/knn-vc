import pandas as pd
import glob

audio_dir = 'data/test'

audio_files = glob.glob(f'{audio_dir}/**/*.wav')
df_audio = pd.DataFrame(audio_files, columns=['audio_path'])
df_audio['feat_path'] = df_audio['audio_path'].apply(lambda x: x.replace('.wav', '.pt').replace('data', 'outputs'))

# save to csv
df_audio.to_csv('data_splits/test_arvoice.csv', index=None)


audio_dir = 'data/train'
audio_files = glob.glob(f'{audio_dir}/**/*.wav')
df_audio = pd.DataFrame(audio_files, columns=['audio_path'])
df_audio['feat_path'] = df_audio['audio_path'].apply(lambda x: x.replace('.wav', '.pt').replace('data', 'outputs'))
df_audio['speaker'] = df_audio['audio_path'].apply(lambda x: x.split('/')[-2])

# split the df to train and valid with balanced speakers
df_train = pd.DataFrame()
df_valid = pd.DataFrame()

for spk in df_audio['speaker'].unique():
    df_spk = df_audio[df_audio['speaker'] == spk]
    n = len(df_spk)
    n_train = int(n * 0.9)
    df_train = df_train._append(df_spk.iloc[:n_train])
    df_valid = df_valid._append(df_spk.iloc[n_train:])

df_train.to_csv('data_splits/train_arvoice.csv', index=None)
df_valid.to_csv('data_splits/valid_arvoice.csv', index=None)













