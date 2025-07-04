from matcher import KNeighborsVC
from hubconf import wavlm_large
from hifigan.utils import AttrDict
import json
import torch
from pathlib import Path
from hifigan.models import Generator as HiFiGAN
import os
import pandas as pd
from collections import defaultdict
import soundfile as sf
from tqdm import tqdm
import torchaudio
import numpy as np
import random
import glob

# PAIRS = [('female_ab', 'female_ad'), ('male_aa', 'male_ac'),('male_ac','female_ab'),('female_ad', 'male_aa'), ('male_asc','female_ab')]

def load_noise_files(noise_dir):
    """Load all noise files from a directory"""
    noise_files = []
    if noise_dir and os.path.exists(noise_dir):
        # Support common audio formats
        extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a']
        for ext in extensions:
            # noise_files.extend(glob.glob(os.path.join(noise_dir, ext)))
            noise_files.extend(glob.glob(os.path.join(noise_dir, '**', ext), recursive=True))
    return noise_files

def load_noise_file(noise_path, target_sr=16000):
    """Load a single noise file and resample if necessary"""
    try:
        noise_audio, noise_sr = torchaudio.load(noise_path, normalize=True)
        
        # Resample if necessary
        if noise_sr != target_sr:
            resampler = torchaudio.transforms.Resample(noise_sr, target_sr)
            noise_audio = resampler(noise_audio)
        
        return noise_audio.squeeze().numpy()
    except Exception as e:
        print(f"Error loading noise file {noise_path}: {e}")
        return None

def add_noise(wav, snr=0.01):
    """ Add Gaussian noise to the waveform """
    noise = torch.randn_like(wav) * snr
    return wav + noise

def add_noise_at_snr(clean_signal, noise_signal, snr_db=10):
    """
    Add noise to clean_signal at a specified SNR in dB.
    """
    # Ensure same length
    if len(noise_signal) < len(clean_signal):
        repeats = int(np.ceil(len(clean_signal) / len(noise_signal)))
        noise_signal = np.tile(noise_signal, repeats)
    
    noise_signal = noise_signal[:len(clean_signal)]

    # Compute power of signals
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise_signal ** 2)

    # Compute scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_linear
    scaling_factor = np.sqrt(desired_noise_power / noise_power)

    # Scale noise and add to signal
    noisy_signal = clean_signal + noise_signal * scaling_factor

    return noisy_signal

def add_varying_noise(aud, noise_files=None, snr_levels=None, use_gaussian=True):
    """
    Add varying types of noise to the audio
    
    Args:
        aud: Input audio tensor
        noise_files: List of paths to noise files
        snr_levels: List of SNR levels in dB
        use_gaussian: Whether to also add Gaussian noise
    
    Returns:
        List of tuples (noisy_audio, noise_type, snr_level)
    """
    results = []
    clean_audio = aud.squeeze().cpu().numpy()
    
    # Default SNR levels if not provided
    if snr_levels is None:
        snr_levels = [20, 15, 10, 5, 0]  # dB
    
    # Add Gaussian noise at different levels
    if use_gaussian:
        for snr_db in snr_levels:
            # Convert SNR from dB to linear scale for Gaussian noise
            snr_linear = 10 ** (snr_db / 10)
            noise_level = 1.0 / np.sqrt(snr_linear)
            
            noisy_audio = add_noise(aud, snr=noise_level)
            results.append((noisy_audio, 'gaussian', snr_db))
    
    # Add noise from files at different levels
    if noise_files:
        for noise_file in noise_files:
            noise_signal = load_noise_file(noise_file)
            if noise_signal is not None:
                for snr_db in snr_levels:
                    noisy_audio = add_noise_at_snr(clean_audio, noise_signal, snr_db)
                    noise_type = Path(noise_file).stem
                    results.append((torch.tensor(noisy_audio).unsqueeze(0), noise_type, snr_db))
    
    return results

def hifigan_wavlm(pretrained=True, ckpt_dir=None, device='cuda'):
    # load the generator from chekpoint
    cp = Path(__file__).parent.absolute()

    with open(cp/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)
    
    if pretrained and ckpt_dir is not None:
        # load the pretrained wegihts from file
        if ckpt_dir.endswith('.pt'):
            state_dict_g = torch.load(ckpt_dir)
        else:
            # read all the files in the dir and get the latest checkpoint
            ckpt_files = os.listdir(ckpt_dir)
            ckpt_files = [f for f in ckpt_files if f.endswith('.pt') and f.startswith('g')]
            ckpt_files.sort()
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
            state_dict_g = torch.load(ckpt_path)

        generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h

def knn_vc(pretrained=True, progress=True, ckpt_path=None, device='cuda') -> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, ckpt_path, device)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc

def main(args):

    valid_speakers = ['female_ab','female_ad', 'male_aa', 'male_ac', 'ar-XA-Wavenet-A', 'ar-XA-Wavenet-B']
    # valid_speakers = ['male_ae', 'ar-XA-Wavenet-A', 'ar-XA-Wavenet-B', 'female_af', 'female_ag', 'male_ae']
    # get the knnvc model
    knnvc = knn_vc(pretrained=True, progress=True, ckpt_path=args.ckpt_path, device=args.device)
    df = pd.read_csv(args.stats_csv)
    df = df[df['split'] == 'test']
    df= df[df['speaker'].isin(valid_speakers)]

    with open(args.pairs, 'r') as f:
        PAIRS = [tuple(line.strip().split(' ')) for line in f.readlines()]

    # Load noise files if provided
    noise_files = []
    if args.noise_dir:
        noise_files = load_noise_files(args.noise_dir)
        print(f"Loaded {len(noise_files)} noise files from {args.noise_dir}")

    # Parse SNR levels
    snr_levels = [int(x) for x in args.snr_levels.split(',')] if args.snr_levels else [20, 15, 10, 5, 0]

    pair = defaultdict(list)
    # total_speakers = df['speaker'].unique()
    # spk_pairs = [(s1, s2) for i, s1 in enumerate(total_speakers) for s2 in total_speakers[i+1:] if s1 != s2]

    for source , target in PAIRS:
        
        tgt_folder = f'{source}-{target}'
        if args.add_noise:
            tgt_folder += '-noise'
        output_dir = Path(args.out_dir) / tgt_folder

        os.makedirs(output_dir, exist_ok=True)
        if len(os.listdir(output_dir)) > 0 and not args.force_overwrite:
            print(f"Skipping {source}-{target} as output directory is not empty.")
            continue
        
        # get the source test wav paths
        src_wav_paths = df[df['speaker'] == source]['audio_path'].values
        ref_wav_paths = df[df['speaker'] == target].sort_values('duration', ascending=False).head(args.n_ref)['audio_path'].tolist()

        for src_wav_path in tqdm(src_wav_paths, desc=f"Processing {source}-{target}"):
            # Load the source audio file
            aud, sr = torchaudio.load(src_wav_path, normalize=True)

            if args.add_noise:
                # Generate multiple noisy versions
                noisy_versions = add_varying_noise(
                    aud, 
                    noise_files=noise_files, 
                    snr_levels=snr_levels, 
                    use_gaussian=args.use_gaussian
                )
                
                # Save noisy versions
                for noisy_aud, noise_type, snr_db in noisy_versions:
                    # Create subdirectory structure
                    noise_subdir = output_dir / 'noise' / f'{noise_type}_snr{snr_db}db'
                    os.makedirs(noise_subdir, exist_ok=True)
                    
                    noisy_wav_path = noise_subdir / f'{Path(src_wav_path).stem}.wav'
                    sf.write(noisy_wav_path, noisy_aud.squeeze().cpu().numpy(), samplerate=sr)
                    
                    # Process with kNN-VC
                    query_seq = knnvc.get_features(noisy_aud)
                    matching_set = knnvc.get_matching_set(ref_wav_paths)
                    out_wav = knnvc.match(query_seq, matching_set, topk=4)

                    # Save the generated audio
                    out_wav = out_wav.squeeze().cpu().numpy()
                    gen_subdir = output_dir / 'gen' / f'{noise_type}_snr{snr_db}db'
                    os.makedirs(gen_subdir, exist_ok=True)
                    out_wav_path = gen_subdir / f'{Path(src_wav_path).stem}.wav'
                    sf.write(out_wav_path, out_wav, samplerate=16000)

                    # Record metadata
                    pair['gen_wav_path'].append(str(out_wav_path))
                    pair['src_wav_path'].append(src_wav_path)
                    pair['noise_type'].append(noise_type)
                    pair['snr_db'].append(snr_db)
                    pair['source_speaker'].append(source)
                    pair['target_speaker'].append(target)
            else:
                # Process without noise
                query_seq = knnvc.get_features(aud)
                matching_set = knnvc.get_matching_set(ref_wav_paths)
                out_wav = knnvc.match(query_seq, matching_set, topk=4)

                # Save the generated audio tensor as a .wav file
                out_wav = out_wav.squeeze().cpu().numpy()
                out_wav_path = output_dir / f'{Path(src_wav_path).stem}.wav'
                
                os.makedirs(out_wav_path.parent, exist_ok=True)
                # Save the generated audio tensor as a .wav file using soundfile
                sf.write(out_wav_path, out_wav, samplerate=16000)

                pair['gen_wav_path'].append(str(out_wav_path))
                pair['src_wav_path'].append(src_wav_path)
                pair['noise_type'].append('clean')
                pair['snr_db'].append('N/A')
                pair['source_speaker'].append(source)
                pair['target_speaker'].append(target)

    
    pd.DataFrame(pair).to_csv(Path(args.out_dir) / 'match.csv', index=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_csv', type=str, default='data_splits/stats.csv', help='Path to the stats csv')
    parser.add_argument('--pairs', type=str, default='data_splits/pairs.txt', help='Path to the pairs txt file')
    parser.add_argument('--out_dir', type=str, default='outputs/cloned_audio_pair', help='Path to the output directory')
    parser.add_argument('--ckpt_path', type=str, default='outputs/checkpoints', help='Path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_ref', type=int, default=10, help='Number of reference speakers')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the generated audio')
    parser.add_argument('--noise_dir', type=str, default=None, help='Directory containing noise files')
    parser.add_argument('--snr_levels', type=str, default='20,15,10,5', help='Comma-separated SNR levels in dB')
    parser.add_argument('--use_gaussian', action='store_true', help='Use Gaussian noise in addition to noise files')
    parser.add_argument('--force_overwrite', action='store_true', help='Force overwrite existing output directories')
    args = parser.parse_args()
    main(args)