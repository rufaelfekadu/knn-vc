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

# PAIRS = [('female_ab', 'female_ad'), ('male_aa', 'male_ac'),('male_ac','female_ab'),('female_ad', 'male_aa'), ('male_asc','female_ab')]

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

    valid_speakers = ['female_ab','female_ad', 'male_aa', 'male_ac', 'male_asc', 'ar-XA-Wavenet-A', 'ar-XA-Wavenet-B', 'female_ag']
    # get the knnvc model
    knnvc = knn_vc(pretrained=True, progress=True, ckpt_path=args.ckpt_path, device=args.device)
    df = pd.read_csv(args.stats_csv)
    # df = df[df['split'] == 'test']
    df= df[df['speaker'].isin(valid_speakers)]

    with open(args.pairs, 'r') as f:
        PAIRS = [tuple(line.strip().split(' ')) for line in f.readlines()]

    pair = defaultdict(list)
    # total_speakers = df['speaker'].unique()
    # spk_pairs = [(s1, s2) for i, s1 in enumerate(total_speakers) for s2 in total_speakers[i+1:] if s1 != s2]

    for source , target in PAIRS:

        output_dir = Path(args.out_dir) / f'{source}-{target}'
        os.makedirs(output_dir, exist_ok=True)
        if len(os.listdir(output_dir)) > 0:
            print(f"Skipping {source}-{target} as output directory is not empty.")
            continue
        
        
        # get the source test wav paths
        src_wav_paths = df[df['speaker'] == source]['audio_path'].values
        ref_wav_paths = df[df['speaker'] == target].sort_values('duration', ascending=False).head(args.n_ref)['audio_path'].tolist()
        print(df)
        
        for src_wav_path in tqdm(src_wav_paths):
            query_seq = knnvc.get_features(src_wav_path)
            matching_set = knnvc.get_matching_set(ref_wav_paths)
            out_wav = knnvc.match(query_seq, matching_set, topk=4)

            # Save the generated audio tensor as a .wav file
            out_wav = out_wav.squeeze().cpu().numpy()
            out_wav_path = output_dir / f'{Path(src_wav_path).stem}.wav'
            
            # Save the generated audio tensor as a .wav file using soundfile
            sf.write(out_wav_path, out_wav, samplerate=16000)

            pair['gen_wav_path'].append(out_wav_path)
            pair['src_wav_path'].append(src_wav_path)

    
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
    args = parser.parse_args()
    main(args)