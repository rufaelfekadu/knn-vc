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

    valid_speakers = ['female_ab','female_ad', 'male_aa', 'male_ac', 'male_asc']

    df =  pd.read_csv(args.manifest, delimiter="\t")
    source_speakers = df[df['split'] == 'source_'+args.split]['speaker_id'].unique().tolist()
    target_speakers = df[df['split'] == 'target']['speaker_id'].unique().tolist()
    # get the knnvc model
    knnvc = knn_vc(pretrained=True, progress=True, ckpt_path=args.ckpt_path, device=args.device)

    pair = defaultdict(list)

    print(f"running inference for {len(target_speakers)} target speakes")

    for target_speaker in tqdm(target_speakers, total=len(target_speakers)):
        ref_wav_paths = df[df['speaker_id'] == target_speaker]['audio_path'].values
        # ref_wav_paths = df[df['speaker'] == target_speaker]['audio_path'].sort_values(ascending=False).head(args.n_ref).values
        
        for source_speaker in source_speakers:

            output_dir = Path(args.out_dir) / f'{source_speaker}->{target_speaker}'
            os.makedirs(output_dir, exist_ok=True)

            source_utterances = df[df['speaker_id'] == source_speaker]['audio_path'].values
            print(f"Converting {source_speaker} -> {target_speaker}")
            for source_utterance in tqdm(source_utterances, desc=f"Converting {source_speaker} -> {target_speaker}", total=len(source_utterances)):

                out_wav_path = output_dir / f'{Path(source_utterance).stem}.wav'

                if os.path.isfile(out_wav_path):
                    continue

                query_seq = knnvc.get_features(source_utterance, vad_trigger_level=0.0)
                matching_set = knnvc.get_matching_set(ref_wav_paths, vad_trigger_level=0.0)
                out_wav = knnvc.match(query_seq, matching_set, topk=4)

                # Save the generated audio tensor as a .wav file
                out_wav = out_wav.squeeze().cpu().numpy()
            
                # Save the generated audio tensor as a .wav file using soundfile
                sf.write(out_wav_path, out_wav, samplerate=16000)

                pair['gen_wav_path'].append(out_wav_path)
                pair['src_wav_path'].append(source_utterance)
                pair['target_speaker'].append(target_speaker)
                pair['source_speaker'].append(source_speaker)
    
    pd.DataFrame(pair).to_csv(Path(args.out_dir) / 'match.csv', index=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='/workspace/datasets/corpora_splits.tsv', help='Path to the stats csv')
    parser.add_argument('--split', type=str, default='test', help='Split to use (train, dev, test)')
    parser.add_argument('--out_dir', type=str, default='outputs/generated/baseline', help='Path to the output directory')
    parser.add_argument('--ckpt_path', type=str, default='outputs/checkpoints/baseline/g_02500000.pt', help='Path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_ref', type=int, default=10, help='Number of reference speakers')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)