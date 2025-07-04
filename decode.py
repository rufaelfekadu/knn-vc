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
import fnmatch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, AutoProcessor, AutoModelForSpeechSeq2Seq
SPEECHT5_PRETRAINED_MODEL = "mbzuai/artst_asr_v2"

# PAIRS = [('female_ab', 'female_ad'), ('male_aa', 'male_ac'),('male_ac','female_ab'),('female_ad', 'male_aa'), ('male_asc','female_ab')]

def load_speecht5_model(device):
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = SpeechT5ForSpeechToText.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir="./downloads").to(
        device
    )
    processor = AutoProcessor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir='./downloads').to(device)
    return model, processor

class SemanticExtractor():
    def __init__(self, model_type='wavlm', device='cuda'):
        self.model_type = model_type
        self.device = device
        if model_type == 'wavlm':
            self.model = wavlm_large(pretrained=True, progress=True, device=device)
        elif model_type == 'artst':
            self.model, self.processor = load_speecht5_model(device)
        else:
            raise ValueError(f"Model type {model_type} not supported")
    def extract_features(self, wav_input_16khz, output_layer, ret_layer_results=False, **kwargs):
        if self.model_type == 'wavlm':
            return self.model.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=ret_layer_results)
        elif self.model_type == 'artst':
            try:
                inputs = self.processor(audio=wav_input_16khz.squeeze().cpu(), return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
                with torch.no_grad():   
                    layer_results = self.model.speecht5.encoder(**inputs, output_hidden_states=True)
                    features = torch.cat(layer_results['hidden_states'], dim=0) # (n_layers, seq_len, dim)
                    return features[output_layer,:,:].unsqueeze(0)
            except:
                print(f"Error extracting features. ")
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
    def eval(self):
        self.model.eval()
        return self
    
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

def hifigan_wavlm(pretrained=True, ckpt_dir=None, device='cuda'):
    # load the generator from chekpoint
    cp = Path(__file__).parent.absolute()
    if ckpt_dir is not None:
        with open(os.path.join(ckpt_dir, 'config.json')) as f:
            data = f.read()
    else:
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
    wavlm =  SemanticExtractor(model_type='artst', device=device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc

def main(args):

    knnvc = knn_vc(pretrained=True, progress=True, ckpt_path=args.ckpt_path, device=args.device)

    valid_speakers = ['ar-XA-Wavenet-D', 'ar-XA-Wavenet-B']
    df = pd.read_csv(args.stats_csv)
    df = df[df['split'] == 'test']
    df = df[df['speaker'].isin(valid_speakers)]
    df['audio_path'] = df.apply(lambda x: os.path.join(args.ref_root, x['speaker'], x['filename']), axis=1)
    
    with open(args.pairs, 'r') as f:
        PAIRS = [tuple(line.strip().split('|')) for line in f.readlines()]

    pair = defaultdict(list)
    for (src , tgt) in PAIRS:

        if src in ['male_ae', 'female_af', 'female_ag']:
            continue

        tgt_folder = f'{src}-{tgt}'
        output_dir = Path(args.out_dir) / tgt_folder
        src_path = Path(args.src_root) / src
        ref_path = Path(args.ref_root) / tgt
        os.makedirs(output_dir, exist_ok=True)
        if len(os.listdir(output_dir)) > 0:
            print(f"Skipping {src}-{tgt} as output directory is not empty.")
            continue
        
        # get the src test wav paths
        src_wav_paths = sorted(find_files(src_path, '*.wav'))
        ref_wav_paths = sorted(find_files(ref_path, '*.wav'), key=lambda x: get_duration(x), reverse=True)[:args.n_ref]
        
        for src_wav_path in tqdm(src_wav_paths, desc=f"Processing {src}-{tgt}"):
            # Load the src audio file
            
            aud, sr = torchaudio.load(src_wav_path, normalize=True)
            if sr != 16000:
                aud = torchaudio.transforms.Resample(sr, 16000)(aud)

            query_seq = knnvc.get_features(aud)
            matching_set = knnvc.get_matching_set(ref_wav_paths)
            out_wav = knnvc.match(query_seq, matching_set, topk=4)

            # Save the generated audio tensor as a .wav file
            out_wav = out_wav.squeeze().cpu().numpy()
            rel_out_path = Path(src_wav_path).relative_to(src_path)
            out_wav_path = output_dir / rel_out_path.with_suffix('.wav')

            os.makedirs(out_wav_path.parent, exist_ok=True)
            sf.write(out_wav_path, out_wav, samplerate=16000)

            pair['gen_wav_path'].append(out_wav_path)
            pair['src_wav_path'].append(src_wav_path)

    pd.DataFrame(pair).to_csv(Path(args.out_dir) / 'match.csv', index=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='data/ArVoice-16k/test', help='Path to the data directory')
    parser.add_argument('--ref_root', type=str, default='data/ArVoice-16k/test', help='Path to the data directory')
    parser.add_argument('--stats_csv', type=str, default='data_splits/stats.csv', help='Path to the stats csv')
    parser.add_argument('--pairs', type=str, default='data_splits/pairs.txt', help='Path to the pairs txt file')
    parser.add_argument('--out_dir', type=str, default='outputs/cloned_audio_pair', help='Path to the output directory')
    parser.add_argument('--ckpt_path', type=str, default='outputs/checkpoints', help='Path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_ref', type=int, default=10, help='Number of reference speakers')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the generated audio')
    args = parser.parse_args()
    main(args)