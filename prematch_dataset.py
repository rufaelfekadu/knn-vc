import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastprogress.fastprogress import master_bar, progress_bar
from torch import Tensor
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, AutoProcessor, AutoModelForSpeechSeq2Seq, SpeechT5Tokenizer, AutoModel

SPEECHT5_PRETRAINED_MODEL = "mbzuai/artst_asr_v2"

from hubconf import wavlm_large

DOWNSAMPLE_FACTOR = 320

global feature_cache
feature_cache = {}
global synthesis_cache
synthesis_cache = {}

# def hook_fn(module, input, output):
#     print(f"Module: {module.__class__.__name__}, Input: {input}, Output: {output}")


def load_speecht5_model(device):
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = SpeechT5ForSpeechToText.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir="./downloads").to(
        device
    )
    processor = AutoProcessor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir='./downloads').to(device)
    # register hook for the encoder
    # model.encoder.register_forward_hook(hook_fn)

    return model, processor

def find_audio(path, ext='.wav'):
    audio_files = []
    for root, dirs, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(ext):
                audio_files.append(os.path.join(root, file))
    return audio_files

def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    audio_files = find_audio(root_path, ext='.wav')
    df = pd.DataFrame(audio_files, columns=['path'])
    df['speaker'] = df['path'].apply(lambda x: x.split('/')[-2])
    return df


def main(args):
    device = torch.device(args.device)
    if args.model_type == "wavlm":
        SYNTH_WEIGHTINGS = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=25).float().to(device)[:, None]
        MATCH_WEIGHTINGS = F.one_hot(torch.tensor(args.matching_layer), num_classes=25).float().to(device)[:, None]
    elif args.model_type == "artst":
        SYNTH_WEIGHTINGS = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=13).float().to(device)[:, None]
        MATCH_WEIGHTINGS = F.one_hot(torch.tensor(args.matching_layer), num_classes=13).float().to(device)[:, None]

    print(f"Matching weightings: {MATCH_WEIGHTINGS.squeeze()}\nSynthesis weightings: {SYNTH_WEIGHTINGS.squeeze()}")
    ls_df = make_librispeech_df(Path(args.audio_path))
    print(f"Loading model {args.model_type}.")
    if args.model_type == "wavlm":
        model = wavlm_large(pretrained=True, progress=True, device=args.device)
        processor = None
    elif args.model_type == "artst":
        model, processor = load_speecht5_model(device=args.device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    extract(ls_df, model, processor, args.device, Path(args.audio_path), Path(args.out_path), SYNTH_WEIGHTINGS, MATCH_WEIGHTINGS)
    
    print("All done!", flush=True)


def path2pools(path: Path, model: nn.Module(), processor, match_weights: Tensor, synth_weights: Tensor, device):
    """Given a waveform `path`, compute the matching pool"""

    uttrs_from_same_spk = sorted(list(path.parent.rglob('*.wav')))
    uttrs_from_same_spk.remove(path)
    matching_pool = []
    synth_pool = []
    for pth in uttrs_from_same_spk:
        if pth in feature_cache and pth in synthesis_cache:
            matching_feats = feature_cache[pth].float() # (seq_len, dim)
            synth_feats = synthesis_cache[pth].float() # (seq_len, dim)
        else:
            feats = get_full_features(pth, model, device, processor)
            matching_feats = ( feats*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)
            synth_feats = ( feats*synth_weights[:, None] ).sum(dim=0) # (seq_len, dim)
            feature_cache[pth] = matching_feats.half().cpu()
            synthesis_cache[pth] = synth_feats.half().cpu()

        matching_pool.append(matching_feats.cpu())
        synth_pool.append(synth_feats.cpu())
    matching_pool = torch.concat(matching_pool, dim=0)
    synth_pool = torch.concat(synth_pool, dim=0)
    return matching_pool, synth_pool # (N, dim)


@torch.inference_mode()
def get_full_features(path, model, device, processor=None):

    x, sr = torchaudio.load(str(path))
    assert sr == 16000

    if model._get_name() == "WavLM":
        # This does not work i.t.o the hifigan training.
        # x = F.pad(x, (DOWNSAMPLE_FACTOR//2, DOWNSAMPLE_FACTOR - DOWNSAMPLE_FACTOR//2), value=0)
        # This does.
        n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
        x = F.pad(x, (0, n_pad), value=0)

        # extract the representation of each layer
        wav_input_16khz = x.to(device)
        rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
    else:
        try:
            inputs = processor(audio=x.squeeze(), return_tensors="pt", padding=True, sampling_rate=16000).to(device)
            with torch.no_grad():   
                layer_results = model.speecht5.encoder(**inputs, output_hidden_states=True)
                features = torch.cat(layer_results['hidden_states'], dim=0) # (n_layers, seq_len, dim)
        except:
            breakpoint()
    return features


def fast_cosine_dist(source_feats, matching_pool):
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


@torch.inference_mode()
def extract(df: pd.DataFrame, model: nn.Module, processor, device, ls_path: Path, out_path: Path, synth_weights: Tensor, match_weights: Tensor):
    
    pb = progress_bar(df.iterrows(), total=len(df))

    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (out_path/rel_path).with_suffix('.pt')
        if args.resume:
            if targ_path.is_file(): continue
        # if targ_path.is_file(): continue
        os.makedirs(targ_path.parent, exist_ok=True)

        if Path(row.path) in feature_cache:
            source_feats = feature_cache[Path(row.path)].float()
        else:
            source_feats = get_full_features(Path(row.path), model, device, processor)
            source_feats = ( source_feats*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)

        matching_pool, synth_pool = path2pools(Path(row.path), model, processor, match_weights, synth_weights, device)

        if not args.prematch:
            out_feats = source_feats.cpu()
        else:
            dists = fast_cosine_dist(source_feats.cpu(), matching_pool.cpu()).cpu()
            best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
            out_feats = synth_pool[best.indices].mean(dim=1) # (N, dim)

        # save matched sequence
        if i < 3: print("Feature has shape: ", out_feats.shape, flush=True)
        # 3. save
        torch.save(out_feats.cpu().half(), str(targ_path))
        if hasattr(pb, 'child'):
            pb.child.comment = str(rel_path)
            pb.child.wait_for = min(pb.child.wait_for, 10)
            pb.main_bar.comment = str(rel_path)
        else:
            pb.wait_for = min(pb.wait_for, 10)
        pb.comment = str(rel_path)
        

        if i % 1000 == 0: 
            print(f"Done {i:,d}/{len(df):,d}", flush=True)
            feature_cache.clear()
            synthesis_cache.clear()
            gc.collect()
            time.sleep(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute matched wavlm features for a librispeech dataset")

    parser.add_argument('--audio_path', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--out_path', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--matching_layer', type=int, default=6)
    parser.add_argument('--synthesis_layer', type=int, default=6)
    parser.add_argument('--prematch', action='store_true', help='prematch')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    main(args)

