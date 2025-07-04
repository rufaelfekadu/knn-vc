#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import numpy as np
import json
import fnmatch

import torch
from tqdm import tqdm


from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.metric_stats import BinaryMetricStats

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="downloads/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
    
def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def compute_scores(ref_files, converted_files):
    
    eer_calculator = BinaryMetricStats()
    
    print(f"Number of reference files: {len(ref_files)}, converted files: {len(converted_files)}")
    pairs_gen = []
    for i, cv_path in enumerate(converted_files):
        basename = get_basename(cv_path)
        
        # negative pairs gen gt
        for k, ref_file in enumerate(ref_files):
            if basename == get_basename(ref_file): 
                continue
            pairs_gen.append((cv_path, ref_file, 0))
    # randomly sample 1500 pairs
    idxs = np.random.choice(len(pairs_gen), size=min(1500, len(pairs_gen)), replace=False)
    pairs_gen = [pairs_gen[i] for i in idxs]
    scores_gen = []
    for i, (cv_path, ref_path, label) in enumerate(tqdm(pairs_gen, desc="Computing EER")):
        p_score, p_pred = verification.verify_files(ref_path, cv_path)
        
        scores_gen.append([f"{get_basename(cv_path)}_{get_basename(ref_path)}", p_score.cpu().item(), p_pred.cpu().item(), label])
        
        eer_calculator.append(
            [f"{get_basename(cv_path)}_{get_basename(ref_path)}"],
            p_score.cpu(),
            torch.tensor([label])
        )
    eer_gen = eer_calculator.summarize(threshold=0.5)
    

    return eer_gen, scores_gen

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--gen_root", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--tgt_root", type=str, default="./data/test-norm/", help="directory of data")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="directory to save results")
    return parser

def main():

    args = get_parser().parse_args()
    src_tgt_pair = os.listdir(args.gen_root)
    os.makedirs(args.output_dir, exist_ok=True)
    # src_tgt_pair = ['female_ad-ar-XA-Wavenet-A', 
    #                 'female_ab-ar-XA-Wavenet-A',
    #                 'female_af-ar-XA-Wavenet-A', 
    #                 'female_ag-ar-XA-Wavenet-A',
    #                 'male_aa-ar-XA-Wavenet-B',
    #                 'male_ac-ar-XA-Wavenet-B',
    #                 'male_ae-ar-XA-Wavenet-B',]
    for pair in src_tgt_pair:
        if not os.path.isdir(os.path.join(args.gen_root, pair)):
            continue
        # if "noise" not in pair:
        #     continue
        # split after the first hyphen only
        srcspk , trgspk = pair.split("-", 1)
        # trgspk = trgspk.replace("-noise", "")
        converted_files = sorted(find_files(os.path.join(args.gen_root, pair), query=f"test_{srcspk}*.wav"))
        ref_files = sorted(find_files(os.path.join(args.tgt_root, f"{trgspk}"), query=f"*test_{srcspk}*.wav"))
        # converted_files = sorted(find_files(os.path.join(args.gen_root, f"ArVoice_syn-{srcspk}", f"{trgspk}"), query="test_*.wav"))
        # ref_files = sorted(find_files(os.path.join(args.tgt_root, f"ArVoice_syn-{srcspk}", f"{trgspk}"), query=f"test_*.wav"))
        print(f"srcspk: {srcspk}, trgspk: {trgspk}")
        print("number of utterances = {}".format(len(converted_files)))
        print("number of reference files = {}".format(len(ref_files)))

        if os.path.exists(os.path.join(args.output_dir, f"eer_{pair}.txt")):
            print(f"Scores for {srcspk}-{trgspk} already exist. skipping...")
            # scores = np.load(os.path.join(args.output_dir, f"scores_{srcspk}-{trgspk}.npy"), allow_pickle=True)
            # eer_gen = json.load(open(os.path.join(args.output_dir, f"eer_{srcspk}-{trgspk}.txt"), "r"))
        else:
            eer_gen, scores = compute_scores(ref_files=ref_files, converted_files=converted_files)
            np.save(os.path.join(args.output_dir, f"scores_{pair}.npy"), np.array(scores))

            with open(os.path.join(args.output_dir, f"eer_{pair}.txt"), "w") as f:
                json.dump(eer_gen, f, indent=4)

    
if __name__ == "__main__":
    main()