import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa
import fnmatch


import torch
import torchaudio
from tqdm import tqdm
import yaml


from transformers import WhisperProcessor, WhisperForConditionalGeneration, SpeechT5ForSpeechToText, SpeechT5Processor, SpeechT5Tokenizer
import jiwer
import re
import pyarabic.araby as araby
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline



ASR_PRETRAINED_MODEL = "clu-ling/whisper-large-v2-arabic-5k-steps"
SPEECHT5_PRETRAINED_MODEL = "mbzuai/artst_asr_v2"


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def load_speecht5_model(device):
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = SpeechT5ForSpeechToText.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir="./downloads").to(
        device
    )
    processor = AutoProcessor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(SPEECHT5_PRETRAINED_MODEL, cache_dir='./downloads').to(device)
    tokenizer = SpeechT5Tokenizer.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    models = {"model": model, "processor": processor, "tokenizer": tokenizer}

    return models

def load_asr_model(device):
    """Load model"""
    print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
    processor = WhisperProcessor.from_pretrained(ASR_PRETRAINED_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_PRETRAINED_MODEL, cache_dir="./downloads").to(
        device
    )
    models = {"model": model, "processor": processor}
    return models

def clean_text(text):
  """Normalizes TRANSCRIPT"""
  text = re.sub(r'[\,\?\.\!\-\;\:\"\“\%\٪\‘\”\�\«\»\،\.\:\؟\؛\*\>\<]', '', text) + " " # special characters
  text = re.sub(r'http\S+', '', text) + " " # links
  text = re.sub(r'[\[\]\(\)\-\/\{\}]', '', text) + " " # brackets
  text = re.sub(r'\s+', ' ', text) + " " # extra white space
  text = araby.strip_diacritics(text) # remove diacrirics
  return text.strip()

def normalize_sentence(sentence):
    """Normalize sentence"""
    sentence = clean_text(sentence)
    return sentence

def transcribe(model, device, wav):
    """Calculate score on one single waveform"""
    # preparation

    inputs = model["processor"](
        audio=wav, sampling_rate=16000, return_tensors="pt"
    )
    inputs = inputs.to(device)
    # forced_decoder_ids = model['processor'].get_decoder_prompt_ids(language="ar", task="transcribe")
    # forward
    predicted_ids = model["model"].generate(**inputs, num_beams=10, early_stopping=True)
    transcription = model["processor"].batch_decode(
        predicted_ids, skip_special_tokens=True
    )
    return transcription

def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription

def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename]  # get rid of the first character "E"

        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)
        # trascribe
        transcription = transcribe(model, device, wav)
        transcription = "".join(str(i) for i in transcription)
        # transcription = transcription.replace(" ", "")

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(
            groundtruth, transcription
        )

        ers[basename] = [
            c_result["cer"] * 100.0,
            w_result["wer"] * 100.0,
            norm_transcription,
            norm_groundtruth,
        ]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]

    # calculate over whole set
    def er(r):
        return (
            float(r["substitutions"] + r["deletions"] + r["insertions"])
            / float(r["substitutions"] + r["deletions"] + r["hits"])
            * 100.0
        )

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--gen_root", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--tgt_root", type=str, default="./data/test-norm/", help="directory of data")
    parser.add_argument("--txt_path", type=str, default="test.txt", help="directory of text files")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="directory to save results")
    return parser

def main():

    args = get_parser().parse_args()
    src_tgt_pair = os.listdir(args.gen_root)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.txt_path, "r") as f:
        lines = f.readlines()
        transcripts = {line.split("|")[0]: line.split("|")[1].strip() for line in lines if len(line.split("|")) == 2}

    asr_model = load_speecht5_model(device="cuda" if torch.cuda.is_available() else "cpu")

    for pair in src_tgt_pair:

        if not os.path.isdir(os.path.join(args.gen_root, pair)):
            continue
        if "noise" in pair:
            continue

        output_path = os.path.join(args.output_dir, f"asr_{pair}.txt")
        if os.path.exists(output_path):
            print(f"Skipping {pair} as output file already exists.")
            continue

        # split after the first hyphen only
        srcspk , trgspk = pair.split("-", 1)
        converted_files = sorted(find_files(os.path.join(args.gen_root, pair), query="*.wav"))[:50]
        # ref_files = sorted(find_files(os.path.join(args.tgt_root,  f"{trgspk}"), query=f"test_*.wav"))
        print(f"srcspk: {srcspk}, trgspk: {trgspk}")
        print("number of utterances = {}".format(len(converted_files)))
        # print("number of reference files = {}".format(len(ref_files)))

        er, cer, wer = _calculate_asr_score(
            model=asr_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            file_list=converted_files,
            groundtruths=transcripts
        )

        with open(output_path, "w") as f:
            f.write(f"cer: {cer:.2f}\n")
            f.write(f"wer: {wer:.2f}\n")
            f.write("basename | cer | wer | transcription | groundtruth\n")
            for k, v in er.items():
                f.write(f"{k}|{v[0]:.2f}|{v[1]:.2f}|{v[2]}|{v[3]}\n")


if __name__ == "__main__":
    main()