import logging
import os
import pathlib
import time
import tempfile
import platform
import sys
import io
import langid
import nltk
import torch
import torchaudio
import soundfile as sf
import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *

import whisper
from vocos import Vocos
import multiprocessing
import wavio

print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
print(f"You are using Python version {platform.python_version()}")
if (sys.version_info[0] < 3 or sys.version_info[1] < 7):
    print("The Python version is too low and may cause problems")

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

langid.set_languages(['en', 'zh', 'ja'])

nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

thread_count = multiprocessing.cpu_count()

print("Use", thread_count, "cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
print("Model is running on ", device, ".")

model = VALLE(
    N_DIM,
    NUM_HEAD,
    NUM_LAYERS,
    norm_first=True,
    add_prenet=False,
    prefix_mode=PREFIX_MODE,
    share_embedding=True,
    nar_scale_factor=1.0,
    prepend_bos=True,
    num_quantizers=NUM_QUANTIZERS,
)
checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.to(device)
model.eval()
print("VALL-E-X Model loaded")

# Encodec model
audio_tokenizer = AudioTokenizer(device)
print("Audio Tokenizer Loaded")

# Vocos decoder
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)
print("Vocos decoder Loaded")

# ASR
if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))
print("Whisper Model Loaded")


# ################################################# Helper Methods Component
# ##############################################################
def numpy_to_wav_stream(data, rate):
    """
    Convert numpy array to WAV format as a byte stream.
    :param data: input audio in the format of Numpy array
    :param rate: sample rate of the output audio
    :return: audio in .wav format
    """
    buffer = io.BytesIO()
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)
    wavio.write(buffer, data, rate)
    buffer.seek(0)  # Reset buffer position

    return buffer


def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True,
                                      sample_len=150)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr


def clear_prompts():
    try:
        path = tempfile.gettempdir()
        for eachfile in os.listdir(path):
            filename = os.path.join(path, eachfile)
            if os.path.isfile(filename) and filename.endswith(".npz"):
                lastmodifytime = os.stat(filename).st_mtime
                endfiletime = time.time() - 60
                if endfiletime > lastmodifytime:
                    os.remove(filename)
    except:
        return


def make_prompt(name, wav, sr, save=True):
    global whisper_model
    whisper_model.to(device)
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    torchaudio.save(f"./prompts/{name}.wav", wav, sr)
    lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    return text, lang


def make_npz_prompt(name, uploaded_audio, sr, transcript_content):

    global model, text_collater, text_tokenizer, audio_tokenizer
    clear_prompts()
    wav_pr = uploaded_audio
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav_pr.size(0) == 1

    if transcript_content == "":
        text_pr, lang_pr = make_prompt(name, wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"
    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    np.savez(os.path.join(tempfile.gettempdir(), f"{name}.npz"),
             audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    return message, os.path.join(tempfile.gettempdir(), f"{name}.npz")

@torch.no_grad()
def infer_from_audio(audio_prompt, text, language, accent, text_prompt):
    """
    Generate speaking audio with inference from audio prompt .
    :param audio_prompt: wav audio file 
    :param text: Content in the output audio in 
    :param language: Language of the text param
    :param accent: accent in the output audio e.g. English, Chinese, Japaneses
    :param text_prompt: Content in the audio_prompt
    :return: audio file in .wav format
    """
    global model, text_collater, text_tokenizer, audio_tokenizer

    # Collect the audio bytes
    byte_stream = io.BytesIO(audio_prompt)

    # read the byte file with soundfile
    wav_pr, sr = sf.read(byte_stream)

    wav_pr = wav_pr.squeeze()

    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    # print(wav_pr.ndim, wav_pr.size(0))
    assert wav_pr.ndim and wav_pr.size(0) == 1

    lang_pr = langid.classify(str(text_prompt))[0]
    lang_token = lang2token[lang_pr]
    text_pr = f"{lang_token}{str(text_prompt)}{lang_token}"

    if language == 'auto-detect':
        lang_token = lang2token[langid.classify(text)[0]]
    else:
        lang_token = langdropdown2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # onload model
    model.to(device)

    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

    # tokenize text
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )

    enroll_x_lens = None
    if text_pr:
        text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
        text_prompts, enroll_x_lens = text_collater(
            [
                text_prompts
            ]
        )

    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        best_of=1,  #  best_of is confict with batch_size by using the same dimension, to do: solve it
    )

    # Decode with Vocos
    frames = encoded_frames.permute(2, 0, 1)
    features = vocos.codes_to_features(frames)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    # offload model
    model.to('cpu')
    torch.cuda.empty_cache()

    print(f"text prompt: {text_pr}\nsythesized text: {text}")
    sampled_audio = samples.squeeze(0).cpu().numpy()

    # Convert numpy arrary to wav format
    wav_stream = numpy_to_wav_stream(sampled_audio, 24000)

    return StreamingResponse(wav_stream, media_type="audio/wav",
                             headers={"Content-Disposition": "attachment; filename=output_audio.wav"})





# ################################################# VALLE-X Component
# ##############################################################

# Directory to scan
data_directory = "/data"

# Files to look for
required_files = {"source_text", "source_audio", "target_text"}

source_text = None
source_audio = None
target_text = None

while True:
    # List all files in the directory
    files_in_directory = set(os.listdir(data_directory))

    # Check if all required files are in the directory
    if required_files.issubset(files_in_directory):
        print("All files found!")

        # Read the contents of each file
        for file in required_files:
            file_path = os.path.join(data_directory, file)
            with open(file_path, 'r') as f:
                if file == "file 1":
                    file1_content = f.read()
                elif file == "file 2":
                    file2_content = f.read()
                elif file == "file 3":
                    file3_content = f.read()

        break

    # Wait for a short period before scanning again
    time.sleep(1)

infer_from_audio(audio_prompt = source_audio, text = target_text, language = 'Chinese', accent = 'no-accent', text_prompt = source_text)
