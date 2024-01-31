import os
import numpy as np
import json
from pydub import AudioSegment
import librosa
from typing import Any, Union, Optional, List, Tuple, Dict
import re
import pickle
from torchaudio.transforms import MelSpectrogram ,TimeMasking, FrequencyMasking
import torch
import torch.nn.functional as F
from torchtext.vocab import Vocab, vocab as create_vocab

from pyctcdecode import build_ctcdecoder

MAX_AUDIO_VALUE = 32768

class Wav2VecProcessor:
    def __init__(self, vocab_path: str, unk_token: str = "<unk>", pad_token: str = "<pad>", word_delim_token: str = "|", sampling_rate: int = 16000, num_mels: int = 80, n_fft: int = 400, hop_length: int = 160, win_length: int = 400, fmin: float = 0.0, fmax: float = 8000.0, freq_augment: int = 27, time_augment: int = 10, time_mask_ratio: float = 0.05, puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])", lm_path: Optional[str] = None, beam_alpha: float = 2.1, beam_beta: float = 9.2) -> None:
        # Text
        self.dictionary = self.create_vocab(vocab_path, pad_token, word_delim_token, unk_token)

        self.word_delim_item = word_delim_token

        self.unk_token = self.find_token(unk_token)
        self.pad_token = self.find_token(pad_token)
        self.word_delim_token = self.find_token(word_delim_token)

        self.special_tokens = [unk_token, pad_token]

        self.puncs = puncs

        if lm_path is not None and os.path.exists(lm_path):
            self.ctc_lm = build_ctcdecoder(
                labels=self.dictionary.get_itos(),
                kenlm_model_path=lm_path,
                alpha=beam_alpha,
                beta=beam_beta
            )
            
        # Audio
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.hop_length = hop_length

        self.mel_transform = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels
        )

        self.freq_masker = FrequencyMasking(freq_mask_param=freq_augment)
        self.time_masker = TimeMasking(time_mask_param=time_augment, p=time_mask_ratio)

    def create_vocab(self, vocab_path: str, pad_token: str, word_delim_token: str, unk_token: str) -> Vocab:
        vocabs = json.load(open(vocab_path, encoding='utf-8'))
        dictionary = dict()
        count = 0
        for item in vocabs:
            count += 1
            dictionary[item] = count

        vocab = Vocab(
            vocab=create_vocab(
                dictionary,
                specials=[pad_token]
            ))
    
        vocab.insert_token(word_delim_token, index=len(vocab))
        vocab.insert_token(unk_token, index=len(vocab))
        
        return vocab

    def read_pickle(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            signal = pickle.load(file)

        signal = librosa.resample(y=signal, orig_sr=8000, target_sr=self.sampling_rate)

        return signal
    
    def read_pcm(self, path: str) -> np.ndarray:
        audio = AudioSegment.from_file(path, frame_rate=8000, channels=1, sample_width=2).set_frame_rate(self.sampling_rate).get_array_of_samples()
        return np.array(audio).astype(np.float64) / MAX_AUDIO_VALUE
    
    def read_audio(self, path: str, role: Optional[int] = None) -> np.ndarray:
        signal, _ = librosa.load(path, sr=self.sampling_rate, mono=False)
        if signal.ndim == 2 and role is not None:
            signal = signal[role]
        return signal
    
    def spectral_normalize(self, x: torch.Tensor, C=1, clip_val=1e-5) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(signal)
        log_mel = self.spectral_normalize(mel_spec)
        return log_mel
        
    def standard_normalize(self, signal: torch.Tensor) -> torch.Tensor:
        return (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None, role: Optional[int] = None) -> torch.Tensor:
        if ".pickle" in path:
            signal = self.read_pickle(path)
        elif ".pcm" in path:
            signal = self.read_pcm(path)
        elif ".mp3" in path or ".flac" in path:
            signal = self.read_audio(path)
        elif ".wav" in path:
            signal = self.read_audio(path, role)

        if start is not None and end is not None:
            signal = signal[int(start * self.sampling_rate) : int(end * self.sampling_rate)]

        signal = torch.FloatTensor(signal)
        return signal

    def load_vocab(self, path: str) -> List[str]:
        if os.path.exists(path):
            return json.load(open(path, encoding='utf-8'))
    
    def find_token(self, char: str) -> int:
        if char in self.dictionary:
            return self.dictionary.__getitem__(char)
        return self.unk_token

    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def text2token(self, sentence: str) -> torch.Tensor:
        sentence = self.clean_text(sentence)
        sentence = sentence.replace(" ", self.word_delim_item)
        tokens = []
        for char in [*sentence]:
            tokens.append(self.find_token(char))
        return torch.tensor(tokens)
    
    def decode_beam_search(self, digit: np.ndarray, beam_width: int = 4, beam_prune_logp: float = -20.0):
        return self.ctc_lm.decode(
                    digit,
                    hotwords=self.beam_config['hotwords'],
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp
                )

    def decode_batch(self, digits: Union[torch.Tensor, np.ndarray, list], group_token: bool = True) -> List[str]:
        sentences = []

        for logit in digits:
            if group_token:
                logit = self.group_tokens(logit)
            sentences.append(self.token2text(logit))
        
        return sentences
    
    def spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_masker(x)
        x = self.time_masker(x)

        return x
    
    def group_tokens(self, logits: Union[torch.Tensor, np.ndarray], length: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        items = []
        prev_item = None

        if length is None:
            length = length = len(logits)

        for i in range(length):
            if prev_item is None:
                items.append(logits[i])
                prev_item = logits[i]
                continue
            
            if logits[i] == self.pad_token:
                prev_item = None
                continue

            if logits[i] == prev_item:
                continue

            items.append(logits[i])
            prev_item = logits[i]
        return items

    def token2text(self, tokens: np.ndarray) -> str:
        text = ""
        for token in tokens:
            if token == self.word_delim_token:
                text += " "
            elif token >= 0:
                text += self.dictionary.lookup_token(token)
            else:
                break
        for item in self.special_tokens:
            text = text.replace(item, "")
        text = re.sub(r"\s\s+", " ", text)
        return text.strip()
    
    def generate_mask(self, lengths: List[int], max_len: Optional[int] = None) -> torch.Tensor:
        masks = []

        if max_len is None:
            max_len = np.max(lengths)

        for length in lengths:
            masks.append(torch.tensor(np.array([True] * length + [False] * (max_len - length), dtype=bool)))
        
        return torch.stack(masks)
    
    def __call__(self, signals: List[torch.Tensor], max_len: Optional[int] = None, return_length: bool = False, set_augment: bool = False) -> torch.Tensor:
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        mels = []
        mel_lengths = []

        for signal in signals:
            signal_length = len(signal)
            padded_signal = F.pad(signal, (0, max_len - signal_length), mode='constant', value=0.0)
            mels.append(self.mel_spectrogram(padded_signal))
            mel_lengths.append((signal_length // self.hop_length) + 1)

        mels = torch.stack(mels).type(torch.FloatTensor)
        if set_augment:
            mels = self.spec_augment(mels)

        if return_length:
            return mels, torch.tensor(mel_lengths)
        
        return mels
    
    def tokenize(self, sentences: List[str], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lengths = []
        for sentence in sentences:
            token = self.text2token(sentence)
            lengths.append(len(token))
            tokens.append(token)

        if max_len is None:
            max_len = np.max(lengths)

        padded_tokens = []
    
        for index, token in enumerate(tokens):
            padded_tokens.append(F.pad(token, (0, max_len - lengths[index]), mode='constant', value=self.pad_token))

        return torch.stack(padded_tokens), torch.tensor(lengths)