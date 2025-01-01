import torch

from TTS_infer_pack.TTS import TTS, TTS_Config
from pydantic import BaseModel


class GPTSovitsTTS:
    class TTS_Request(BaseModel):
        text: str = None
        text_lang: str = None
        ref_audio_path: str = None
        aux_ref_audio_paths: list = None
        prompt_lang: str = None
        prompt_text: str = ""
        top_k: int = 5
        top_p: float = 1
        temperature: float = 1
        text_split_method: str = "cut5"
        batch_size: int = 1
        batch_threshold: float = 0.75
        split_bucket: bool = True
        speed_factor: float = 1.0
        fragment_interval: float = 0.3
        seed: int = -1
        media_type: str = "wav"
        streaming_mode: bool = False
        parallel_infer: bool = True
        repetition_penalty: float = 1.35
    def __init__(self, config):
        self.config = TTS_Config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(self.config)

    def check_params(self, req: dict):
        text: str = req.get("text", "")
        text_lang: str = req.get("text_lang", "")
        ref_audio_path: str = req.get("ref_audio_path", "")
        streaming_mode: bool = req.get("streaming_mode", False)
        media_type: str = req.get("media_type", "wav")
        prompt_lang: str = req.get("prompt_lang", "")
        text_split_method: str = req.get("text_split_method", "cut5")

        if ref_audio_path in [None, ""]:
            raise "ref_audio_path is required"
        if text in [None, ""]:
            raise "text is required"
        if (text_lang in [None, ""]):
            raise "text_lang is required"
        elif text_lang.lower() not in self.config.languages:
            raise "text_lang is not supported"
        if (prompt_lang in [None, ""]):
            raise "prompt_lang is required"
        elif prompt_lang.lower() not in self.config.languages:
            raise "prompt_lang is not supported"
        if media_type not in ["wav", "raw", "ogg", "aac"]:
            raise "media_type is not supported"
        elif media_type == "ogg" and not streaming_mode:
            raise "ogg format is not supported in non-streaming mode"

        if text_split_method not in cut_method_names:
            raise f"text_split_method:{text_split_method} is not supported"

        return None

    def generate_audio(self, request: dict):
        pass
