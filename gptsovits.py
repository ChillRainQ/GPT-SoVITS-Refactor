import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import resources.pretrained_models.resources as resources
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from tools.i18n import i18n



class GPTSovits:
    """
    gpt sovits客户端
    """
    class Request:
        pass
    def __init__(self, config):
        if config is None:
            config = {}
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = True and torch.cuda.is_available()
        self.punctuation = set(['!', '?', '…', ',', '.', '-', " "])
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
        self.tokenizer = AutoTokenizer.from_pretrained(resources.BERT_PATH)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(resources.BERT_PATH)
        self.ssl_model = cnhubert.CNHubert(resources.CNHUBERT_PATH)
        self.i18n = i18n.I18nAuto(language=config.get("language", "Auto"))
        self.vq_model, self.hps = self.change_sovits_weights(resources.SOVITS_LIST[0])  # 报错
        self.init()

    def init(self):
        """
        初始化
        """
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
            self.ssl_model = self.ssl_model.half().to(self.device)
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)
            self.ssl_model = self.ssl_model.to(self.device)
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        self.ssl_model.eval()
        self.vq_model.eval()

    def generate_audio(self, request):
        pass

    def generate_audio_stream(self):
        pass

    def language_check(self):
        pass


    def change_sovits_weights(self, sovits_path):
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        return vq_model, hps

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

if __name__ == "__main__":
    gptsovits = GPTSovits(None)
    # gptsovits.change_sovits_weights(resources.SOVITS_LIST[0])