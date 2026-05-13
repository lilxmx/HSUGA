from hsuga.models.hsuga import HSUGA_SASRec, HSUGA_GRU4Rec, HSUGA_Bert4Rec
from hsuga.models.backbones import SASRec, SASRec_seq, GRU4Rec, GRU4Rec_seq, Bert4Rec

MODEL_REGISTRY = {
    "hsuga_sasrec": HSUGA_SASRec,
    "hsuga_gru4rec": HSUGA_GRU4Rec,
    "hsuga_bert4rec": HSUGA_Bert4Rec,
    "llmesr_mean_sasrec": HSUGA_SASRec,
    "llmesr_sasrec": HSUGA_SASRec,
    "llmesr_mean_gru4rec": HSUGA_GRU4Rec,
    "llmesr_gru4rec": HSUGA_GRU4Rec,
    "llmesr_mean_bert4rec": HSUGA_Bert4Rec,
    "llmesr_bert4rec": HSUGA_Bert4Rec,
    "sasrec": SASRec_seq,
    "gru4rec": GRU4Rec,
    "bert4rec": Bert4Rec,
}


def build_model(cfg, user_num, item_num, device):
    model_name = cfg.model.name
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](user_num, item_num, device, cfg)
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
