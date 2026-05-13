
import os
import time
import argparse
import torch
import yaml

from generators.generator import Seq2SeqGeneratorAllUser, GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from generators.icsrec_generator import ICSRecSeq2SeqGenerator, ICSRecSeqGenerator, ICSRecBertGenerator
from trainers.sequence_trainer import SeqTrainer
from trainers.rcl_trainer import RCLTrainer
from trainers.icsrec_trainer import ICSRecTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='hsuga_llmesr_sasrec',
                    choices=[
                    # HSUGA (full method: dual-view + GAA + optional HSU)
                    "hsuga_llmesr_sasrec", "hsuga_llmesr_gru4rec", "hsuga_llmesr_bert4rec",
                    # Legacy names (backward compatible, mapped to HSUGA)
                    "llmesr_mean_sasrec", "llmesr_mean_bert4rec", "llmesr_mean_gru4rec",
                    "llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec",
                    # Pure backbones
                    "sasrec", "bert4rec", "gru4rec",
                    # LLM2Rec baselines
                    "sasrec_llm2rec", "sasrec_with_alignment",
                    "bert4rec_llm2rec", "bert4rec_with_alignment",
                    "gru4rec_llm2rec", "gru4rec_with_alignment",
                    # RCL baselines
                    "rcl_sasrec", "rcl_gru4rec", "rcl_bert4rec",
                    "rcl_hsu_sasrec", "rcl_hsu_gru4rec", "rcl_hsu_bert4rec",
                    # ICSRec baselines
                    "icsrec_sasrec", "icsrec_bert4rec", "icsrec_gru4rec",
                    "icsrec_hsu_sasrec", "icsrec_hsu_bert4rec", "icsrec_hsu_gru4rec",
                    # LLMEmb baselines
                    "llmemb_sasrec", "llmemb_gru4rec", "llmemb_bert4rec",
                    "llmemb_gaa_sasrec", "llmemb_gaa_gru4rec", "llmemb_gaa_bert4rec"
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="steam", 
                    choices=["yelp", "fashion", "beauty","steam"],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--run_id",
                    default=None,
                    type=str,
                    help="Override run_id (used by run_train.sh for consistency)")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_item_emb",
                    default=False,
                    action="store_true",
                    help="save the item embedding from SRS model (for LLMEmb alignment)")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=None,  
                    help="the threshold to split the short and long seq (auto-set by dataset if not specified)")
parser.add_argument("--ts_item",
                    type=int,
                    default=None, 
                    help="the threshold to split the long-tail and popular items (auto-set by dataset if not specified)")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=1,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
# TODO 
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
# TODO 
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
# TODO cosine similarity
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--sim_long_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users of long users for enhancement")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=4,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=8,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")
parser.add_argument("--hidden_mode", 
                    nargs='?',
                    const='',
                    default='_qwen_mean_llm',
                    help="last or mean of LLM's last hidden method")
parser.add_argument("--filter_item",
                    default=False,
                    action="store_true",
                    help="whether filter the item")
parser.add_argument("--weight_sum",
                    default=False,
                    action="store_true",
                    help="whether mean or weight sum sim users ")
parser.add_argument("--do_analysis",
                    default=False,
                    action="store_true",
                    help="whether do test to analy user len and pred_rank")
parser.add_argument("--similar_gate",
                    type=float,
                    default=-1,
                    help="the gate for filter sim_users")
parser.add_argument("--filter_similar_metric",
                    type=str,
                    default='pearson',
                    help="which metric is used to filter similar users.")
parser.add_argument("--filter_similar_user",
                    action='store_true', 
                    default=False,
                    help="whether filter the similar users")
parser.add_argument("--sim_filter_percentile",
                    type=float,
                    default=0.0,
                    help="Filter similar users using relative percentile (0.0-1.0). "
                         "0.0 means disabled, 0.5 means keep top 50% of similar users based on their own score distribution. "
                         "This is per-user adaptive filtering, different from global similar_gate threshold.")
parser.add_argument("--item_from_llm",
                    action='store_true', 
                    default=False,
                    help="whether the item embedding is from LLM")
parser.add_argument("--use_llm2rec",
                    action='store_true', 
                    default=False,
                    help="whether use LLM2Rec pretrained item embeddings")
parser.add_argument("--llm2rec_emb_path",
                    type=str,
                    default=None,  # Auto-select based on dataset if None
                    help="path to LLM2Rec pretrained embeddings")

# ============================================================================
# LLMEmb parameters (Large Language Model Empowered Embedding Generator)
# ============================================================================
parser.add_argument("--llmemb_path",
                    type=str,
                    default=None,  # Auto-select based on dataset if None: data/{dataset}/handled/llmemb_pca.pkl
                    help="path to LLMEmb pretrained LLM item embeddings")
parser.add_argument("--srs_emb_path",
                    type=str,
                    default=None,  # Auto-select based on dataset if None: data/{dataset}/handled/pca64_itm_emb_np.pkl
                    help="path to SRS item embeddings for alignment (used by LLMEmb)")
parser.add_argument("--freeze_emb",
                    action='store_true',
                    default=False,
                    help="whether to freeze the LLM item embeddings in LLMEmb")
# LLMEmb-GAA specific parameters
parser.add_argument("--gaa_alpha",
                    type=float,
                    default=0.1,
                    help="weight for GAA (Group-Aware Alignment) loss in LLMEmb-GAA")
parser.add_argument("--gaa_tau",
                    type=float,
                    default=1.0,
                    help="temperature for GAA contrastive loss (when user_sim_func='cl')")
parser.add_argument("--use_dynamic_k",
                    action='store_true',
                    default=False,
                    help="Whether to use dynamic K value (based on user interaction length + consistency calculation). "
                         "When enabled, compute recall count for each user online based on w1*s_len + w2*s_div")
parser.add_argument("--dynamic_k_w1",
                    type=float,
                    default=0.5,
                    help="Weight of length signal s_len in dynamic K (short sequences need more similar users)")
parser.add_argument("--dynamic_k_w2",
                    type=float,
                    default=0.5,
                    help="Weight of divergence signal s_div in dynamic K (divergent users need more similar users)")
parser.add_argument("--dynamic_k_min",
                    type=int,
                    default=2,
                    help="Minimum value of dynamic K")
parser.add_argument("--dynamic_k_max",
                    type=int,
                    default=18,
                    help="Maximum value of dynamic K")
parser.add_argument("--dynamic_k_analysis_path",
                    type=str,
                    default=None,
                    help="User analysis file path (CSV). Default: data/analysis_output/user_dynamic_k_analysis.csv")

# GAA User Bank parameters (avoid repeated encoding of similar user sequences)
parser.add_argument("--gaa_use_user_bank",
                    action='store_true',
                    default=False,
                    help="Whether to enable user_bank mode. True: get similar user representations from bank; False: encode each time (original behavior)")
parser.add_argument("--gaa_bank_momentum",
                    type=float,
                    default=0.9,
                    help="user_bank EMA update coefficient (0.9~0.99). Larger values are more stable, smaller values respond faster")
parser.add_argument("--gaa_bank_dtype",
                    type=str,
                    default="fp32",
                    choices=["fp32", "fp16"],
                    help="user_bank storage precision")
parser.add_argument("--gaa_bank_on_cpu",
                    action='store_true',
                    default=True,
                    help="Whether to store user_bank on CPU (recommended True, saves GPU memory)")
parser.add_argument("--gaa_warmup_epochs",
                    type=int,
                    default=1,
                    help="Bank warmup epochs, only update bank without computing GAA loss for first N epochs")
parser.add_argument("--gaa_bank_min_fill_ratio",
                    type=float,
                    default=0.1,
                    help="Bank minimum fill ratio, skip GAA loss when coverage is insufficient")

# RCL specific parameters
parser.add_argument("--rcl_ssl",
                    type=int,
                    default=8,
                    help="RCL SSL type (0=no SSL, 1-4=basic CL, 5-11=RCL)")
parser.add_argument("--rcl_scale",
                    type=float,
                    default=0.1,
                    help="RCL SSL loss scale")
parser.add_argument("--rcl_neg_size",
                    type=int,
                    default=1,
                    help="RCL number of hard negative samples")
parser.add_argument("--rcl_perc",
                    type=int,
                    default=95,
                    help="RCL percentile for positive user selection")
parser.add_argument("--rcl_neg_perc1",
                    type=int,
                    default=90,
                    help="RCL upper percentile for hard negative selection")
parser.add_argument("--rcl_neg_perc2",
                    type=int,
                    default=80,
                    help="RCL lower percentile for hard negative selection")
parser.add_argument("--rcl_smooth_loss",
                    type=float,
                    default=0.1,
                    help="RCL smooth loss weight")
parser.add_argument("--rcl_max",
                    type=int,
                    default=1,
                    help="RCL max strategy (0 or 1)")

# ============================================================================
# HSU Gated Fusion parameters (B2: Gated Fusion)
# Used to inject offline-computed HSU user semantic vectors into RCL user representations
# ============================================================================
parser.add_argument("--use_hsu_fusion",
                    action='store_true',
                    default=False,
                    help="Whether to enable HSU gated fusion. When disabled, model behavior is identical to original RCL")
parser.add_argument("--hsu_bank_path",
                    type=str,
                    default=None,
                    help="HSU bank file path (pkl/npy/pt). Example: data/steam/handled/usr_emb_np_qwen_mean.pkl")
parser.add_argument("--hsu_dim",
                    type=int,
                    default=3584,
                    help="HSU vector original dimension D_h (automatically updated to pca_dim when PCA is enabled)")

# === HSU PCA dimensionality reduction branch ===
# Motivation: Compress high-dimensional static semantic vectors (3584-dim) into a more compact semantic subspace,
#             reduce fusion parameters, improve training stability, reduce interference with main ranking learning
parser.add_argument("--use_hsu_pca",
                    action='store_true',
                    default=False,
                    help="Whether to enable HSU PCA dimensionality reduction branch. "
                         "When enabled, automatically load preprocessed PCA file (e.g., usr_emb_np_qwen_mean_pca512.pkl)")
parser.add_argument("--hsu_pca_dim",
                    type=int,
                    default=512,
                    choices=[64, 128, 256, 512],
                    help="HSU PCA dimensionality after reduction. Ensure corresponding PCA file is generated via data/pca_reduce_hsu.py")
parser.add_argument("--hsu_gate_type",
                    type=str,
                    default="scalar",
                    choices=["scalar", "vector"],
                    help="[Compatible with old parameter] Gate type, recommend using --hsu_fusion_type instead")
parser.add_argument("--hsu_fusion_type",
                    type=str,
                    default=None,
                    choices=["scalar", "vector", "attention", "mlp", "adaptive", 
                             "cross_attention", "bilinear", "residual_mlp"],
                    help="Fusion type: scalar/vector (basic gating), attention/mlp/adaptive/cross_attention/bilinear/residual_mlp (advanced fusion)")
parser.add_argument("--hsu_gate_init",
                    type=float,
                    default=-10.0,
                    help="Gate initialization value. Default -10.0, makes sigmoid(gate)≈0, ensuring initial degradation to original model")
parser.add_argument("--hsu_fusion_dropout",
                    type=float,
                    default=0.1,
                    help="Dropout probability during HSU fusion")
parser.add_argument("--hsu_num_heads",
                    type=int,
                    default=4,
                    help="Number of attention heads, used for attention/cross_attention fusion types")
parser.add_argument("--hsu_mlp_ratio",
                    type=float,
                    default=2.0,
                    help="MLP expansion ratio, used for mlp/residual_mlp fusion types")
parser.add_argument("--hsu_use_layernorm",
                    action='store_true',
                    default=False,
                    help="Whether to apply LayerNorm to HSU vectors")
parser.add_argument("--hsu_proj_type",
                    type=str,
                    default="linear",
                    choices=["linear", "bottleneck", "lowrank"],
                    help="HSU projection type: linear (single layer, more parameters), bottleneck (bottleneck structure, progressive compression), lowrank (low-rank decomposition, fewer parameters)")
parser.add_argument("--hsu_proj_bottleneck_dim",
                    type=int,
                    default=512,
                    help="HSU projection bottleneck layer dimension, used for bottleneck/lowrank types. Recommended 256-512")

# ============================================================================
# ICSRec Intent Contrastive Learning parameters
# ============================================================================
parser.add_argument("--intent_num",
                    type=int,
                    default=512,
                    help="ICSRec intent clustering number")
parser.add_argument("--ics_temperature",
                    type=float,
                    default=1.0,
                    help="ICSRec contrastive learning temperature parameter")
parser.add_argument("--ics_sim",
                    type=str,
                    default="dot",
                    choices=["dot", "cos"],
                    help="ICSRec similarity calculation method")
parser.add_argument("--ics_lambda",
                    type=float,
                    default=0.1,
                    help="ICSRec CICL (Coarse-grained Intent Contrastive Learning) loss weight")
parser.add_argument("--ics_beta",
                    type=float,
                    default=0.1,
                    help="ICSRec FICL (Fine-grained Intent Contrastive Learning) loss weight")
parser.add_argument("--ics_rec_weight",
                    type=float,
                    default=1.0,
                    help="ICSRec recommendation task loss weight")
parser.add_argument("--ics_cl_mode",
                    type=str,
                    default="cf",
                    choices=["c", "f", "cf"],
                    help="ICSRec contrastive learning mode: c=only CICL, f=only FICL, cf=both")
parser.add_argument("--ics_use_fnm",
                    action='store_true',
                    default=True,
                    help="ICSRec whether to use False Negative Mining")

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()

# Automatically set default values for ts_user and ts_item based on dataset
DATASET_DEFAULTS = {
    "beauty": {"ts_user": 9, "ts_item": 4},
    "fashion": {"ts_user": 3, "ts_item": 4},
    "steam": {"ts_user": 16, "ts_item": 69},
}
if args.ts_user is None:
    args.ts_user = DATASET_DEFAULTS.get(args.dataset.lower(), {}).get("ts_user", 16)
if args.ts_item is None:
    args.ts_item = DATASET_DEFAULTS.get(args.dataset.lower(), {}).get("ts_item", 69)

set_seed(args.seed) # fix the random seed

# --- Run Management: generate run_id and unified output directory ---
_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
if args.run_id is None:
    _run_id = f"{args.model_name}_{args.dataset}_{_time_str}"
else:
    _run_id = args.run_id
args.run_id = _run_id
args.run_dir = os.path.join("./runs", _run_id)
args.output_dir = os.path.join(args.run_dir, "checkpoints")
args.pretrain_dir = os.path.join("./saved", args.dataset, args.pretrain_dir)
args.keepon_path = os.path.join(args.output_dir, "pytorch_model.bin")

os.makedirs(args.run_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.run_dir, "tensorboard"), exist_ok=True)
os.makedirs(os.path.join(args.run_dir, "results"), exist_ok=True)

print(f"\n{'='*64}")
print(f"[RUN] ID:  {_run_id}")
print(f"[RUN] Dir: {os.path.abspath(args.run_dir)}")
print(f"{'='*64}\n")

_config_snapshot = {k: str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v
                    for k, v in vars(args).items()}
with open(os.path.join(args.run_dir, "config.yaml"), "w") as _f:
    yaml.dump(_config_snapshot, _f, default_flow_style=False, allow_unicode=True)


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    # generator is used to manage dataset; Load train, validation, test dataset
    # Original *AllUser Dataset already returns user_id, RCL can use it directly
    # Generator selection: consistent with llmesr, different backbones use corresponding Generators
    # Data format description:
    #   - GRU4Rec (GeneratorAllUser):     pos/neg are single values [B]
    #   - Bert4Rec (BertGeneratorAllUser): pos/neg are sequences [B, T] (labels/neg_labels)
    #   - SASRec (Seq2SeqGeneratorAllUser): pos/neg are sequences [B, T]
    #   - ICSRec: uses dedicated Generator, supports dual-view data augmentation
    
    # ICSRec models use dedicated Generators (including HSU fusion versions)
    if args.model_name in ["icsrec_gru4rec", "icsrec_hsu_gru4rec"]:
        generator = ICSRecSeqGenerator(args, logger, device)
    elif args.model_name in ["icsrec_bert4rec", "icsrec_hsu_bert4rec"]:
        generator = ICSRecBertGenerator(args, logger, device)
    elif args.model_name in ["icsrec_sasrec", "icsrec_hsu_sasrec"]:
        generator = ICSRecSeq2SeqGenerator(args, logger, device)
    # Original models
    elif args.model_name in ['hsuga_llmesr_gru4rec', 'llmesr_mean_gru4rec', 'llmesr_gru4rec', "gru4rec", "gru4rec_llm2rec", "gru4rec_with_alignment",
                           "rcl_gru4rec", "rcl_hsu_gru4rec", "llmemb_gru4rec", "llmemb_gaa_gru4rec"]:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["hsuga_llmesr_bert4rec", "llmesr_mean_bert4rec", "llmesr_bert4rec", "bert4rec", "bert4rec_llm2rec", "bert4rec_with_alignment",
                             "rcl_bert4rec", "rcl_hsu_bert4rec", "llmemb_bert4rec", "llmemb_gaa_bert4rec"]:
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name in ["hsuga_llmesr_sasrec", "llmesr_mean_sasrec", "llmesr_sasrec", "sasrec", "sasrec_llm2rec", "sasrec_with_alignment",
                             "rcl_sasrec", "rcl_hsu_sasrec", "llmemb_sasrec", "llmemb_gaa_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # Select Trainer:
    #   - ICSRec models (including ICSRec+HSU) use ICSRecTrainer
    #   - RCL models (including RCL+HSU) use RCLTrainer
    #   - Others use SeqTrainer
    if args.model_name in ["icsrec_sasrec", "icsrec_bert4rec", "icsrec_gru4rec",
                           "icsrec_hsu_sasrec", "icsrec_hsu_bert4rec", "icsrec_hsu_gru4rec"]:
        trainer = ICSRecTrainer(args, logger, writer, device, generator)
    elif args.model_name in ["rcl_sasrec", "rcl_gru4rec", "rcl_bert4rec", "rcl_hsu_sasrec", "rcl_hsu_gru4rec", "rcl_hsu_bert4rec"]:
        trainer = RCLTrainer(args, logger, writer, device, generator)
    else:
        trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test: # false
        trainer.test()
    elif args.do_emb: # false
        trainer.save_user_emb()
    elif args.do_item_emb:  # Save item embedding (for LLMEmb)
        trainer.save_item_emb()
    elif args.do_group: # false
        trainer.test_group()
    elif args.do_analysis:
        pass
        # trainer.do_analysis()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



