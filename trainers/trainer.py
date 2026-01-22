# here put the import lib
import os
import torch
from tqdm import trange
from utils.earlystop import EarlyStoppingNew
from utils.utils import get_n_params
from models.LLMESR import LLMESR_SASRec, LLMESR_Bert4Rec, LLMESR_GRU4Rec
from models.SASRec import *
from models.GRU4Rec import *
from models.Bert4Rec import *
from models.SASRec_LLM2Rec import SASRec_LLM2Rec, SASRec_WithAlignment
from models.BERT4Rec_LLM2Rec import BERT4Rec_LLM2Rec, BERT4Rec_WithAlignment
from models.GRU4Rec_LLM2Rec import GRU4Rec_LLM2Rec, GRU4Rec_WithAlignment
from models.RCL import RCL_SASRec, RCL_GRU4Rec, RCL_Bert4Rec
from models.ICSRec import ICSRec_SASRec, ICSRec_BERT4Rec, ICSRec_GRU4Rec
from models.LLMEmb import GRU4Rec_LLMEmb, SASRec_LLMEmb, Bert4Rec_LLMEmb
from models.LLMEmb_GAA import GRU4Rec_LLMEmb_GAA, SASRec_LLMEmb_GAA, Bert4Rec_LLMEmb_GAA



class Trainer(object):

    def __init__(self, args, logger, writer, device, generator):
        """Set optimizer, learning rate, early stopping mechanism, load train/validation/test dataloaders, load item and user popularity, load model"""
        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0    # define the start epoch for keepon training

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        logger.info('# of model parameters: ' + str(get_n_params(self.model, logger)))
        
        # Print model structure and gradient information
        self._print_model_structure()

        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        if args.keepon:
            self._load_pretrained_model()

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        
        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)
        self.generator = generator

        # get item pop and user len. item_pop: the array of the target item's popularity
        self.item_pop = generator.get_item_pop()
        self.user_len = generator.get_user_len()

        #self.watch_metric = 'NDCG@10'  # use which metric to select model
        self.watch_metric = args.watch_metric

    
    def _create_model(self):
        '''create your model'''
        # New: lightweight LLM2Rec integration models
        if self.args.model_name in ["sasrec_llm2rec", "sasrec_with_alignment"]:
            if self.args.model_name == "sasrec_llm2rec":
                self.model = SASRec_LLM2Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "sasrec_with_alignment":
                self.model = SASRec_WithAlignment(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["bert4rec_llm2rec", "bert4rec_with_alignment"]:
            if self.args.model_name == "bert4rec_llm2rec":
                self.model = BERT4Rec_LLM2Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "bert4rec_with_alignment":
                self.model = BERT4Rec_WithAlignment(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["gru4rec_llm2rec", "gru4rec_with_alignment"]:
            if self.args.model_name == "gru4rec_llm2rec":
                self.model = GRU4Rec_LLM2Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "gru4rec_with_alignment":
                self.model = GRU4Rec_WithAlignment(self.user_num, self.item_num, self.device, self.args)
        # Original models
        elif self.args.model_name in ["llmesr_mean_sasrec", "llmesr_sasrec", "sasrec"]:
            if self.args.model_name in ["llmesr_mean_sasrec", "llmesr_sasrec"]:
                self.model = LLMESR_SASRec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "sasrec":
                self.model = SASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["llmesr_mean_gru4rec", "llmesr_gru4rec", "gru4rec"]:
            if self.args.model_name in ["llmesr_mean_gru4rec", "llmesr_gru4rec"]:
                self.model = LLMESR_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "gru4rec":
                self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name in ["llmesr_mean_bert4rec", "llmesr_bert4rec", "bert4rec"]:
            if self.args.model_name in ["llmesr_mean_bert4rec", "llmesr_bert4rec"]:
                self.model = LLMESR_Bert4Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "bert4rec":
                self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        # RCL models (without HSU fusion)
        elif self.args.model_name in ["rcl_sasrec", "rcl_gru4rec", "rcl_bert4rec"]:
            if self.args.model_name == "rcl_sasrec":
                self.model = RCL_SASRec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "rcl_gru4rec":
                self.model = RCL_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "rcl_bert4rec":
                self.model = RCL_Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        # RCL + HSU Gated Fusion models (automatically enable HSU fusion)
        elif self.args.model_name in ["rcl_hsu_sasrec", "rcl_hsu_gru4rec", "rcl_hsu_bert4rec"]:
            # Force enable HSU fusion
            self.args.use_hsu_fusion = True
            # Check required parameters
            if self.args.hsu_bank_path is None:
                raise ValueError(
                    f"Model '{self.args.model_name}' requires --hsu_bank_path to be specified. "
                    f"Example: --hsu_bank_path data/{self.args.dataset}/handled/usr_emb_np_qwen_mean.pkl"
                )
            self.logger.info(f"[HSU Fusion] Enabled for model '{self.args.model_name}'")
            self.logger.info(f"[HSU Fusion] Bank path: {self.args.hsu_bank_path}")
            self.logger.info(f"[HSU Fusion] Gate type: {self.args.hsu_gate_type}, init: {self.args.hsu_gate_init}")
            
            if self.args.model_name == "rcl_hsu_sasrec":
                self.model = RCL_SASRec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "rcl_hsu_gru4rec":
                self.model = RCL_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "rcl_hsu_bert4rec":
                self.model = RCL_Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        # ICSRec Intent Contrastive Learning models
        elif self.args.model_name in ["icsrec_sasrec", "icsrec_bert4rec", "icsrec_gru4rec"]:
            if self.args.model_name == "icsrec_sasrec":
                self.model = ICSRec_SASRec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "icsrec_bert4rec":
                self.model = ICSRec_BERT4Rec(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "icsrec_gru4rec":
                self.model = ICSRec_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        # LLMEmb: LLM Empowered Embedding Generator
        elif self.args.model_name in ["llmemb_sasrec", "llmemb_gru4rec", "llmemb_bert4rec"]:
            if self.args.model_name == "llmemb_sasrec":
                self.model = SASRec_LLMEmb(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "llmemb_gru4rec":
                self.model = GRU4Rec_LLMEmb(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "llmemb_bert4rec":
                self.model = Bert4Rec_LLMEmb(self.user_num, self.item_num, self.device, self.args)
        # LLMEmb + GAA (Group-Aware Alignment) 增强版
        elif self.args.model_name in ["llmemb_gaa_sasrec", "llmemb_gaa_gru4rec", "llmemb_gaa_bert4rec"]:
            if self.args.model_name == "llmemb_gaa_sasrec":
                self.model = SASRec_LLMEmb_GAA(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "llmemb_gaa_gru4rec":
                self.model = GRU4Rec_LLMEmb_GAA(self.user_num, self.item_num, self.device, self.args)
            elif self.args.model_name == "llmemb_gaa_bert4rec":
                self.model = Bert4Rec_LLMEmb_GAA(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError
        
        self.model.to(self.device)
    
    
    def _print_model_structure(self):
        """Print model structure and parameter gradient status"""
        self.logger.info("=" * 60)
        self.logger.info("Model Structure and Gradient Status")
        self.logger.info("=" * 60)
        
        # Print model structure
        self.logger.info(f"Model Type: {type(self.model).__name__}")
        self.logger.info("-" * 60)
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # Print parameter information grouped by module
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            grad_status = "✓ trainable" if param.requires_grad else "✗ frozen"
            if param.requires_grad:
                trainable_params += param_count
            else:
                frozen_params += param_count
            self.logger.info(f"  {name}: shape={list(param.shape)}, {grad_status}")
        
        self.logger.info("-" * 60)
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        self.logger.info(f"Frozen Parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        # Check key components
        if hasattr(self.model, 'use_llm2rec') and self.model.use_llm2rec:
            self.logger.info("-" * 60)
            self.logger.info("LLM2Rec Mode (Paper-compliant):")
            self.logger.info("  Item representation: ONLY adapter(llm_item_emb)")
            self.logger.info("  item_emb: FROZEN (not used in forward, not in optimizer)")
            if hasattr(self.model, 'llm_item_emb'):
                self.logger.info(f"  llm_item_emb: shape={list(self.model.llm_item_emb.weight.shape)}, "
                               f"requires_grad={self.model.llm_item_emb.weight.requires_grad}")
            if hasattr(self.model, 'adapter'):
                self.logger.info(f"  adapter: {self.model.adapter}")
            self.logger.info("-" * 60)
            self.logger.info("Sanity Check Tips:")
            self.logger.info("  After 1 training step, verify:")
            self.logger.info("    - item_emb.weight.grad should be None")
            self.logger.info("    - adapter.weight.grad should NOT be None")
            self.logger.info("    - backbone params grad should NOT be None")
        
        if hasattr(self.model, 'alpha'):
            self.logger.info(f"  Alignment alpha: {self.model.alpha}")
        
        # Check HSU fusion module
        if hasattr(self.model, 'use_hsu_fusion') and self.model.use_hsu_fusion:
            self.logger.info("-" * 60)
            self.logger.info("HSU Gated Fusion Mode:")
            self.logger.info(f"  HSU bank entries: {self.model.hsu_bank.num_entries}")
            self.logger.info(f"  HSU dim: {self.model.hsu_bank.hsu_dim}")
            self.logger.info(f"  Gate type: {self.model.hsu_fusion.gate_type}")
            gate_val = self.model.get_hsu_gate_value()
            if gate_val is not None:
                self.logger.info(f"  Initial gate value: {gate_val:.6f} (should be ~0 for degradation)")
            self.logger.info("-" * 60)
            self.logger.info("HSU Fusion Components:")
            self.logger.info(f"  proj: {self.model.hsu_fusion.proj}")
            if hasattr(self.model.hsu_fusion, 'gate_scalar'):
                self.logger.info(f"  gate_scalar: {self.model.hsu_fusion.gate_scalar.item():.4f}")
            if hasattr(self.model.hsu_fusion, 'gate_linear'):
                self.logger.info(f"  gate_linear: {self.model.hsu_fusion.gate_linear}")
        
        self.logger.info("=" * 60)
    

    def _load_pretrained_model(self):

        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        self.start_epoch = checkpoint['epoch']  # load epoch

    
    def _set_optimizer(self):
        # Only pass parameters with requires_grad=True to avoid maintaining optimizer state for frozen parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2,
                                          )

    
    def _set_scheduler(self):
        # default lr_dc_step=1000 lr_dc=0
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)


    def _set_stopper(self):
# default patience=20
        self.stopper = EarlyStoppingNew(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)


    def _train_one_epoch(self, epoch):

        return NotImplementedError
    

    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training,将其按照dict组织"""
        # assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, part_data in enumerate(data):
            inputs[self.generator.train_dataset.var_name[i]] = part_data
        # for i, var_name in enumerate(self.generator.train_dataset.var_name):
            # inputs[var_name] = data[i]

        return inputs
    

    def _prepare_eval_inputs(self, data):
        """Prepare the inputs as a dict for evaluation
        ["seq", "pos", "neg", "positions"]
        """
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs


    def eval(self, epoch=0, test=False):

        return NotImplementedError


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []
        # num_train_epochs 200
        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):
            
            # Set current epoch (for GAA user_bank warmup)
            if hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch)

            t = self._train_one_epoch(epoch)
            
            train_time.append(t)

            # evluate on validation per 20 epochs
            if (epoch % 1) == 0:
                
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                #self.scheduler.step()
                self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler) # stopper.call(). 里面会保存模型权重

                if self.stopper.early_stop:

                    break
        
        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch
    


    def test(self):
        """Do test directly. Set the output dir as the path that save the checkpoint"""
        res = self.eval(test=True)

        return res, -1



    def get_model(self):

        return self.model

    
    def get_model_param_num(self):

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num

        return freeze_num, trainable_num



