# -*- coding: utf-8 -*-
"""
ICSRec 训练器
继承 HSUGA 的 SeqTrainer，添加 ICSRec 特有的对比学习训练逻辑
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
import gc

from trainers.sequence_trainer import SeqTrainer
from trainers.trainer import Trainer
from baselines.icsrec.clustering import create_kmeans
from baselines.icsrec.losses import ICLLoss
from models.ICSRec import (
    ICSRec_SASRec, ICSRec_BERT4Rec, ICSRec_GRU4Rec,
    ICSRec_HSU_SASRec, ICSRec_HSU_BERT4Rec, ICSRec_HSU_GRU4Rec
)
from utils.utils import metric_report, metric_len_report, metric_pop_report, record_csv


class ICSRecTrainer(SeqTrainer):
    """
    ICSRec 训练器
    
    核心特点：
    1. 每个 epoch 开始时进行意图聚类
    2. 计算 CICL (粗粒度意图对比学习) 损失
    3. 计算 FICL (细粒度意图对比学习) 损失
    4. 多任务学习：推荐损失 + 对比学习损失
    """
    
    def __init__(self, args, logger, writer, device, generator):
        # 初始化父类
        super().__init__(args, logger, writer, device, generator)
        
        # ICSRec 特有参数
        self.intent_num = getattr(args, 'intent_num', 512)
        self.temperature = getattr(args, 'ics_temperature', 1.0)
        self.sim = getattr(args, 'ics_sim', 'dot')
        self.lambda_0 = getattr(args, 'ics_lambda', 0.1)  # CICL 权重
        self.beta_0 = getattr(args, 'ics_beta', 0.1)      # FICL 权重
        self.rec_weight = getattr(args, 'ics_rec_weight', 1.0)  # 推荐损失权重
        self.cl_mode = getattr(args, 'ics_cl_mode', 'cf')  # 'c', 'f', 'cf'
        self.use_fnm = getattr(args, 'ics_use_fnm', True)  # False Negative Mining
        
        # 初始化聚类器
        self.cluster = create_kmeans(
            num_cluster=self.intent_num,
            seed=getattr(args, 'seed', 42),
            hidden_size=args.hidden_size,
            gpu_id=getattr(args, 'gpu_id', 0),
            device=device,
            use_gpu=torch.cuda.is_available()
        )
        
        # 初始化对比学习损失
        self.icl_loss_fn = ICLLoss(
            temperature=self.temperature,
            sim=self.sim,
            use_fnm=self.use_fnm
        )
        
        # 创建聚类用的 DataLoader
        self.cluster_loader = generator.make_clusterloader()
        
        self.logger.info("=" * 60)
        self.logger.info("ICSRec Trainer Configuration")
        self.logger.info("=" * 60)
        self.logger.info(f"  Intent Clusters: {self.intent_num}")
        self.logger.info(f"  Temperature: {self.temperature}")
        self.logger.info(f"  Similarity: {self.sim}")
        self.logger.info(f"  CL Mode: {self.cl_mode}")
        self.logger.info(f"  Use FNM: {self.use_fnm}")
        self.logger.info(f"  Lambda (CICL): {self.lambda_0}")
        self.logger.info(f"  Beta (FICL): {self.beta_0}")
        self.logger.info(f"  Rec Weight: {self.rec_weight}")
        self.logger.info("=" * 60)
    
    def _create_model(self):
        """创建 ICSRec 模型"""
        # 原始 ICSRec 模型
        if self.args.model_name == "icsrec_sasrec":
            self.model = ICSRec_SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "icsrec_bert4rec":
            self.model = ICSRec_BERT4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "icsrec_gru4rec":
            self.model = ICSRec_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        # ICSRec + HSU 融合增强模型
        elif self.args.model_name == "icsrec_hsu_sasrec":
            self.args.use_hsu_fusion = True  # 强制启用 HSU 融合
            self._check_hsu_config()
            self.model = ICSRec_HSU_SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "icsrec_hsu_bert4rec":
            self.args.use_hsu_fusion = True
            self._check_hsu_config()
            self.model = ICSRec_HSU_BERT4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "icsrec_hsu_gru4rec":
            self.args.use_hsu_fusion = True
            self._check_hsu_config()
            self.model = ICSRec_HSU_GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        else:
            raise ValueError(f"Unknown ICSRec model: {self.args.model_name}")
        
        self.model.to(self.device)
    
    def _check_hsu_config(self):
        """检查 HSU 融合所需的配置"""
        if self.args.hsu_bank_path is None:
            raise ValueError(
                f"Model '{self.args.model_name}' requires --hsu_bank_path to be specified. "
                f"Example: --hsu_bank_path data/{self.args.dataset}/handled/usr_emb_np_qwen_mean.pkl"
            )
        self.logger.info(f"[HSU Fusion] Enabled for model '{self.args.model_name}'")
        self.logger.info(f"[HSU Fusion] Bank path: {self.args.hsu_bank_path}")
        self.logger.info(f"[HSU Fusion] Gate type: {self.args.hsu_gate_type}, init: {self.args.hsu_gate_init}")
    
    def _train_cluster(self):
        """
        训练意图聚类器
        在每个 epoch 开始时调用
        """
        if self.cl_mode not in ['f', 'cf']:
            return  # 如果不使用 FICL，跳过聚类
        
        self.logger.info("Training Intent Clusters...")
        self.model.eval()
        
        kmeans_training_data = []
        is_hsu_model = "hsu" in self.args.model_name
        
        with torch.no_grad():
            for batch in tqdm(self.cluster_loader, desc="Clustering"):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self._prepare_train_inputs(batch)
                
                # 获取序列表示（HSU 模型需要传递 user_id）
                if self.args.model_name in ["icsrec_gru4rec"]:
                    seq_output = self.model.log2feats(inputs["seq"])
                elif self.args.model_name in ["icsrec_hsu_gru4rec"]:
                    user_id = inputs.get("user_id")
                    seq_output = self.model.log2feats(inputs["seq"], user_id=user_id)
                elif is_hsu_model:
                    user_id = inputs.get("user_id")
                    seq_output = self.model.log2feats(inputs["seq"], inputs["positions"], user_id=user_id)
                else:
                    seq_output = self.model.log2feats(inputs["seq"], inputs["positions"])
                
                # 取最后一个时间步
                seq_output = seq_output[:, -1, :].detach().cpu().numpy()
                kmeans_training_data.append(seq_output)
        
        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
        
        # 训练聚类器
        self.cluster.train(kmeans_training_data)
        
        # 清理内存
        del kmeans_training_data
        gc.collect()
        
        self.logger.info(f"Cluster training completed with {self.intent_num} clusters")
    
    def _train_one_epoch(self, epoch):
        """
        训练一个 epoch
        
        ICSRec 的训练流程（与原始ICSRec保持一致）：
        1. 进行意图聚类（每个 epoch 开始时，如果使用FICL）
        2. 计算推荐损失（CrossEntropyLoss，只对最后一个位置）
        3. 计算 CICL 损失（粗粒度对比学习）
        4. 计算 FICL 损失（细粒度对比学习）
        5. 多任务优化
        """
        # 1. 训练意图聚类
        self._train_cluster()
        
        # 2. 模型训练
        tr_loss = 0
        rec_loss_total = 0
        icl_loss_total = 0
        nb_tr_steps = 0
        train_time = []
        
        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        
        for batch in prog_iter:
            batch = tuple(t.to(self.device) for t in batch)
            train_start = time.time()
            
            inputs = self._prepare_train_inputs(batch)
            
            # ===== 使用 target_item 作为目标（与原始ICSRec一致） =====
            # 注意：对于短序列，pos[:, -1] 可能是padding (0)，
            # 但 target_item 始终是正确的目标物品ID
            target_for_rec = inputs["target_item"]  # 用于推荐损失
            target_for_fnm = inputs["target_item"]  # 用于FNM（与原始ICSRec的target_pos_1[:,-1]等价）
            
            # Sanity check: 仅在第一个epoch的第一个batch检查（使用instance变量避免重复）
            if nb_tr_steps == 0 and epoch == 0:
                # 验证 target_item 不是padding (0)
                num_padding = torch.sum(target_for_rec == 0).item()
                if num_padding > 0:
                    self.logger.warning(f"[Sanity Check] {num_padding}/{len(target_for_rec)} samples have padding as target!")
                # 验证 logits shape
                with torch.no_grad():
                    if self.args.model_name == "icsrec_gru4rec":
                        test_output = self.model.log2feats(inputs["seq"])
                    else:
                        test_output = self.model.log2feats(inputs["seq"], inputs["positions"])
                    test_logits = self.model.predict_full(test_output[:, -1, :])
                    expected_shape = (inputs["seq"].shape[0], self.item_num + 2)
                    if test_logits.shape != expected_shape:
                        self.logger.error(f"Logits shape error: expected {expected_shape}, got {test_logits.shape}")
                    else:
                        self.logger.info(f"[Sanity Check PASSED] logits shape correct: {test_logits.shape}")
            
            # 获取两个视图的序列表示（用于对比学习）
            # HSU 模型需要传递 user_id
            is_hsu_model = "hsu" in self.args.model_name
            
            if self.args.model_name in ["icsrec_gru4rec"]:
                seq_output_1 = self.model.log2feats(inputs["seq"])
                seq_output_2 = self.model.log2feats(inputs["seq_aug"])
            elif self.args.model_name in ["icsrec_hsu_gru4rec"]:
                user_id = inputs.get("user_id")
                seq_output_1 = self.model.log2feats(inputs["seq"], user_id=user_id)
                seq_output_2 = self.model.log2feats(inputs["seq_aug"], user_id=user_id)
            elif is_hsu_model:
                user_id = inputs.get("user_id")
                seq_output_1 = self.model.log2feats(inputs["seq"], inputs["positions"], user_id=user_id)
                seq_output_2 = self.model.log2feats(inputs["seq_aug"], inputs["positions_aug"], user_id=user_id)
            else:
                seq_output_1 = self.model.log2feats(inputs["seq"], inputs["positions"])
                seq_output_2 = self.model.log2feats(inputs["seq_aug"], inputs["positions_aug"])
            
            # 计算推荐损失（与原始ICSRec一致，使用CrossEntropyLoss）
            # HSU 模型的 forward 也需要 user_id
            if is_hsu_model:
                user_id = inputs.get("user_id")
                rec_loss = self.model(
                    inputs["seq"], 
                    inputs["positions"], 
                    target_for_rec,  # 使用 target_item
                    user_id=user_id
                )
            else:
                rec_loss = self.model(
                    inputs["seq"], 
                    inputs["positions"], 
                    target_for_rec  # 使用 target_item
                )
            
            # 计算对比学习损失
            # FNM使用 target_item 作为intent_id
            icl_loss = self.icl_loss_fn(
                seq_output_1, seq_output_2,
                cluster=self.cluster,
                target_item=target_for_fnm,  # 使用 target_item
                lambda_0=self.lambda_0,
                beta_0=self.beta_0,
                cl_mode=self.cl_mode
            )
            
            # 总损失
            joint_loss = self.rec_weight * rec_loss + icl_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            joint_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            tr_loss += joint_loss.item()
            rec_loss_total += rec_loss.item()
            if isinstance(icl_loss, torch.Tensor):
                icl_loss_total += icl_loss.item()
            else:
                icl_loss_total += icl_loss
            nb_tr_steps += 1
            
            # 显示损失
            prog_iter.set_postfix(
                loss='%.4f' % (tr_loss / nb_tr_steps),
                rec='%.4f' % (rec_loss_total / nb_tr_steps),
                icl='%.4f' % (icl_loss_total / nb_tr_steps)
            )
            
            train_end = time.time()
            train_time.append(train_end - train_start)
        
        # 记录到 TensorBoard
        self.writer.add_scalar('train/joint_loss', tr_loss / nb_tr_steps, epoch)
        self.writer.add_scalar('train/rec_loss', rec_loss_total / nb_tr_steps, epoch)
        self.writer.add_scalar('train/icl_loss', icl_loss_total / nb_tr_steps, epoch)
        
        self.logger.info(f"Epoch {epoch}: Loss={tr_loss/nb_tr_steps:.4f}, "
                        f"Rec={rec_loss_total/nb_tr_steps:.4f}, "
                        f"ICL={icl_loss_total/nb_tr_steps:.4f}")
        
        return np.mean(train_time)
    
    def _prepare_train_inputs(self, data):
        """准备训练输入"""
        # ICSRec 的 var_name: 
        # ["seq", "pos", "neg", "positions", "user_id", "seq_aug", "positions_aug", "target_item"]
        inputs = {}
        for i, part_data in enumerate(data):
            inputs[self.generator.train_dataset.var_name[i]] = part_data
        return inputs
    
    def eval(self, epoch=0, test=False):
        """
        评估模型
        
        复用父类的评估逻辑，但使用 ICSRec 模型的预测方法
        """
        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"] > 0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():
                inputs["item_indices"] = torch.cat(
                    [inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1
                )
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])
        
        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(
            pred_rank.detach().cpu().numpy(),
            seq_len.detach().cpu().numpy(),
            aug_len=self.args.aug_seq_len,
            args=self.args
        )
        res_pop_dict = metric_pop_report(
            pred_rank.detach().cpu().numpy(),
            self.item_pop,
            target_items.detach().cpu().numpy(),
            args=self.args
        )
        
        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}
        
        if test:
            record_csv(self.args, res_dict)
        
        return res_dict

