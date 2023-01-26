import pickle
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter

from .debised_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy


class Debiased:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Flexmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FlexMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if args.dataset.upper() == 'IMAGENET':
            p_target = None
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                p_target = p_target.cuda(args.gpu)
            # print('p_target:', p_target)

        p_model = None

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)

        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

        print('dataloader length',len(self.loader_dict['train_ulb']),len(self.loader_dict['train_lb']))

        ulb_iter = iter(self.loader_dict['train_ulb'])
        for (_, images_x, targets_x) in (self.loader_dict['train_lb']):
            try:
                images_u  = next(ulb_iter)
            except StopIteration:
                break

        # for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
        #                                                           self.loader_dict['train_ulb']):
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            if False or args.multiviews:
                images_u_w, images_u_w2, images_u_s, images_u_s2 = images_u
                images_u_w = torch.cat([images_u_w.cuda(args.gpu, non_blocking=True), images_u_w2.cuda(args.gpu, non_blocking=True)], dim=0)
                images_u_s = torch.cat([images_u_s.cuda(args.gpu, non_blocking=True), images_u_s2.cuda(args.gpu, non_blocking=True)], dim=0)
            else:
                images_u_w, images_u_s = images_u

            # measure data loading time
            # data_time.update(time.time() - end)

            if args.gpu is not None:
                images_x = images_x.cuda(args.gpu, non_blocking=True)
                images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
                images_u_s = images_u_s.cuda(args.gpu, non_blocking=True)

            targets_x = targets_x.cuda(args.gpu, non_blocking=True)
            targets_u = targets_u.cuda(args.gpu, non_blocking=True)

            # warmup learning rate
            # if epoch < args.warmup_epoch:
            #     warmup_step = args.warmup_epoch * len(train_loader_u)
            #     curr_step = epoch * len(train_loader_u) + i + 1
            #     lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
            # curr_lr.update(optimizer.param_groups[0]['lr'])

            # model forward
            batch_size_x = images_x.shape[0]
            if not args.eman:
                inputs = torch.cat((images_x, images_u_w, images_u_s))
                logits = self.main(inputs)
                logits_x = logits[:batch_size_x]
                logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)
            else:
                inputs = torch.cat((images_x, images_u_s))
                logits = self.model(inputs)
                logits_x = logits[:batch_size_x]
                logits_u_s = logits[batch_size_x:]
                with torch.no_grad():  # no gradient to ema model
                    logits_u_w = self.ema_model(images_u_w)
            
            # if args.multiviews:
            #     logits_u_w1, logits_u_w2 = logits_u_w.chunk(2)
            #     logits_u_s1, logits_u_s2 = logits_u_s.chunk(2)
            #     logits_u_w = (logits_u_w1 + logits_u_w2) / 2
            #     logits_u_s = (logits_u_s1 + logits_u_s2) / 2
            
            # producing debiased pseudo-labels
            pseudo_label = causal_inference(logits_u_w.detach(), qhat, exp_idx=0, tau=args.tau)
            # if args.multiviews:
            #     pseudo_label1 = causal_inference(logits_u_w1.detach(), qhat, exp_idx=0, tau=args.tau)
            #     max_probs1, pseudo_targets_u1 = torch.max(pseudo_label1, dim=-1)
            #     mask1 = max_probs1.ge(args.threshold).float()
            #     pseudo_label2 = causal_inference(logits_u_w2.detach(), qhat, exp_idx=0, tau=args.tau)
            #     max_probs2, pseudo_targets_u2 = torch.max(pseudo_label2, dim=-1)
            #     mask2 = max_probs2.ge(args.threshold).float()

            max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            # update qhat
            qhat_mask = mask if args.masked_qhat else None
            qhat = update_qhat(torch.softmax(logits_u_w.detach(), dim=-1), qhat, momentum=args.qhat_m, qhat_mask=qhat_mask)

            # adaptive marginal loss
            delta_logits = torch.log(qhat)
            # if args.multiviews:
            #     logits_u_s1 = logits_u_s1 + args.tau*delta_logits
            #     logits_u_s2 = logits_u_s2 + args.tau*delta_logits
            # else:
            logits_u_s = logits_u_s + args.tau*delta_logits

            # loss for labeled samples
            loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # loss for unlabeled samples
            per_cls_weights = None
            # if args.multiviews:
            #     loss_u = 0
            #     pseudo_targets_list = [pseudo_targets_u, pseudo_targets_u1, pseudo_targets_u2]
            #     masks_list = [mask, mask1, mask2]
            #     logits_u_list = [logits_u_s1, logits_u_s2]
            #     for idx, targets_u in enumerate(pseudo_targets_list):
            #         for logits_u in logits_u_list:
            #             loss_u += (F.cross_entropy(logits_u, targets_u, reduction='none', weight=per_cls_weights) * masks_list[idx]).mean()
            #     loss_u = loss_u/(len(pseudo_targets_list)*len(logits_u_list))
            # else:
            loss_u = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=per_cls_weights) * mask).mean()
            
            # if args.use_clip:
            #     # add clip's predictions
            #     indexs_u = indexs_u.cuda(args.gpu, non_blocking=True)
            #     targets_u_clip = clip_preds_list[indexs_u][:,0].view(-1)
            #     targets_u_clip = targets_u_clip.cuda(args.gpu, non_blocking=True)
            #     # add mask for clip with thresholding
            #     probs_list = clip_probs_list[indexs_u].cuda(args.gpu, non_blocking=True)
            #     max_probs, _ = torch.max(probs_list, dim=-1)
            #     mask_clip = max_probs.ge(0.4).float()
            #     # apply clip predictions to low-confidence predictions
            #     mask_delta = (mask_clip - mask - mask1 - mask2).ge(0.01).float()
            #     loss_u_clip = [F.cross_entropy(logits_u, targets_u_clip, reduction='none', weight=per_cls_weights) * mask_delta for logits_u in logits_u_list]
            #     loss_u = (torch.stack(loss_u_clip, dim=0).mean() + loss_u) / 2.0

            # CLD loss for unlabled samples (optional)
            if args.CLDLoss:
                prob_s = torch.softmax(logits_u_s, dim=-1)
                prob_w = torch.softmax(logits_u_w.detach(), dim=-1)
                loss_cld = CLDLoss(prob_s, prob_w, mask=None, weights=per_cls_weights)
            else:
                loss_cld = 0

            # total loss
            total_loss = loss_x + args.lambda_u * loss_u + args.lambda_cld * loss_cld

            # total_loss = sup_loss + self.lambda_u * unsup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = loss_x.detach()
            tb_dict['train/unsup_loss'] = loss_u.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            logger.inline(self.it, tb_dict)
            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            # print(x,y)
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        # save_filename = os.path.join(save_path, save_name)
        # # copy EMA parameters to ema_model for saving with model as temp
        # self.model.eval()
        # self.ema.apply_shadow()
        # ema_model = self.model.state_dict()
        # self.ema.restore()
        # self.model.train()

        # torch.save({'model': self.model.state_dict(),
        #             'optimizer': self.optimizer.state_dict(),
        #             'scheduler': self.scheduler.state_dict(),
        #             'it': self.it,
        #             'ema_model': ema_model},
        #            save_filename)

        # self.print_fn(f"model saved: {save_filename}")
        pass

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
def causal_inference(current_logit, qhat, exp_idx, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob

def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat
def CLDLoss(prob_s, prob_w, mask=None, weights=None):
    cl_w, c_w = get_centroids(prob_w)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * (1 - mask)).mean()
    return loss

if __name__ == "__main__":
    pass
