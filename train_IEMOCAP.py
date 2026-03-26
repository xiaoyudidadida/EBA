import os
import math
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss, SpeakerDetectionModel_MELD, \
    SpeakerDetectionModel_IEMOCAP, GraphNN
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
import faiss
from itertools import chain
import warnings

warnings.filterwarnings('ignore')
import dill

seed = 1746  # We use seed = 1746 on IEMOCAP and seed = 67137 on MELD
meld_speakers = ['Rachel', 'Joey', 'Ross', 'Monica', 'Chandler', 'Phoebe']
meld_labels = ['neutral', 'surprise', 'fear', 'sad', 'joy', 'disgust', 'angry']
iemocap_speakers = ['Ses01_M', 'Ses01_F', 'Ses02_M', 'Ses02_F', 'Ses03_M', 'Ses03_F', 'Ses04_M', 'Ses04_F']
iemocap_labels = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
dataset_type = 'MELD'


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
	losses, preds, labels, masks = [], [], [], []
	alphas, alphas_f, alphas_b, vids = [], [], [], []
	max_sequence_len = []

	assert not train or optimizer != None
	if train:
		model.train()
	else:
		model.eval()

	seed_everything()

	# 修复：使用 torch.set_grad_enabled 动态控制计算图
	with torch.set_grad_enabled(train):
		for data in dataloader:
			if train:
				optimizer.zero_grad()

			textf, visuf, acouf, qmask, umask, label, speakers = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
			max_sequence_len.append(textf.size(0))

			log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
			lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
			labels_ = label.view(-1)
			loss = loss_function(lp_, labels_, umask)

			pred_ = torch.argmax(lp_, 1)
			preds.append(pred_.data.cpu().numpy())
			labels.append(labels_.data.cpu().numpy())
			masks.append(umask.view(-1).cpu().numpy())

			# 获取 item() 避免图累积
			losses.append(loss.item() * masks[-1].sum())

			if train:
				loss.backward()
				optimizer.step()
			else:
				# 评估时由于已在 no_grad 下，直接 append 也是安全的
				alphas += [a.cpu() for a in alpha]  # 修复：将注意力权重移至 CPU，防止 VRAM 堆积
				alphas_f += [a.cpu() for a in alpha_f]
				alphas_b += [a.cpu() for a in alpha_b]
				vids += data[-1]

		if preds != []:
			preds = np.concatenate(preds)
			labels = np.concatenate(labels)
			masks = np.concatenate(masks)
		else:
			return float('nan'), float('nan'), [], [], [], float('nan'), []

		avg_loss = round(np.sum(losses) / np.sum(masks), 4)
		avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
		avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

		return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


class Stance_loss(nn.Module):
    def __init__(self, temperature, contrast_mode='all',
                 base_temperature=0.07):
        super(Stance_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # 提取当前特征所在的计算设备，确保后续所有张量操作与该设备对齐
        device = features.device

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 直接在目标设备上初始化单位矩阵，避免 CPU 到 GPU 的内存拷贝
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # 使用 .to(device) 替代 .cuda() 保证设备无关性
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, contrast_count)

        # 直接在目标设备上生成序列，避免 .cuda() 产生的额外内存分配
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss



class Batched_Stance_loss(nn.Module):
    """
    专为消除 for 循环设计的批量并行对比损失。
    严格保证节点与节点之间的相似度计算绝对隔离，实现 100% 数学等价。
    """
    def __init__(self, temperature=0.07):
        super(Batched_Stance_loss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features 形状预期: [N, K, Dim]，例如 [有效节点数, 12, 256]
        # labels 形状预期: [K]，例如 [12]
        N, K, dim = features.shape
        device = features.device

        # 1. 构造局部的标签掩码 [K, K]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 2. 消除对角线自身对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(K, device=device).view(-1, 1),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        # 3. 将掩码扩展到整个 Batch [N, K, K]
        mask_pos = mask_pos.unsqueeze(0).expand(N, K, K)
        mask_neg = mask_neg.unsqueeze(0).expand(N, K, K)

        # 4. 核心：bmm 并行计算相似度，严格隔离 N 个节点
        # [N, K, Dim] x [N, Dim, K] -> [N, K, K]
        similarity = torch.exp(torch.bmm(features, features.transpose(1, 2)) / self.temperature)

        # 5. 计算正负样本
        pos = torch.sum(similarity * mask_pos, dim=2)  # [N, K]
        neg = torch.sum(similarity * mask_neg, dim=2)  # [N, K]

        # 6. 计算损失，加 1e-8 防止 log(0)
        loss_matrix = -torch.log(pos / (pos + neg + 1e-8) + 1e-8)  # [N, K]

        # 7. 还原原代码的计算尺度：单节点的均值，再对所有节点求和
        loss = torch.sum(torch.mean(loss_matrix, dim=1))

        return loss

def Entropy(x):
    x = torch.softmax(x, dim=0)
    return torch.dot(x, torch.log2(x) / math.log2(len(data_speakers))) * (-1)



target_criterion = Stance_loss(0.07)
batched_target_criterion = Batched_Stance_loss(0.07)

proto_k=2

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False,
							  dataset='IEMOCAP'):
	global target_criterion, ss_t
	losses, preds, labels = [], [], []
	tp1, tl1, tp2, tl2 = [], [], [], []
	scores, vids = [], []
	global gt1, gt2, gt3, gt4, ga, gv
	global textproto, audioproto, visproto
	probs = {}
	ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

	if cuda:
		ei, et, en = ei.cuda(), et.cuda(), en.cuda()

	assert not train or optimizer != None
	if train:
		model.train()
		model.dsu.training = True
		gt1.train()
		gt2.train()
		gt3.train()
		gt4.train()
		ga.train()
		gv.train()
	else:
		model.eval()
		model.dsu.training = False
		gt1.eval()
		gt2.eval()
		gt3.eval()
		gt4.eval()
		ga.eval()
		gv.eval()

	seed_everything()

	with torch.set_grad_enabled(train):
		for data in dataloader:
			if train:
				optimizer.zero_grad()

			if dataset_type == 'MELD':
				ttextf1, ttextf2, ttextf3, ttextf4, tvisuf, tacouf, qmask, _, _, _, umask, label, speakers = [d.cuda()
																											  for d in
																											  data[
																											  :-1]] if cuda else data[
																																 :-1]
			else:
				ttextf1, ttextf2, ttextf3, ttextf4, tvisuf, tacouf, qmask, _, umask, label, speakers = [d.cuda() for d
																										in data[
																										   :-1]] if cuda else data[
																															  :-1]
			vids = data[-1]

			lengths = umask.sum(dim=1).long().tolist()

			# 1. 明确获取 batch_size 和 seq_len
			batch_size = umask.size(0)
			seq_len = umask.size(1)

			# 2. 将原始输入 [seq_len, batch_size, dim] 转置为 [batch_size, seq_len, dim] 以匹配 umask
			ttextf1_b = ttextf1.transpose(0, 1)
			ttextf2_b = ttextf2.transpose(0, 1)
			ttextf3_b = ttextf3.transpose(0, 1)
			ttextf4_b = ttextf4.transpose(0, 1)
			tacouf_b = tacouf.transpose(0, 1)
			tvisuf_b = tvisuf.transpose(0, 1)

			# 3. 生成有效掩码 [batch_size, seq_len]
			valid_mask = (umask != 0)

			# 4. 提取有效的 speaker 标签
			s_t = speakers[valid_mask].clone()
			s_t[s_t == -1] = len(data_speakers)
			s_t = s_t.float().cuda() if cuda else s_t.float()

			# 5. 利用布尔索引提取有效特征，完全消除 Python for 循环和 torch.cat 的显存碎片
			flat_ttextf1 = ttextf1_b[valid_mask]
			flat_ttextf2 = ttextf2_b[valid_mask]
			flat_ttextf3 = ttextf3_b[valid_mask]
			flat_ttextf4 = ttextf4_b[valid_mask]
			flat_tacouf = tacouf_b[valid_mask]
			flat_tvisuf = tvisuf_b[valid_mask]

			# 6. 送入 GraphNN 计算 (全程在 GPU 上高并行运行)
			pt1, wt1 = gt1(flat_ttextf1, torch.cat(textproto[0], dim=0).cuda())
			pt2, wt2 = gt2(flat_ttextf2, torch.cat(textproto[1], dim=0).cuda())
			pt3, wt3 = gt3(flat_ttextf3, torch.cat(textproto[2], dim=0).cuda())
			pt4, wt4 = gt4(flat_ttextf4, torch.cat(textproto[3], dim=0).cuda())
			pa, wa = ga(flat_tacouf, torch.cat(audioproto, dim=0).cuda())
			pv, wv = gv(flat_tvisuf, torch.cat(visproto, dim=0).cuda())

			# 7. 预分配组合后的特征张量内存 [batch_size, seq_len, new_dim]
			textf1 = torch.zeros(batch_size, seq_len, ttextf1_b.size(-1) + pt1.size(-1), device=ttextf1_b.device)
			textf2 = torch.zeros(batch_size, seq_len, ttextf2_b.size(-1) + pt2.size(-1), device=ttextf2_b.device)
			textf3 = torch.zeros(batch_size, seq_len, ttextf3_b.size(-1) + pt3.size(-1), device=ttextf3_b.device)
			textf4 = torch.zeros(batch_size, seq_len, ttextf4_b.size(-1) + pt4.size(-1), device=ttextf4_b.device)
			acouf = torch.zeros(batch_size, seq_len, tacouf_b.size(-1) + pa.size(-1), device=tacouf_b.device)
			visuf = torch.zeros(batch_size, seq_len, tvisuf_b.size(-1) + pv.size(-1), device=tvisuf_b.device)

			# 8. 利用 GPU 并行掩码直接拼接赋值
			textf1[valid_mask] = torch.cat([flat_ttextf1, pt1[:, -1]], dim=-1)
			textf2[valid_mask] = torch.cat([flat_ttextf2, pt2[:, -1]], dim=-1)
			textf3[valid_mask] = torch.cat([flat_ttextf3, pt3[:, -1]], dim=-1)
			textf4[valid_mask] = torch.cat([flat_ttextf4, pt4[:, -1]], dim=-1)
			acouf[valid_mask] = torch.cat([flat_tacouf, pa[:, -1]], dim=-1)
			visuf[valid_mask] = torch.cat([flat_tvisuf, pv[:, -1]], dim=-1)

			# 9. 转置回 [seq_len, batch_size, dim] 以满足下游 Base Model 的需求
			textf1 = textf1.transpose(0, 1)
			textf2 = textf2.transpose(0, 1)
			textf3 = textf3.transpose(0, 1)
			textf4 = textf4.transpose(0, 1)
			visuf = visuf.transpose(0, 1)
			acouf = acouf.transpose(0, 1)
			# -------------------------------------------------------------

			if args.multi_modal:
				if args.mm_fusion_mthd == 'concat':
					if modals == 'avl':
						textf = torch.cat([acouf, visuf, textf1, textf2, textf3, textf4], dim=-1)
					elif modals == 'av':
						textf = torch.cat([acouf, visuf], dim=-1)
					elif modals == 'vl':
						textf = torch.cat([visuf, textf1, textf2, textf3, textf4], dim=-1)
					elif modals == 'al':
						textf = torch.cat([acouf, textf1, textf2, textf3, textf4], dim=-1)
					else:
						raise NotImplementedError
				elif args.mm_fusion_mthd == 'gated':
					textf = textf
			else:
				if modals == 'a':
					textf = acouf
				elif modals == 'v':
					textf = visuf
				elif modals == 'l':
					textf = textf
				else:
					raise NotImplementedError

			flat_label = label[valid_mask]

			if args.multi_modal and args.mm_fusion_mthd == 'gated':
				log_prob, e_i, e_n, e_t, e_l, extra_loss = model(textf, qmask, umask, lengths, acouf, visuf,
																 epoch=epoch, labels=flat_label)
			elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
				log_prob, e_i, e_n, e_t, e_l, extra_loss = model([textf1, textf2, textf3, textf4], qmask, umask,
																 lengths,
																 acouf, visuf, epoch=epoch, labels=flat_label)
			elif args.multi_modal and args.mm_fusion_mthd == 'concat_DHT':
				log_prob, e_i, e_n, e_t, e_l, extra_loss = model([textf1, textf2, textf3, textf4], qmask, umask,
																 lengths,
																 acouf, visuf, epoch=epoch, labels=flat_label)
			else:
				log_prob, e_i, e_n, e_t, e_l, extra_loss = model(textf, qmask, umask, lengths, epoch=epoch,
																 labels=flat_label)

			# 把 flat_label 覆盖给 label，供后面的 loss_function 和指标统计使用
			label = flat_label

		# 计算基础 Loss（含 RRA/MEB extra_loss）
			loss = loss_function(log_prob, label) + extra_loss + 0.01 * (
						target_criterion(wt1[:, -1:], s_t) + target_criterion(wt2[:, -1:], s_t) +
						target_criterion(wt3[:, -1:], s_t) + target_criterion(wt4[:, -1:], s_t) +
						target_criterion(wa[:, -1:], s_t) + target_criterion(wv[:, -1:], s_t))
			loss = loss + 0.01 * (
					target_criterion(pt1[:, -1:], s_t) + target_criterion(pt2[:, -1:], s_t) +
					target_criterion(pt3[:, -1:], s_t) + target_criterion(pt4[:, -1:], s_t) +
					target_criterion(pa[:, -1:], s_t) + target_criterion(pv[:, -1:], s_t))

				# ---------------- 修复：缩进正确，批量计算辅助 Loss ----------------
			num_labels = len(data_labels)
			# 生成单个节点内部的原型标签 [12]
			ss_t_base = torch.tensor([i for i in range(len(data_speakers)) for _ in range(num_labels)],
									 dtype=torch.float32, device=pt1.device)

			# 提取 [N, 12, Dim] 形状的原型特征
			pt1_proto = pt1[:, :-1, :]
			pt2_proto = pt2[:, :-1, :]
			pt3_proto = pt3[:, :-1, :]
			pt4_proto = pt4[:, :-1, :]
			pa_proto = pa[:, :-1, :]
			pv_proto = pv[:, :-1, :]

			# 一次性无循环计算所有节点的局部对比损失，严格等价且极速
			loss = loss + 0.8 * 0.0001 * (
					batched_target_criterion(pt1_proto, ss_t_base) +
					batched_target_criterion(pt2_proto, ss_t_base) +
					batched_target_criterion(pt3_proto, ss_t_base) +
					batched_target_criterion(pt4_proto, ss_t_base) +
					batched_target_criterion(pa_proto, ss_t_base) +
					batched_target_criterion(pv_proto, ss_t_base)
			)
			# -------------------------------------------------------------

			preds.append(torch.argmax(log_prob, 1).cpu().numpy())
			labels.append(label.cpu().numpy())

			# ---------------- 修复：缩进正确，切片化概率回收 ----------------
			log_prob_np = log_prob.data.cpu().numpy()
			now = 0
			for i in range(len(umask)):
				curr_length = lengths[i]
				probs[vids[i]] = log_prob_np[now: now + curr_length].tolist()
				now += curr_length
			# -------------------------------------------------------------

			losses.append(loss.item())
			if train:
				loss.backward()
				optimizer.step()

	if preds != []:
		preds = np.concatenate(preds)
		labels = np.concatenate(labels)
	else:
		return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [],{}

	vids += data[-1]
	ei = ei.data.cpu().numpy()
	et = et.data.cpu().numpy()
	en = en.data.cpu().numpy()
	el = np.array(el)
	labels = np.array(labels)
	preds = np.array(preds)
	vids = np.array(vids)

	avg_loss = round(np.sum(losses) / len(losses), 4)
	avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
	avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

	return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, probs


def k_cluster(x, d, k):
    x = x.cpu().numpy()
    kmeans = faiss.Kmeans(d, k, gpu=True)
    kmeans.train(x)
    D = kmeans.centroids
    return torch.from_numpy(D).cuda()


def collect_feature(dataloader, D_text, D_audio, D_visual):
	global proto_k, data_speakers
	textfeature = [[[[] for _ in range(len(data_labels))] for _ in range(len(data_speakers))] for _ in range(4)]
	audiofeature, visfeature = [[[] for _ in range(len(data_labels))] for _ in range(len(data_speakers))], [
		[[] for _ in range(len(data_labels))] for _ in range(len(data_speakers))]

	global textproto, audioproto, visproto
	textproto, audioproto, visproto = [[0 for _ in range(len(data_speakers))] for _ in range(4)], [0 for _ in range(
		len(data_speakers))], [0 for _ in range(len(data_speakers))]

	# 修改 1：引入 torch.no_grad()，阻断隐式计算图缓存
	with torch.no_grad():
		for data in dataloader:
			textf = [0 for _ in range(4)]
			if dataset_type == 'MELD':
				textf[0], textf[1], textf[2], textf[3], visuf, acouf, _, _, _, _, _, label, speakers = [d.cuda() for d
																										in
																										data[
																										:-1]] if cuda else data[
																														   :-1]
			else:
				textf[0], textf[1], textf[2], textf[3], visuf, acouf, _, _, _, label, speakers = [d.cuda() for d in
																								  data[
																								  :-1]] if cuda else data[
																													 :-1]
			ttextf = [[] for _ in range(4)]
			tvisuf, tacouf = [], []
			for i in range(4):
				textf[i] = textf[i].transpose(0, 1)
			visuf = visuf.transpose(0, 1)
			acouf = acouf.transpose(0, 1)

			for i in range(len(speakers)):
				for j in range(len(speakers[i])):
					if speakers[i][j] == -1:
						continue
					# 修改 2：使用 .cpu() 将大规模特征张量转移到系统内存
					for k in range(4):
						textfeature[k][speakers[i][j]][label[i][j]].append(textf[k][i][j].cpu())
						ttextf[k].append(textf[k][i][j].cpu())
					audiofeature[speakers[i][j]][label[i][j]].append(acouf[i][j].cpu())
					visfeature[speakers[i][j]][label[i][j]].append(visuf[i][j].cpu())
					tacouf.append(acouf[i][j].cpu())
					tvisuf.append(visuf[i][j].cpu())

	k = proto_k
	# 修改 3：在 CPU 端完成拼接与均值计算，最后仅将结果 Prototype 推送回 GPU
	for i in range(len(data_speakers)):
		for j in range(4):
			textproto[j][i] = torch.cat(
				[torch.mean(torch.stack(textfeature[j][i][l], dim=0), dim=0, keepdim=True) for l in
				 range(len(data_labels))], dim=0).cuda() if cuda else torch.cat(
				[torch.mean(torch.stack(textfeature[j][i][l], dim=0), dim=0, keepdim=True) for l in
				 range(len(data_labels))], dim=0)

		audioproto[i] = torch.cat(
			[torch.mean(torch.stack(audiofeature[i][l], dim=0), dim=0, keepdim=True) for l in range(len(data_labels))],
			dim=0).cuda() if cuda else torch.cat(
			[torch.mean(torch.stack(audiofeature[i][l], dim=0), dim=0, keepdim=True) for l in range(len(data_labels))],
			dim=0)

		visproto[i] = torch.cat(
			[torch.mean(torch.stack(visfeature[i][l], dim=0), dim=0, keepdim=True) for l in range(len(data_labels))],
			dim=0).cuda() if cuda else torch.cat(
			[torch.mean(torch.stack(visfeature[i][l], dim=0), dim=0, keepdim=True) for l in range(len(data_labels))],
			dim=0)


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False,
                        help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat',
                        help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--edge_ratio', type=float, default=0.01, help='edge_ratio')

    parser.add_argument('--num_convs', type=int, default=3, help='num_convs in EH')

    parser.add_argument('--opn', default='corr', help='option')

    parser.add_argument('--proto_k', type=int, default=2, help='num of prototypes')

    # ---- RRA / MEB 消融实验控制参数 ----
    parser.add_argument('--use_rra', action='store_true', default=False,
                        help='Enable Residual Reliability Alignment (RRA) module')
    parser.add_argument('--use_meb', action='store_true', default=False,
                        help='Enable Multi-center Emotion Ball (MEB) module')

    # ---- 随机种子参数 ----
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the global random seed (IEMOCAP default=1746, MELD default=67137). '
                             'If not provided, the hardcoded default seed is used.')

    args = parser.parse_args()
    today = datetime.datetime.now()

    # ---- 种子覆盖：如果传入了 --seed，则使用传入值 ----
    if args.seed is not None:
        seed = args.seed
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.Dataset
    else:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(
            args.Deep_GCN_nlayers) + '_' + args.Dataset

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'
    if args.use_rra:
        name_ = name_ + '_RRA'
    if args.use_meb:
        name_ = name_ + '_MEB'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = 342 if args.Dataset == 'IEMOCAP' else feat2dim['denseface']
    D_text = 1024  # feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512  # 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    global n_speakers
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1
    proto_k = args.proto_k
    global gt1, gt2, gt3, gt4, ga, gv
    gt1, gt2, gt3, gt4, ga, gv = GraphNN(D_text), GraphNN(D_text), GraphNN(D_text), GraphNN(D_text), GraphNN(
        D_audio), GraphNN(D_visual),

    global data_speakers, data_labels
    if args.Dataset == 'MELD':
        dataset_type, data_speakers, data_labels = 'MELD', meld_speakers, meld_labels
    else:
        dataset_type, data_speakers, data_labels = 'IEMOCAP', iemocap_speakers, iemocap_labels
    global ss_t
    ss_t = []
    for i in range(len(data_speakers)):
        for j in range(len(data_labels)):
            for j in range(proto_k):
                ss_t.append(i)
    ss_t = torch.Tensor(ss_t)
    if args.graph_model:
        seed_everything()

        model = Model(args.base_model,
                      D_m * 2, D_g, D_p, D_e, D_h, D_a, graph_h,
                      n_speakers=n_speakers,
                      max_seq_len=200,
                      window_past=args.windowp,
                      window_future=args.windowf,
                      n_classes=n_classes,
                      listener_state=args.active_listener,
                      context_attention=args.attention,
                      dropout=args.dropout,
                      nodal_attention=args.nodal_attention,
                      no_cuda=args.no_cuda,
                      graph_type=args.graph_type,
                      use_topic=args.use_topic,
                      alpha=args.alpha,
                      multiheads=args.multiheads,
                      graph_construct=args.graph_construct,
                      use_GCN=args.use_gcn,
                      use_residue=args.use_residue,
                      D_m_v=D_visual * 2,
                      D_m_a=D_audio * 2,
                      modals=args.modals,
                      att_type=args.mm_fusion_mthd,
                      av_using_lstm=args.av_using_lstm,
                      Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                      dataset=args.Dataset,
                      use_speaker=args.use_speaker,
                      use_modal=args.use_modal,
                      norm=args.norm,
                      edge_ratio=args.edge_ratio,
                      num_convs=args.num_convs,
                      opn=args.opn,
                      D_text=D_text * 2,
                      use_rra=args.use_rra,
                      use_meb=args.use_meb)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'
    # load checkpoint
    # with open('analysis/new_iemocap.pkl','rb') as f:
    # 	model,gt1,gt2,gt3,gt4,ga,gv=dill.load(f)

    if cuda:
        model.cuda()
        gt1.cuda()
        gt2.cuda()
        gt3.cuda()
        gt4.cuda()
        ga.cuda()
        gv.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                # loss_function = FocalLoss()
                loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    # ---- Optimizer（model.parameters() 已包含 rra/meb 子模块，无重复） ----
    optimizer = optim.Adam(
        params=chain(model.parameters(), gt1.parameters(), gt2.parameters(), gt3.parameters(), gt4.parameters(),
                     ga.parameters(), gv.parameters()),
        lr=args.lr, weight_decay=args.l2)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=8,pin_memory=True)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=8,pin_memory=True)
    else:
        print("There is no such dataset")

    collect_feature(train_loader, D_text, D_audio, D_visual)

    print('Emotion detection:')
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_prob = None
    test_label = False
    if test_label:
        state = torch.load('best_model_IEMOCAP/model.pth')
        model.load_state_dict(state['net'])
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                                           loss_function,
                                                                                                           test_loader,
                                                                                                           0, cuda,
                                                                                                           args.modals,
                                                                                                           dataset=args.Dataset)

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                                    loss_function,
                                                                                                    train_loader, e,
                                                                                                    cuda, args.modals,
                                                                                                    optimizer, True,
                                                                                                    dataset=args.Dataset)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _,_ = train_or_eval_graph_model(model, loss_function,
                                                                                                 valid_loader, e, cuda,
                                                                                                 args.modals,
                                                                                                 dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, test_probs = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore)


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e,
                                                                                  optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
                                                                                                                 loss_function,
                                                                                                                 test_loader,
                                                                                                                 e)
            all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss = test_loss

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore, best_label, best_pred = test_fscore, test_label, test_pred
            best_prob = test_probs

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore,
                   round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score:', max(all_fscore))
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = name_
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_ + 'record', False):
        record[key_ + 'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))