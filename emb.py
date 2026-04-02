"""
emb.py - SGF-RRA-MEB: Residual Reliability Alignment + Multi-center Emotion Ball
已经在model.py集成
基于 sgf_modified_framework_plan.md 框架设计文档实现。
在 SGF Backbone 的"前后"各插入一个轻量增强模块:
    - Module A: RRA (Residual Reliability Alignment) — 插在三模态编码之后、SGF 主干之前
    - Module B: MEB (Multi-center Emotion Ball)     — 插在 SGF 最终融合情绪表示 z 之后、分类器之前

总损失: L_total = L_SGF + λ_a * L_align + λ_b * L_meb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# Part I: Module A — Residual Reliability Alignment (RRA)
# ============================================================================
# 插入位置: Text/Audio/Video Encoder → [RRA] → Original SGF Backbone
#
# 核心设计:
#   1. 统一投影: 将三模态映射到同一维度 d=256
#   2. 共享分支 + 特异分支: [u_i^m; s_i^m] = MLP_m(p_i^m)
#   3. 一致性 + 可靠性打分: r_i^m = MLP_r([g_bar_i^m, q_i^m])
#   4. 残差式重标定: h_tilde_i^m = p_i^m + alpha_i^m * W_o^m * [u_i^m || s_i^m]
# ============================================================================


class UnifiedProjection(nn.Module):
    """
    Step 1: 将三模态分别投影到统一维度 d

    Args:
        modal_dim:      各模态原始特征维度 (dict: {'t': dim_t, 'a': dim_a, 'v': dim_v})
        unified_dim:    统一投影维度，默认为 256（见框架文档推荐值）
    """

    def __init__(self, modal_dim: dict, unified_dim: int = 256):
        super().__init__()
        self.unified_dim = unified_dim
        self.proj_t = nn.Linear(modal_dim['t'], unified_dim) if 't' in modal_dim else None
        self.proj_a = nn.Linear(modal_dim['a'], unified_dim) if 'a' in modal_dim else None
        self.proj_v = nn.Linear(modal_dim['v'], unified_dim) if 'v' in modal_dim else None

    def forward(self, h_t=None, h_a=None, h_v=None):
        """
        Args:
            h_t: Text 特征, shape [seq_len, batch, dim] 或 [N, dim]
            h_a: Audio 特征
            h_v: Video 特征
        Returns:
            dict: {'t': p_t, 'a': p_a, 'v': p_v}, 均为 [*, unified_dim]
        """
        out = {}
        if h_t is not None and self.proj_t is not None:
            out['t'] = self.proj_t(h_t)
        if h_a is not None and self.proj_a is not None:
            out['a'] = self.proj_a(h_a)
        if h_v is not None and self.proj_v is not None:
            out['v'] = self.proj_v(h_v)
        return out


class SharedSpecificBranch(nn.Module):
    """
    Step 2: 每个模态拆分成 shared branch (u) + specific branch (s)
        [u_i^m; s_i^m] = MLP_m(p_i^m)

    shared 分支负责跨模态对齐，specific 分支保留模态特异信息。

    Args:
        unified_dim: 统一投影维度
        branch_dim:   每个分支的维度 (shared + specific 各占一半)
    """

    def __init__(self, unified_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        half = unified_dim // 2
        self.shared_dim = half
        self.specific_dim = unified_dim - half

        # 每个模态各有一个 MLP
        self.mlp_t = nn.Linear(unified_dim, unified_dim)
        self.mlp_a = nn.Linear(unified_dim, unified_dim)
        self.mlp_v = nn.Linear(unified_dim, unified_dim)

        # 输出投影
        self.proj_shared_t = nn.Linear(unified_dim, self.shared_dim)
        self.proj_shared_a = nn.Linear(unified_dim, self.shared_dim)
        self.proj_shared_v = nn.Linear(unified_dim, self.shared_dim)

        self.proj_specific_t = nn.Linear(unified_dim, self.specific_dim)
        self.proj_specific_a = nn.Linear(unified_dim, self.specific_dim)
        self.proj_specific_v = nn.Linear(unified_dim, self.specific_dim)

    def forward(self, p_dict: dict):
        """
        Args:
            p_dict: {'t': p_t, 'a': p_a, 'v': p_v}, shape [*, unified_dim]
        Returns:
            u_dict: shared branch 输出
            s_dict: specific branch 输出
        """
        u_dict, s_dict = {}, {}

        for modal, p in p_dict.items():
            if modal == 't':
                h = F.relu(self.mlp_t(p))
                u_dict['t'] = self.proj_shared_t(h)
                s_dict['t'] = self.proj_specific_t(h)
            elif modal == 'a':
                h = F.relu(self.mlp_a(p))
                u_dict['a'] = self.proj_shared_a(h)
                s_dict['a'] = self.proj_specific_a(h)
            elif modal == 'v':
                h = F.relu(self.mlp_v(p))
                u_dict['v'] = self.proj_shared_v(h)
                s_dict['v'] = self.proj_specific_v(h)

        return u_dict, s_dict


class ReliabilityScorer(nn.Module):
    def __init__(self, branch_dim: int = 128, quality_dim: int = 1):
        super().__init__()
        # input = [g_bar, q] -> g_bar 是标量(1维)，q 是标量(1维)，总维度为 2
        in_dim = 1 + quality_dim

        self.scorer_t = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.scorer_a = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.scorer_v = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _compute_quality(self, u: torch.Tensor) -> torch.Tensor:
        """
        质量指标 q_i^m: 使用特征 L2 norm 作为简单置信度

        Args:
            u: shared branch 输出, shape [N, branch_dim]
        Returns:
            q: 质量分数, shape [N, 1]
        """
        return u.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _compute_cross_similarity(self, u_t: torch.Tensor, u_a: torch.Tensor, u_v: torch.Tensor):
        """
        计算模态对之间的 cosine 相似度

        Returns:
            g_ta, g_tv, g_av: 每对的 cosine 相似度
            g_bar_t, g_bar_a, g_bar_v: 每个模态与另外两模态的平均相似度
        """
        # 归一化
        u_t_norm = F.normalize(u_t, dim=-1)
        u_a_norm = F.normalize(u_a, dim=-1)
        u_v_norm = F.normalize(u_v, dim=-1)

        g_ta = (u_t_norm * u_a_norm).sum(dim=-1, keepdim=True)
        g_tv = (u_t_norm * u_v_norm).sum(dim=-1, keepdim=True)
        g_av = (u_a_norm * u_v_norm).sum(dim=-1, keepdim=True)

        g_bar_t = (g_ta + g_tv) / 2.0
        g_bar_a = (g_ta + g_av) / 2.0
        g_bar_v = (g_tv + g_av) / 2.0

        return g_bar_t, g_bar_a, g_bar_v

    def forward(self, u_dict: dict, s_dict: dict = None):
        device = next(iter(u_dict.values())).device
        N = next(iter(u_dict.values())).size(0)

        # 动态收集存在的模态
        alpha_list = []
        r_dict = {}

        for m in ['t', 'a', 'v']:
            u = u_dict.get(m)
            if u is not None:
                q_m = self._compute_quality(u)

                # 计算与其余存在的模态的平均相似度
                g_bar_m = torch.zeros_like(q_m)
                count = 0
                u_norm = F.normalize(u, dim=-1)
                for other_m in ['t', 'a', 'v']:
                    if other_m != m and other_m in u_dict:
                        other_u_norm = F.normalize(u_dict[other_m], dim=-1)
                        sim = (u_norm * other_u_norm).sum(dim=-1, keepdim=True)
                        g_bar_m += sim
                        count += 1
                if count > 0:
                    g_bar_m = g_bar_m / count

                scorer = getattr(self, f'scorer_{m}')
                r_m = scorer(torch.cat([g_bar_m, q_m], dim=-1))
                alpha_m = torch.sigmoid(r_m)

                alpha_list.append(alpha_m)
                r_dict[m] = r_m
            else:
                # 填充 dummy 占位符以保持维度对齐（如果下游需要固定的特征拼接）
                alpha_list.append(torch.zeros(N, 1, device=device))

        alpha = torch.cat(alpha_list, dim=-1)  # 始终保证返回 [N, 3] 结构供下游索引
        return alpha, r_dict


class ResidualReliabilityAlignment(nn.Module):
    def __init__(self, modal_dim: dict, unified_dim: int = 256, dropout: float = 0.3,
                 use_align_loss: bool = True):
        super().__init__()
        self.unified_dim = unified_dim
        self.use_align_loss = use_align_loss

        self.projection = UnifiedProjection(modal_dim, unified_dim)
        self.branch = SharedSpecificBranch(unified_dim)
        self.scorer = ReliabilityScorer(branch_dim=unified_dim // 2)

        # 【核心修改 1】：输出层必须映射回各个模态的“原始维度”
        branch_out_dim = unified_dim // 2 + (unified_dim - unified_dim // 2)
        self.output_proj_t = nn.Linear(branch_out_dim, modal_dim['t']) if 't' in modal_dim else None
        self.output_proj_a = nn.Linear(branch_out_dim, modal_dim['a']) if 'a' in modal_dim else None
        self.output_proj_v = nn.Linear(branch_out_dim, modal_dim['v']) if 'v' in modal_dim else None

        self.dropout_t = nn.Dropout(0.5)
        self.dropout_a = nn.Dropout(0.1)
        self.dropout_v = nn.Dropout(0.1)

        # 【核心修改 2】：为脆弱模态引入可学习的残差缩放因子 (零初始化)
        # 保证初始阶段网络严格等价于 Baseline，稳定后再缓慢学习残差
        # self.gamma_t = nn.Parameter(torch.zeros(1))
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_v = nn.Parameter(torch.zeros(1))

    def forward(self, h_t=None, h_a=None, h_v=None, return_align_loss: bool = False):
        p_dict = self.projection(h_t=h_t, h_a=h_a, h_v=h_v)
        u_dict, s_dict = self.branch(p_dict)
        alpha, r_dict = self.scorer(u_dict, s_dict)

        h_tilde_dict = {}
        modalities = list(p_dict.keys())
        orig_dict = {'t': h_t, 'a': h_a, 'v': h_v}

        for m in modalities:
            orig_h = orig_dict[m]
            u = u_dict[m]
            s = s_dict[m]
            us_cat = torch.cat([u, s], dim=-1)
            idx = {'t': 0, 'a': 1, 'v': 2}[m]
            alpha_m = alpha[:, idx:idx + 1]

            proj = getattr(self, f'output_proj_{m}')
            us_proj = F.relu(proj(us_cat))

            if m == 't':
                orig_dropped = self.dropout_t(orig_h)
                us_dropped = self.dropout_t(us_proj)
                h_tilde = alpha_m * orig_dropped + (1.0 - alpha_m) * us_dropped
            elif m == 'a':
                h_tilde = orig_h + self.gamma_a * alpha_m * self.dropout_a(us_proj)
            elif m == 'v':
                h_tilde = orig_h + self.gamma_v * alpha_m * self.dropout_v(us_proj)

            h_tilde_dict[m] = h_tilde

        # 传入 u_dict 和 s_dict 计算新损失
        L_align = self._compute_rra_loss(u_dict, s_dict) if self.use_align_loss else torch.tensor(0.0, device=next(iter(p_dict.values())).device)

        if return_align_loss:
            return h_tilde_dict, L_align, alpha  # 新增返回 alpha
        return h_tilde_dict, alpha

    def _compute_rra_loss(self, u_dict: dict, s_dict: dict):
        """
        最新模态对齐改进：非对称 InfoNCE 软对齐 + 共享/特异分支正交解耦
        """
        device = next(iter(u_dict.values())).device
        N = next(iter(u_dict.values())).size(0)

        # 1. L_ortho: 正交约束，强制 shared(u) 和 specific(s) 捕获不同维度的信息
        L_ortho = torch.tensor(0.0, device=device)
        for m in u_dict.keys():
            u = F.normalize(u_dict[m], dim=-1)
            s = F.normalize(s_dict[m], dim=-1)
            # 归一化后内积的平方作为正交惩罚
            L_ortho += (u * s).sum(dim=-1).pow(2).mean()

        # 2. L_nce: 跨模态对比对齐 (Asymmetric InfoNCE)
        L_nce = torch.tensor(0.0, device=device)
        tau = 0.1  # 对比温度
        labels = torch.arange(N, device=device)

        # 确立文本 't' 为绝对锚点 (Anchor)
        if 't' in u_dict:
            # 【关键】：detach() 彻底阻断弱模态对强模态的反向污染
            u_t_anchor = F.normalize(u_dict['t'].detach(), dim=-1)

            count = 0
            for m in ['a', 'v']:
                if m in u_dict:
                    u_m = F.normalize(u_dict[m], dim=-1)
                    # 计算单向相似度矩阵 [N, N]
                    logits = torch.matmul(u_m, u_t_anchor.t()) / tau
                    L_nce += F.cross_entropy(logits, labels)
                    count += 1

            if count > 0:
                L_nce = L_nce / count
        else:
            # 降级方案（若无文本）：双向 InfoNCE
            m1, m2 = list(u_dict.keys())[0], list(u_dict.keys())[1]
            u1, u2 = F.normalize(u_dict[m1], dim=-1), F.normalize(u_dict[m2], dim=-1)
            logits = torch.matmul(u1, u2.t()) / tau
            L_nce = F.cross_entropy(logits, labels)

        # 融合损失：正交约束权重大，对比损失权重小
        return L_nce * 0.05 + L_ortho * 1.0


# ============================================================================
# Part II: Module B — Multi-center Emotion Ball (MEB)
# ============================================================================
# 插入位置: Fused Emotion Representation z → [MEB] → Classifier
#
# 核心设计:
#   - 每个情绪类别 k 有 K_k 个球 (K_k >= 1，默认 K_k=2)
#   - 每球定义为 (c_{k,j}, R_{k,j}): 球心 + 半径
#   - 三项损失: L_intra (类内紧凑) + L_overlap (类间防重叠) + L_div (多球防塌缩)
#
# 低频类别可退化为 K_k=1 (单球)，默认每类 K_k=2。
# ============================================================================


class AngularMultiCenterEmotionBall(nn.Module):
    """
    基于单位超球面 (Unit Hypersphere) 和角距离 (Angular Distance) 的 MEB 模块。
    彻底解决高维欧氏空间距离爆炸问题，并支持 RRA 传导的置信度加权。
    """

    def __init__(self, z_dim: int, n_classes: int, K_per_class: int = 2,
                 tau_b: float = 0.1, dropout: float = 0.3):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.K = K_per_class
        self.tau_b = tau_b  # 温度系数需要调小，因为 Cosine 范围在 [-1, 1]

        # 可学习的球心参数: [n_classes, K, z_dim]
        self.ball_centers = nn.Parameter(torch.randn(n_classes, K_per_class, z_dim))

        # 初始角度半径 R (约束在 0 到 1 之间，代表 1-cos(theta) 的容忍度)
        self.ball_radii = nn.Parameter(torch.ones(n_classes, K_per_class) * 0.2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('ema_radii', torch.ones(n_classes, K_per_class) * 0.2)
        self.ema_momentum = 0.99
        self._is_initialized = False

    def _init_ball_centers_kmeans(self, z_features: torch.Tensor, labels: torch.Tensor):
        z_norm = F.normalize(z_features, dim=-1)
        with torch.no_grad():
            for k in range(self.n_classes):
                mask = (labels == k)
                if mask.sum() < 1: continue
                feats_k = z_norm[mask]
                if feats_k.size(0) <= self.K:
                    self.ball_centers.data[k, :1] = feats_k.mean(dim=0, keepdim=True)
                else:
                    idx = torch.randperm(feats_k.size(0), device=feats_k.device)[:self.K]
                    self.ball_centers.data[k] = feats_k[idx]

    def forward(self, z: torch.Tensor, labels: torch.Tensor = None,
                sample_rel: torch.Tensor = None, update_radii: bool = True):

        z_dropped = self.dropout(z)

        L_meb_dict = {'total': torch.tensor(0.0, device=z.device)}
        if labels is not None:
            L_meb_dict = self._compute_angular_meb_loss(z_dropped, labels, sample_rel, update_radii)

        # 保持前向特征原样输出，MEB 仅提供 Loss 梯度约束 Backbone
        return z, L_meb_dict

    def _compute_angular_meb_loss(self, z: torch.Tensor, labels: torch.Tensor,
                                  sample_rel: torch.Tensor, update_radii: bool):
        device = z.device
        batch_size = z.size(0)

        # 【核心 1】：投影到单位超球面
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(self.ball_centers, dim=-1)  # [n_classes, K, dim]

        # 确保半径物理意义合法 (角距离上限设为 1.0)
        radii = torch.clamp(self.ball_radii.abs(), min=0.05, max=1.0)

        # 如果没有传入置信度，默认全为 1
        if sample_rel is None:
            sample_rel = torch.ones(batch_size, 1, device=device)

        # ---- 1) 计算所有样本到自身类别中心簇的相似度与软分配权重 ----
        L_intra = torch.tensor(0.0, device=device)
        valid_count = 0

        for i in range(batch_size):
            k = labels[i].item()
            z_i = z_norm[i:i + 1]  # [1, dim]
            c_k = c_norm[k]  # [K, dim]

            # 余弦相似度 [1, K]
            sim_ik = torch.matmul(z_i, c_k.transpose(0, 1))
            # 角距离 D = 1 - cos(theta)
            dist_ik = 1.0 - sim_ik

            # 基于相似度计算软分配权重 softmax(sim / tau)
            q_ik = F.softmax(sim_ik / self.tau_b, dim=-1).detach()  # [1, K]

            # 加权角距离与加权半径
            dist_w = (q_ik * dist_ik).sum()
            r_w = (q_ik * radii[k:k + 1]).sum()

            # 【核心 2】：引入 RRA 的置信度 sample_rel 调控 Loss 严苛度
            # 高置信度样本受到严格约束，低置信度样本 Loss 被衰减
            loss_i = sample_rel[i, 0] * F.relu(dist_w - r_w)
            L_intra += loss_i
            valid_count += 1

            # 动态更新 EMA 半径 (简化逻辑：向当前 batch 的加权距离滑动)
            if update_radii and self.training:
                with torch.no_grad():
                    self.ema_radii[k] = self.ema_momentum * self.ema_radii[k] + \
                                        (1 - self.ema_momentum) * dist_ik.squeeze(0)

        if valid_count > 0:
            L_intra = L_intra / valid_count

        if update_radii and self.training:
            self.ball_radii.data = self.ema_radii.data.clone()

        # ---- 2) L_overlap: 类间防重叠 (约束不同类别的球心) ----
        # 要求不同类的球心余弦相似度不能大于 margin_ov (例如 0.3)
        margin_ov = 0.3
        c_flat = c_norm.view(self.n_classes * self.K, -1)
        # 计算所有球心两两之间的余弦相似度矩阵 [n_classes*K, n_classes*K]
        sim_matrix = torch.matmul(c_flat, c_flat.transpose(0, 1))

        # 构造掩码，屏蔽同类球心之间的比较
        mask = torch.ones_like(sim_matrix)
        for k in range(self.n_classes):
            mask[k * self.K: (k + 1) * self.K, k * self.K: (k + 1) * self.K] = 0.0

        # 仅惩罚相似度大于 margin_ov 的跨类球心
        overlap_penalties = F.relu(sim_matrix - margin_ov) * mask
        L_overlap = overlap_penalties.sum() / (mask.sum() + 1e-6)

        # ---- 3) L_div: 类内多球防塌缩 ----
        # 同一类内的 K 个球心不能靠得太近，要求相似度小于 margin_div (例如 0.8)
        margin_div = 0.8
        L_div = torch.tensor(0.0, device=device)
        if self.K > 1:
            div_count = 0
            for k in range(self.n_classes):
                for p in range(self.K):
                    for q in range(p + 1, self.K):
                        sim_pq = torch.dot(c_norm[k, p], c_norm[k, q])
                        L_div += F.relu(sim_pq - margin_div)
                        div_count += 1
            L_div = L_div / div_count

        # 总损失融合
        L_total = 1.0 * L_intra + 0.5 * L_overlap + 0.5 * L_div

        return {
            'total': L_total,
            'intra': L_intra,
            'overlap': L_overlap,
            'div': L_div
        }

    def _update_ema_radii(self, z_ball: torch.Tensor, labels: torch.Tensor,
                          q_matrix: torch.Tensor):
        batch_size = z_ball.size(0)
        new_radii = self.ball_radii.data.clone()

        for k in range(self.n_classes):
            mask_k = (labels == k)
            if mask_k.sum() < 1:
                continue

            z_k = z_ball[mask_k]
            q_k = q_matrix[mask_k, k * self.K:(k + 1) * self.K]
            sum_q = q_k.sum(dim=0) + 1e-6

            centers_k = self.ball_centers[k]
            dist_sq = torch.cdist(z_k, centers_k, p=2).pow(2)

            weighted_dist_sq = (q_k * dist_sq).sum(dim=0)
            batch_radii = (weighted_dist_sq / sum_q).sqrt()

            new_radii[k] = batch_radii

        # 加入严格的上下界限制，防止 EMA 半径崩溃
        new_radii = torch.clamp(new_radii, min=0.1, max=5.0)

        self.ball_radii.data = (self.ema_momentum * self.ball_radii.data +
                                (1 - self.ema_momentum) * new_radii)

    def get_ball_params(self):
        """返回当前球心与半径，供外部分析/可视化使用"""
        return {
            'centers': self.ball_centers.data,
            'radii': self.ball_radii.data.abs() + 1e-6
        }




