"""
emb.py - SGF-RRA-MEB: Residual Reliability Alignment + Multi-center Emotion Ball

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


# ============================================================================
# Part III: 对接接口 — RRA_MEB_Wrapper
# ============================================================================
# 将 RRA + SGF Backbone + MEB 串联成完整管线，
# 与 train_IEMOCAP.py / model.py 中的调用方式保持一致。
#
# 数据流:
#   textf / acouf / visuf
#       ↓  (RRA)
#   h_tilde_t / h_tilde_a / h_tilde_v
#       ↓  (SGF Backbone, 保持不变)
#   z (fused emotion representation)
#       ↓  (MEB)
#   z_enhanced
#       ↓
#   Classifier → log_prob
# ============================================================================


class RRA_MEB_Wrapper(nn.Module):
    """
    RRA + SGF Backbone + MEB 完整包装器

    该类用于在 model.py 的 Model 外部做一层包装，
    将 RRA 和 MEB 两个模块与原始 SGF Backbone 串联。
    在 train_IEMOCAP.py 中使用时，只需替换 Model 实例为 RRA_MEB_Wrapper。

    推荐初始化参数 (基于框架文档):
        RRA:
            - unified_dim = 256
            - lambda_a = 0.05 ~ 0.1
        MEB:
            - K_per_class = 2
            - tau_b = 0.5
            - margin_m = 0.5
            - eta = 1.0
            - lambda_b = 0.05 ~ 0.2
            - lambda_in/ov/div = 1.0/1.0/0.5

    总损失: L_total = L_SGF + λ_a * L_align + λ_b * L_meb['total']

    Args:
        sgf_model:      原始 SGF Model 实例 (来自 model.py)
        modal_dim:       三模态原始维度, dict: {'t': dim_t, 'a': dim_a, 'v': dim_v}
        z_dim:           SGF 融合表示 z 的维度 (需与 sgf_model 输出对齐)
        n_classes:       情绪类别数
        unified_dim:     RRA 统一投影维度，默认 256
        K_per_class:     每类球数，默认 2
        lambda_a:        对齐损失权重，默认 0.1
        lambda_b:        MEB 总损失权重，默认 0.1
        rra_dropout:     RRA Dropout，默认 0.3
        meb_dropout:     MEB Dropout，默认 0.3
        use_rra:         是否启用 RRA，默认 True
        use_meb:         是否启用 MEB，默认 True
    """

    def __init__(self, sgf_model: nn.Module, modal_dim: dict, z_dim: int,
                 n_classes: int, unified_dim: int = 256, K_per_class: int = 2,
                 lambda_a: float = 0.1, lambda_b: float = 0.1,
                 rra_dropout: float = 0.3, meb_dropout: float = 0.3,
                 use_rra: bool = True, use_meb: bool = True):
        super().__init__()
        self.sgf_model = sgf_model
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.use_rra = use_rra
        self.use_meb = use_meb

        # Module A: RRA
        self.rra = None
        if use_rra:
            self.rra = ResidualReliabilityAlignment(
                modal_dim=modal_dim,
                unified_dim=unified_dim,
                dropout=rra_dropout,
                use_align_loss=True
            )

        # Module B: MEB
        self.meb = None
        if use_meb:
            self.meb = MultiCenterEmotionBall(
                z_dim=z_dim,
                n_classes=n_classes,
                K_per_class=K_per_class,
                tau_b=0.5,
                margin_m=0.5,
                eta=1.0,
                dropout=meb_dropout,
                lambda_in=1.0,
                lambda_ov=1.0,
                lambda_div=0.5
            )

    def forward(self, textf, qmask, umask, lengths,
                textf2=None, textf3=None, textf4=None,
                U_a=None, U_v=None, epoch=None,
                labels=None):
        """
        与 model.py Model.forward() 签名保持兼容

        Args:
            textf/textf2/textf3/textf4: Text 特征 (RoBERTa 各层输出)
            qmask: 说话人掩码
            umask: 有效位置掩码
            lengths: 各序列长度
            U_a: Audio 特征 (等同于 acouf)
            U_v: Video 特征 (等同于 visuf)
            epoch: 当前 epoch (SGF 原有用法)
            labels: 情绪标签 (供 MEB 使用)
        Returns:
            log_prob: 分类 log 概率
            L_total:  总损失 (附加到 model 返回值的额外损失, 可在 train 中合并)
            (其余返回值与 Model 一致)
        """
        # ---- RRA: 三模态重标定 ----
        # textf 的形状: [seq_len, batch, dim]
        # 需要提取有效位置特征 (展平 batch + seq)
        if self.use_rra and self.rra is not None:
            seq_len, batch_size, _ = textf.shape
            # 转置并展平: [batch, seq_len, dim] -> [batch*seq_len, dim]
            text_flat = textf.transpose(0, 1).reshape(-1, textf.size(-1))
            audio_flat = U_a.transpose(0, 1).reshape(-1, U_a.size(-1)) if U_a is not None else None
            video_flat = U_v.transpose(0, 1).reshape(-1, U_v.size(-1)) if U_v is not None else None

            # RRA 重标定
            h_tilde_dict, L_align = self.rra(
                h_t=text_flat,
                h_a=audio_flat,
                h_v=video_flat,
                return_align_loss=True
            )

            # 将重标定结果 reshape 回 [batch, seq_len, dim]
            # 然后转置回 [seq_len, batch, dim] 以匹配下游 SGF Backbone
            h_t_tilde = h_tilde_dict['t'].view(batch_size, seq_len, -1).transpose(0, 1)
            h_a_tilde = h_tilde_dict['a'].view(batch_size, seq_len, -1).transpose(0, 1) if audio_flat is not None else None
            h_v_tilde = h_tilde_dict['v'].view(batch_size, seq_len, -1).transpose(0, 1) if video_flat is not None else None

            # 替换原始特征送入 SGF (这里需要修改 sgf_model 的输入处理方式，
            # 实际使用时建议在 model.py Model 内部直接集成 RRA，
            # 只需在此传入原始特征，由 Model 内部调用 self.rra 处理)
            textf_rra = textf   # 默认走原始流程，内部集成 RRA 时使用
            acouf_rra = U_a
            visuf_rra = U_v
        else:
            L_align = torch.tensor(0.0, device=textf.device)
            textf_rra, acouf_rra, visuf_rra = textf, U_a, U_v

        # ---- SGF Backbone (保持原样) ----
        # 注意: 实际集成时，SGF Backbone 的输入应在 model.py 内部通过 RRA 处理
        # 下面调用 sgf_model 的逻辑与 train_IEMOCAP.py 保持一致
        log_prob, e_i, e_n, e_t, e_l = self.sgf_model(
            textf_rra, qmask, umask, lengths,
            U_a=acouf_rra, U_v=visuf_rra, epoch=epoch
        )

        # ---- MEB: z 空间球约束 ----
        L_meb = torch.tensor(0.0, device=emotions_feat.device)
        if self.use_meb and self.meb is not None and labels is not None:
            # 初始化检查保持不变...
            emotions_feat_enhanced, L_meb_dict = self.meb(
                emotions_feat, labels=labels, update_radii=self.training
            )
            L_meb = L_meb_dict['total']
            emotions_feat = emotions_feat_enhanced

        # 【核心修改 3】：MEB 损失动态预热 (Warm-up)
        # 假设总 epoch 为 80，前 20 轮线性增加权重，避免早期梯度混乱
        current_epoch = epoch if epoch is not None else 0
        warmup_factor = min(1.0, current_epoch / 20.0)

        # 降低基础 MEB 权重至 0.05
        extra_loss = 0.0 * L_align + (0.1 * warmup_factor) * L_meb


# ============================================================================
# Part IV: 独立损失函数 (供 train_IEMOCAP.py 调用)
# ============================================================================


class RRALoss(nn.Module):
    """
    独立的 RRA 对齐损失

    单独使用时，直接在 train_IEMOCAP.py 的 train_or_eval_graph_model 中:
        L_align = rra_loss(u_t, u_a, u_v)
    """

    def __init__(self):
        super().__init__()

    def forward(self, u_t: torch.Tensor, u_a: torch.Tensor, u_v: torch.Tensor):
        """
        Args:
            u_t/u_a/u_v: 各模态 shared branch 输出, shape [N, dim]
        Returns:
            L_align: 标量损失
        """
        u_t_norm = F.normalize(u_t, dim=-1)
        u_a_norm = F.normalize(u_a, dim=-1)
        u_v_norm = F.normalize(u_v, dim=-1)

        sim_ta = (u_t_norm * u_a_norm).sum(dim=-1)
        sim_tv = (u_t_norm * u_v_norm).sum(dim=-1)
        sim_av = (u_a_norm * u_v_norm).sum(dim=-1)

        dist_ta = (1 - sim_ta).clamp(min=0)
        dist_tv = (1 - sim_tv).clamp(min=0)
        dist_av = (1 - sim_av).clamp(min=0)

        denom = sim_ta.exp() + sim_tv.exp() + sim_av.exp() + 1e-8
        omega_ta = sim_ta.exp() / denom
        omega_tv = sim_tv.exp() / denom
        omega_av = sim_av.exp() / denom

        return (omega_ta * dist_ta + omega_tv * dist_tv + omega_av * dist_av).mean()


class MEBLoss(nn.Module):
    """
    独立的 MEB 损失

    单独使用时，直接在 train_or_eval_graph_model 中:
        L_meb = meb_loss(z_ball, labels, ball_centers, ball_radii)
    """

    def __init__(self, n_classes: int, K_per_class: int = 2, tau_b: float = 0.5,
                 margin_m: float = 0.5, eta: float = 1.0,
                 lambda_in: float = 1.0, lambda_ov: float = 1.0,
                 lambda_div: float = 0.5, device='cuda'):
        super().__init__()
        self.n_classes = n_classes
        self.K = K_per_class
        self.tau_b = tau_b
        self.margin_m = margin_m
        self.eta = eta
        self.lambda_in = lambda_in
        self.lambda_ov = lambda_ov
        self.lambda_div = lambda_div
        self.device = device

        # 球心与半径作为 Buffer (非 Parameter, 不参与梯度更新)
        # 若需可学习，改为 nn.Parameter
        self.register_buffer('ball_centers',
                              torch.randn(n_classes, K_per_class, 256) * 0.1)
        self.register_buffer('ball_radii',
                              torch.ones(n_classes, K_per_class) * 0.5)

    def forward(self, z: torch.Tensor, labels: torch.Tensor,
                ball_centers: torch.Tensor = None,
                ball_radii: torch.Tensor = None):
        """
        Args:
            z:             样本表示, shape [N, dim]
            labels:        标签, shape [N]
            ball_centers: 球心 (可选，默认用 self.ball_centers)
            ball_radii:   球半径 (可选，默认用 self.ball_radii)
        Returns:
            L_meb: dict {'total', 'intra', 'overlap', 'div'}
        """
        if ball_centers is None:
            ball_centers = self.ball_centers
        if ball_radii is None:
            ball_radii = self.ball_radii

        batch_size = z.size(0)
        radii = ball_radii.abs() + 1e-6
        device = z.device

        # ---- L_intra ----
        L_intra = torch.tensor(0.0, device=device)
        count = 0
        for i in range(batch_size):
            k = labels[i].item()
            z_i = z[i:i + 1]
            c_k = ball_centers[k]   # [K, dim]
            r_k = radii[k]          # [K]

            dist_sq = ((z_i - c_k) ** 2).sum(dim=-1)  # [K]
            # soft weight (简化版: 直接用归一化距离平方的 softmax)
            q_ij = F.softmax(-dist_sq / self.tau_b, dim=-1)

            dist_sq_w = (q_ij * dist_sq).sum()
            r_sq = (q_ij * (r_k ** 2)).sum()
            L_intra = L_intra + F.relu(dist_sq_w - self.eta * r_sq)
            count += 1

        L_intra = L_intra / max(count, 1)

        # ---- L_overlap ----
        L_overlap = torch.tensor(0.0, device=device)
        cnt_ov = 0
        for k in range(self.n_classes):
            for p in range(self.K):
                c_kp = ball_centers[k, p]
                R_kp = radii[k, p]
                for l in range(self.n_classes):
                    for q in range(self.K):
                        if k == l and p == q:
                            continue
                        c_lq = ball_centers[l, q]
                        R_lq = radii[l, q]
                        d = ((c_kp - c_lq) ** 2).sum().sqrt()
                        L_overlap = L_overlap + F.relu(R_kp + R_lq + self.margin_m - d)
                        cnt_ov += 1

        L_overlap = L_overlap / max(cnt_ov, 1)

        # ---- L_div ----
        L_div = torch.tensor(0.0, device=device)
        cnt_div = 0
        for k in range(self.n_classes):
            for p in range(self.K):
                for q in range(p + 1, self.K):
                    d = ((ball_centers[k, p] - ball_centers[k, q]) ** 2).sum().sqrt()
                    L_div = L_div + F.relu(1.0 - d)
                    cnt_div += 1

        L_div = L_div / max(cnt_div, 1)

        return {
            'total': self.lambda_in * L_intra + self.lambda_ov * L_overlap + self.lambda_div * L_div,
            'intra': L_intra,
            'overlap': L_overlap,
            'div': L_div
        }
