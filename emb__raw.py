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

    同时额外维护一个轻量 AV 子共享分支，专门捕获音频-视频的非语言情绪协同线索。

    Args:
        unified_dim: 统一投影维度
        branch_dim:   每个分支的维度 (shared + specific 各占一半)
    """

    def __init__(self, unified_dim: int = 256, branch_dim: int = 128):
        super().__init__()
        half = unified_dim // 2
        self.shared_dim = half
        self.specific_dim = unified_dim - half
        # AV 子共享分支维度
        self.subshared_dim = unified_dim // 4
        self.proj_sub_a = nn.Linear(unified_dim, self.subshared_dim)
        self.proj_sub_v = nn.Linear(unified_dim, self.subshared_dim)
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
            sub_dict: AV 子共享分支输出 {'a': sub_a, 'v': sub_v}，t 为 None
        """
        u_dict, s_dict, sub_dict = {}, {}, {}

        for modal, p in p_dict.items():
            if modal == 't':
                h = F.relu(self.mlp_t(p))
                u_dict['t'] = self.proj_shared_t(h)
                s_dict['t'] = self.proj_specific_t(h)
                sub_dict['t'] = None
            elif modal == 'a':
                h = F.relu(self.mlp_a(p))
                u_dict['a'] = self.proj_shared_a(h)
                s_dict['a'] = self.proj_specific_a(h)
                sub_dict['a'] = self.proj_sub_a(h)
            elif modal == 'v':
                h = F.relu(self.mlp_v(p))
                u_dict['v'] = self.proj_shared_v(h)
                s_dict['v'] = self.proj_specific_v(h)
                sub_dict['v'] = self.proj_sub_v(h)

        return u_dict, s_dict, sub_dict


class ReliabilityScorer(nn.Module):
    def __init__(self, branch_dim: int = 128, quality_dim: int = 1):
        super().__init__()
        # input = [sim_ta, sim_tv, q] for text; [sim_ta/sim_tv, sim_av_sub, q] for a/v
        in_dim = 2 + quality_dim  # 2 个相似度 + 1 个质量 = 3

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

    def forward(self, u_dict: dict, s_dict: dict = None,
                sub_dict: dict = None, sim_av_sub: torch.Tensor = None):
        """
        升级后的 scorer：文本看与两边的一致性，音视频额外看 AV 子共享一致性。

        Args:
            u_dict: shared branch 输出
            s_dict: specific branch 输出（未使用）
            sub_dict: AV 子共享分支输出 {'a': sub_a, 'v': sub_v}
            sim_av_sub: 预计算的 AV 子共享相似度 [N, 1]，用于 a/v 的 scorer 输入
        Returns:
            alpha: 可靠性权重 [N, 3]
            r_dict: 原始评分
        """
        device = next(iter(u_dict.values())).device
        N = next(iter(u_dict.values())).size(0)

        alpha_list = []
        r_dict = {}

        for m in ['t', 'a', 'v']:
            u = u_dict.get(m)
            if u is not None:
                q_m = self._compute_quality(u)

                if m == 't':
                    # 文本：看与 audio 和 video 的一致性
                    g_ta = torch.zeros_like(q_m)
                    g_tv = torch.zeros_like(q_m)
                    u_norm = F.normalize(u, dim=-1)
                    if 'a' in u_dict:
                        other_norm = F.normalize(u_dict['a'], dim=-1)
                        g_ta = (u_norm * other_norm).sum(dim=-1, keepdim=True)
                    if 'v' in u_dict:
                        other_norm = F.normalize(u_dict['v'], dim=-1)
                        g_tv = (u_norm * other_norm).sum(dim=-1, keepdim=True)
                    scorer_input = torch.cat([g_ta, g_tv, q_m], dim=-1)

                elif m == 'a':
                    # 音频：看与文本的一致性 + AV 子共享一致性
                    g_ta = torch.zeros_like(q_m)
                    u_norm = F.normalize(u, dim=-1)
                    if 't' in u_dict:
                        other_norm = F.normalize(u_dict['t'], dim=-1)
                        g_ta = (u_norm * other_norm).sum(dim=-1, keepdim=True)
                    g_av_sub = sim_av_sub if sim_av_sub is not None else torch.zeros_like(q_m)
                    scorer_input = torch.cat([g_ta, g_av_sub, q_m], dim=-1)

                elif m == 'v':
                    # 视频：看与文本的一致性 + AV 子共享一致性
                    g_tv = torch.zeros_like(q_m)
                    u_norm = F.normalize(u, dim=-1)
                    if 't' in u_dict:
                        other_norm = F.normalize(u_dict['t'], dim=-1)
                        g_tv = (u_norm * other_norm).sum(dim=-1, keepdim=True)
                    g_av_sub = sim_av_sub if sim_av_sub is not None else torch.zeros_like(q_m)
                    scorer_input = torch.cat([g_tv, g_av_sub, q_m], dim=-1)

                scorer = getattr(self, f'scorer_{m}')
                r_m = scorer(scorer_input)
                alpha_m = torch.sigmoid(r_m)

                alpha_list.append(alpha_m)
                r_dict[m] = r_m
            else:
                alpha_list.append(torch.zeros(N, 1, device=device))

        alpha = torch.cat(alpha_list, dim=-1)

        # 【两级门控】：替代三模态总 softmax，避免文本压制音视频
        # 1. 提取原始 logit
        r_t = r_dict.get('t', torch.zeros(N, 1, device=device))
        r_a = r_dict.get('a', torch.zeros(N, 1, device=device))
        r_v = r_dict.get('v', torch.zeros(N, 1, device=device))

        # 2. 文本单独门控（不参与竞争）
        g_t = torch.sigmoid(r_t)

        # 3. AV 对子整体可靠性
        g_av = (sim_av_sub + 1.0) / 2.0

        # 4. A/V 内部 softmax 竞争
        tau_gating = 0.5
        w_av = F.softmax(torch.cat([r_a, r_v], dim=-1) / tau_gating, dim=-1)

        # 5. 组合最终权重（给 a/v 保底，不让文本无限压制）
        alpha_t = (0.55 + 0.30 * g_t).clamp(max=0.95)
        alpha_a = (0.10 + 0.35 * g_av * w_av[:, 0:1]).clamp(max=0.95)
        alpha_v = (0.10 + 0.35 * g_av * w_av[:, 1:2]).clamp(max=0.95)

        alpha = torch.cat([alpha_t, alpha_a, alpha_v], dim=-1)
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

        # 可学习的残差缩放因子 (零初始化)，gamma_t 初始等价 baseline
        # self.gamma_t = nn.Parameter(torch.tensor([0.05]))
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_v = nn.Parameter(torch.zeros(1))

        # RRA 损失权重
        self.lambda_nce = 0.05
        self.lambda_ortho = 1.0
        self.lambda_hsic = 0.00
        self.lambda_pair = 0.01

    def forward(self, h_t=None, h_a=None, h_v=None,
                speaker_ids=None, return_align_loss: bool = False):
        p_dict = self.projection(h_t=h_t, h_a=h_a, h_v=h_v)
        u_dict, s_dict, sub_dict = self.branch(p_dict)

        # 计算 AV 子共享相似度（用于 scorer 和后续两级门控）
        device = next(iter(u_dict.values())).device
        N = next(iter(u_dict.values())).size(0)
        sim_av_sub = torch.zeros(N, 1, device=device)  # 必须是 [N,1] 与 q_m 维度匹配
        if 'a' in sub_dict and 'v' in sub_dict and sub_dict['a'] is not None and sub_dict['v'] is not None:
            sub_a_norm = F.normalize(sub_dict['a'], dim=-1)
            sub_v_norm = F.normalize(sub_dict['v'], dim=-1)
            sim_av_sub = (sub_a_norm * sub_v_norm).sum(dim=-1, keepdim=True)

        alpha, r_dict = self.scorer(u_dict, s_dict, sub_dict, sim_av_sub)
        modalities = list(p_dict.keys())
        orig_dict = {'t': h_t, 'a': h_a, 'v': h_v}
        h_tilde_dict = {}
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

        # 传入 u_dict, s_dict, sub_dict 计算新损失
        L_align = self._compute_rra_loss(u_dict, s_dict, sub_dict,
                                         speaker_ids) if self.use_align_loss else torch.tensor(0.0, device=next(
            iter(p_dict.values())).device)

        if return_align_loss:
            return h_tilde_dict, L_align, alpha, sim_av_sub, r_dict
        return h_tilde_dict, alpha, sim_av_sub, r_dict

    # def _compute_hsic(self, u: torch.Tensor, s: torch.Tensor):
    #     """Linear HSIC: HSIC(u,s) = E[uu']E[ss'] 的 trace"""
    #     N = u.size(0)
    #     K_u = u @ u.t() / N
    #     K_s = s @ s.t() / N
    #     return (K_u * K_s).sum() / (N * N)
    def _compute_hsic(self, u: torch.Tensor, s: torch.Tensor, sigma: float = 1.0):
        """
        基于 RBF (Radial Basis Function) 核函数的 HSIC 计算。
        目标: 通过映射到无限维 RKHS，严格惩罚 shared (u) 和 specific (s) 之间的非线性依赖。
        """
        N = u.size(0)
        # 当 batch size 极小时，计算 HSIC 无统计学意义，直接截断
        if N < 2:
            return torch.tensor(0.0, device=u.device)

        # 为了保证 RBF 核计算的数值稳定性与尺度统一，强制进行 L2 归一化
        u_norm = F.normalize(u, dim=-1)
        s_norm = F.normalize(s, dim=-1)

        def get_rbf_kernel(x, sig):
            # 高效计算欧式距离平方矩阵: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
            x_sq = (x ** 2).sum(dim=1).view(-1, 1)
            dist_sq = x_sq + x_sq.view(1, -1) - 2.0 * torch.mm(x, x.t())
            # 引入 clamp 截断由于浮点精度误差导致的微小负数，保证距离非负
            dist_sq = torch.clamp(dist_sq, min=0.0)
            # 生成 RBF 核矩阵
            K = torch.exp(-dist_sq / (2.0 * sig ** 2))
            return K

        # 计算核矩阵 (默认 sigma=1.0，因为输入已归一化，最大距离平方为 4)
        K_u = get_rbf_kernel(u_norm, sigma)
        K_s = get_rbf_kernel(s_norm, sigma)

        # 构造中心化矩阵 H = I - 1/N * (1 1^T)
        H = torch.eye(N, device=u.device) - torch.ones((N, N), device=u.device) / float(N)

        # 计算中心化后的核矩阵 Kc = H * K * H
        Kc_u = torch.mm(torch.mm(H, K_u), H)
        Kc_s = torch.mm(torch.mm(H, K_s), H)

        # 计算 HSIC 经验估计值: Tr(Kc_u * Kc_s) / (N-1)^2
        hsic_value = torch.trace(torch.mm(Kc_u, Kc_s)) / ((N - 1) ** 2)

        return hsic_value

    def _compute_rra_loss(self, u_dict: dict, s_dict: dict, sub_dict: dict = None, speaker_ids=None):
        """
        最新模态对齐改进：非对称 InfoNCE 软对齐 + 共享/特异分支正交解耦 + AV 子共享对齐
        """
        device = next(iter(u_dict.values())).device
        N = next(iter(u_dict.values())).size(0)

        # 1. L_ortho: 正交约束，强制 shared(u) 和 specific(s) 捕获不同维度的信息
        L_ortho = torch.tensor(0.0, device=device)
        for m in u_dict.keys():
            u = F.normalize(u_dict[m], dim=-1)
            s = F.normalize(s_dict[m], dim=-1)
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

        # 3. L_pair: AV 子共享对齐（TSD 核心：捕获非语言情绪协同线索）
        # 【修复】：移除 .detach()，允许梯度回传到 proj_sub_a / proj_sub_v
        L_pair = torch.tensor(0.0, device=device)
        if sub_dict is not None and 'a' in sub_dict and 'v' in sub_dict:
            if sub_dict['a'] is not None and sub_dict['v'] is not None:
                sub_a_norm = F.normalize(sub_dict['a'], dim=-1)
                sub_v_norm = F.normalize(sub_dict['v'], dim=-1)
                pair_logits = torch.matmul(sub_a_norm, sub_v_norm.t()) / tau
                L_pair = F.cross_entropy(pair_logits, labels)

        # 融合损失：正交约束 + 对比损失 + HSIC（已关闭） + AV子共享对齐
        L_hsic = torch.tensor(0.0, device=device)
        if self.lambda_hsic > 0:
            for m in ['t', 'a', 'v']:
                if m in u_dict and m in s_dict:
                    L_hsic += self._compute_hsic(u_dict[m], s_dict[m])
            L_hsic = L_hsic / max(1, sum(1 for m in ['t', 'a', 'v'] if m in u_dict and m in s_dict))

        total = self.lambda_nce * L_nce + self.lambda_ortho * L_ortho + self.lambda_hsic * L_hsic + self.lambda_pair * L_pair

        return {
            'nce': L_nce,
            'ortho': L_ortho,
            'hsic': torch.tensor(0.0, device=device),
            'pair': L_pair,
            'total': total
        }


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
    Loss-only Dual-Space MEB
    目标：
        1) 在 MEB 内部把 z 划分为共享空间 z_sh 和特异空间 z_sp
        2) 粒球仅在 z_sh 上构建
        3) z_sp 只做弱约束（正交 + 方差），不做风格偏移
        4) forward 返回原始 z，不改写分类输入，只通过 loss 回传约束
    """

    def __init__(
            self,
            z_dim: int,
            n_classes: int,
            K_per_class: int = 2,
            tau_b: float = 0.15,
            dropout: float = 0.0,
            shared_ratio: float = 0.75,
            lambda_overlap: float = 0.3,
            lambda_div: float = 0.2,
            lambda_radius: float = 0.02,
            lambda_ortho: float = 0.02,
            lambda_var: float = 0.005,
            lambda_usage: float = 0.10,
            ema_momentum: float = 0.92,
            radius_min: float = 0.15,
            radius_max: float = 0.8,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.K = K_per_class
        self.tau_b = tau_b
        self.dropout = nn.Dropout(dropout)
        self.lambda_radius = lambda_radius
        self.lambda_overlap = lambda_overlap
        self.lambda_div = lambda_div
        self.lambda_ortho = lambda_ortho
        self.lambda_var = lambda_var
        self.lambda_usage = lambda_usage
        self.ema_momentum = ema_momentum
        self.radius_min = radius_min
        self.radius_max = radius_max

        # 1) 双空间划分：shared / specific
        self.z_sh_dim = int(z_dim * shared_ratio)
        self.z_sp_dim = z_dim - self.z_sh_dim

        self.pre_ln = nn.LayerNorm(z_dim)

        # 用单层线性投影，避免你前面 MLP 扰动过大
        self.proj_shared = nn.Linear(z_dim, self.z_sh_dim)
        self.proj_specific = nn.Linear(z_dim, self.z_sp_dim)
        self.proj_sh_to_z = nn.Linear(self.z_sh_dim, z_dim)
        # 2) 球心只建在 shared 空间
        self.ball_centers = nn.Parameter(
            torch.randn(n_classes, K_per_class, self.z_sh_dim)
        )

        # torch.zeros 经过 Sigmoid 映射后默认处于 radius_min 和 radius_max 的中点
        # self.raw_radii = nn.Parameter(
        #     torch.zeros(n_classes, K_per_class)
        # )
        # EMA 目标半径：只做 target，不直接硬覆盖参数
        self.register_buffer(
            "ema_radii",
            torch.ones(n_classes, K_per_class) * 0.2
        )
        # 检验情况
        self._is_initialized = False
        self.debug_mode = True
        self.debug_epoch_interval = 10  # 确保这行存在且拼写一致
        self.last_debug_epoch = -1  # 确保这行存在且拼写一致
        # ============================================================

        self.register_buffer("prev_centers", torch.zeros(n_classes, K_per_class, self.z_sh_dim))
        self.register_buffer("prev_radii", torch.ones(n_classes, K_per_class) * 0.2)

    # =========================================================
    # 基础函数
    # =========================================================
    def _split_spaces(self, z: torch.Tensor):
        z0 = self.pre_ln(z)
        z_sh = self.proj_shared(z0)
        z_sp = self.proj_specific(z0)
        return z_sh, z_sp

    def _compute_ortho_loss(self, z_sh: torch.Tensor, z_sp: torch.Tensor):
        """
        计算 shared 和 specific 空间的正交损失 (Cross-Correlation)
        通过惩罚交叉相关矩阵的 Frobenius 范数，确保两个空间线性独立，且完美兼容维度不一致的情况。
        """
        # 沿批次维度 (Batch, dim=0) 归一化，使得每个神经元在当前 batch 内向量长度为 1
        z_sh_n = F.normalize(z_sh, p=2, dim=0)
        z_sp_n = F.normalize(z_sp, p=2, dim=0)

        # 计算交叉相关矩阵，形状: [D_sh, D_sp]
        corr_matrix = torch.matmul(z_sh_n.t(), z_sp_n)

        # 惩罚相关性矩阵中元素的平方均值
        return corr_matrix.pow(2).mean()

    def _compute_specific_var_loss(self, z_sp: torch.Tensor, vmin: float = 0.05):
        var = z_sp.var(dim=0, unbiased=False)
        return F.relu(vmin - var).mean()

    def _farthest_point_init(self, feats: torch.Tensor, K: int):
        N, D = feats.shape
        if N == 0:
            return torch.randn(K, D, device=feats.device)

        if N == 1:
            return feats.repeat(K, 1)

        centers = []
        first_idx = 0
        centers.append(feats[first_idx:first_idx + 1])

        min_dist = 1.0 - torch.matmul(feats, centers[0].t()).squeeze(-1)

        for _ in range(1, K):
            farthest_idx = torch.argmax(min_dist).item()
            new_center = feats[farthest_idx:farthest_idx + 1]
            centers.append(new_center)

            dist_to_new = 1.0 - torch.matmul(feats, new_center.t()).squeeze(-1)
            min_dist = torch.minimum(min_dist, dist_to_new)

        centers = torch.cat(centers, dim=0)
        if centers.size(0) < K:
            repeat_num = K - centers.size(0)
            centers = torch.cat([centers, centers[:repeat_num]], dim=0)
        return centers

    # =========================================================
    # 保持与 model.py 兼容：函数名不变
    # =========================================================
    def _init_ball_centers_kmeans(self, z_features: torch.Tensor, labels: torch.Tensor):
        """
        保留函数名以兼容 model.py
        实际做法：
            z -> z_sh -> normalize -> farthest init centers
            然后基于类内样本到中心的软分配距离，初始化半径
        """
        with torch.no_grad():
            z_sh, _ = self._split_spaces(z_features)
            z_sh = F.normalize(z_sh, dim=-1)

            for k in range(self.n_classes):
                mask = (labels == k)
                if mask.sum() < 1:
                    continue

                feats_k = z_sh[mask]  # [Nk, D]
                if feats_k.size(0) <= self.K:
                    center_mean = feats_k.mean(dim=0, keepdim=True)
                    centers_k = center_mean.repeat(self.K, 1)
                else:
                    centers_k = self._farthest_point_init(feats_k, self.K)

                self.ball_centers.data[k] = centers_k

                # ===== 半径初始化：按当前类样本到中心的软分配距离 =====
                sim = torch.matmul(feats_k, centers_k.t())
                dist = 1.0 - sim
                q = F.softmax(sim / self.tau_b, dim=-1)

                sum_q = q.sum(dim=0) + 1e-6
                init_r = (q * dist).sum(dim=0) / sum_q

                # 【核心修复：建立 5% 的安全缓冲带，拒绝 Sigmoid 极值】
                margin = (self.radius_max - self.radius_min) * 0.05
                safe_min = self.radius_min + margin
                safe_max = self.radius_max - margin

                init_r_safe = torch.clamp(init_r, min=safe_min, max=safe_max)
                # norm_r = (init_r_safe - self.radius_min) / (self.radius_max - self.radius_min)
                #
                # # 此时反解出的参数必定落在 [-2.94, 2.94] 的高梯度活跃区
                # inverse_sigmoid_r = torch.log(norm_r / (1.0 - norm_r))
                #
                # self.raw_radii.data[k] = inverse_sigmoid_r
                self.ema_radii.data[k] = init_r_safe
            self._is_initialized = True

    # =========================================================
    # shared 空间上的角距离粒球损失
    # =========================================================
    def _compute_shared_ball_loss(
            self,
            z_sh: torch.Tensor,
            labels: torch.Tensor,
            sample_rel: torch.Tensor = None,
            update_radii: bool = True,
    ):
        device = z_sh.device
        batch_size = z_sh.size(0)

        z_norm = F.normalize(z_sh, dim=-1)  # [N, Dsh]
        c_norm = F.normalize(self.ball_centers, dim=-1)  # [C, K, Dsh]

        # 【修改点 1】：使用 Sigmoid 映射，保证全局导数连续
        radii = self.ema_radii.clone().detach()

        if sample_rel is None:
            sample_rel = torch.ones(batch_size, 1, device=device)

        L_intra = torch.tensor(0.0, device=device)
        valid_count = 0

        # 用于统计 batch 的软半径目标
        sum_q = torch.zeros(self.n_classes, self.K, device=device)
        sum_qd = torch.zeros(self.n_classes, self.K, device=device)
        q_list_for_grad = {k: [] for k in range(self.n_classes)}
        for i in range(batch_size):

            k = labels[i].item()
            z_i = z_norm[i:i + 1]  # [1, D]
            c_k = c_norm[k]  # [K, D]

            sim_ik = torch.matmul(z_i, c_k.t())  # [1, K]
            dist_ik = 1.0 - sim_ik  # [1, K]
            q_ik = F.softmax(sim_ik / self.tau_b, dim=-1)  # [1, K]

            # 收集带梯度的 q_ik 用于计算信息熵
            q_list_for_grad[k].append(q_ik)

            # 【修复1】：切断 q_ik 在拉近距离时的捷径梯度 (阻断模型通过改变分配来作弊)
            dist_w = (q_ik.detach() * dist_ik).sum()

            # 【修复2】：彻底剥夺分类损失直接推高半径的权限 (半径全权交由 ema_radii 和 L_radius 控制)
            r_w = (q_ik.detach() * radii[k:k + 1]).sum()

            rel_i = sample_rel[i, 0].detach()

            # 【修复3】：废除 Softplus，使用 ReLU 设定硬边界。
            # 仅在点出界时产生拉近特征与球心的梯度，一旦入球，梯度严格为 0
            margin_diff = dist_w - r_w.detach()
            L_intra += rel_i * F.relu(margin_diff)
            valid_count += 1

            if update_radii and self.training:
                # 这里的 detach 是为了 EMA，保留不动
                sum_q[k] += q_ik.detach().squeeze(0)
                sum_qd[k] += (q_ik.detach() * dist_ik.detach()).squeeze(0)

        if valid_count > 0:
            L_intra = L_intra / valid_count

            # ---------- 【关键修正】：使用保留梯度的分配矩阵计算 Balance Loss ----------
            # ---------- 【关键修正】：基于 InfoMax 的双重分配熵约束 ----------
        L_balance = torch.tensor(0.0, device=device)
        valid_classes = 0
        for k in range(self.n_classes):
            if len(q_list_for_grad[k]) > 0:
                # 沿批次拼接该类的所有软分配 [N_k, K]
                q_k_stack = torch.cat(q_list_for_grad[k], dim=0)

                # 1. Marginal Entropy (宏观平衡): 强制该类下的多个球被均匀使用，防单球独大
                sum_q_grad = q_k_stack.sum(dim=0)
                p_k = sum_q_grad / (sum_q_grad.sum() + 1e-8)
                entropy_marginal = - (p_k * torch.log(p_k + 1e-8)).sum()

                # 2. Conditional Entropy (微观尖锐): 强制每个样本必须清晰归属于其中一个球，防模棱两可
                entropy_conditional = - (q_k_stack * torch.log(q_k_stack + 1e-8)).sum(dim=-1).mean()

                # 理想状态：宏观均匀分布 (entropy_marginal 趋近 ln(K)), 微观确定性极高 (entropy_conditional 趋近 0)
                L_balance += (math.log(self.K) - entropy_marginal) + entropy_conditional
                valid_classes += 1

        if valid_classes > 0:
            L_balance = L_balance / valid_classes

        # ---------- 类间 overlap ----------
        margin_ov = 0.3
        c_flat = c_norm.view(self.n_classes * self.K, -1)
        sim_matrix = torch.matmul(c_flat, c_flat.t())

        mask = torch.ones_like(sim_matrix)
        for k in range(self.n_classes):
            mask[k * self.K:(k + 1) * self.K, k * self.K:(k + 1) * self.K] = 0.0

        overlap_penalties = F.relu(sim_matrix - margin_ov) * mask
        L_overlap = overlap_penalties.sum() / (mask.sum() + 1e-6)

        # ---------- 类内多球防塌缩 ----------
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
            L_div = L_div / max(div_count, 1)

        # ---------- EMA 半径目标 ----------
        if update_radii and self.training:
            with torch.no_grad():
                batch_radii = self.ema_radii.clone().detach()
                valid_mask = (sum_q > 1e-6)
                batch_radii[valid_mask] = sum_qd[valid_mask] / (sum_q[valid_mask] + 1e-6)

                batch_radii = torch.clamp(
                    batch_radii,
                    min=self.radius_min,
                    max=self.radius_max
                )

                self.ema_radii.copy_(
                    self.ema_momentum * self.ema_radii
                    + (1.0 - self.ema_momentum) * batch_radii
                )

        # L_radius = F.smooth_l1_loss(radii, self.ema_radii.detach())

        # 【修改点 4】：将 L_balance 加入总损失
        L_ball_total = (
                L_intra
                + self.lambda_overlap * L_overlap
                + self.lambda_div * L_div
                # + self.lambda_radius * L_radius
                + 0.15 * L_balance
        )

        # ===== debug metrics =====
        with torch.no_grad():
            # 当前有效半径
            radii_now = radii.detach()

            # 中心漂移 / 半径漂移
            center_shift = (c_norm - F.normalize(self.prev_centers + 1e-8, dim=-1)).pow(2).sum(dim=-1).sqrt().mean()
            radius_shift = (radii_now - self.prev_radii).abs().mean()

            # 半径与EMA是否贴合
            radius_gap = (radii_now - self.ema_radii).abs().mean()

            # 样本是否落球内
            inside_count, total_count = 0.0, 0.0
            assign_entropy = 0.0

            for i in range(batch_size):
                k = labels[i].item()
                z_i = z_norm[i:i + 1]
                c_k = c_norm[k]
                sim_ik = torch.matmul(z_i, c_k.t())
                dist_ik = 1.0 - sim_ik
                q_ik = F.softmax(sim_ik / self.tau_b, dim=-1)

                dist_w = (q_ik * dist_ik).sum()
                r_w = (q_ik * radii_now[k:k + 1]).sum()

                inside_count += float((dist_w <= r_w).item())
                total_count += 1.0
                assign_entropy += float((-(q_ik * (q_ik + 1e-8).log()).sum()).item())

            inside_rate = inside_count / max(total_count, 1.0)
            assign_entropy = assign_entropy / max(total_count, 1.0)

            # 保存上一时刻参数
            self.prev_centers.copy_(self.ball_centers.data)
            self.prev_radii.copy_(radii_now.data)

        return {
            "ball_total": L_ball_total,
            "intra": L_intra,
            "overlap": L_overlap,
            "div": L_div,
            "radius": torch.tensor(0.0, device=device),
            "center_shift": torch.tensor(0.0, device=device),
            "radius_shift": torch.tensor(0.0, device=device),
            "radius_gap": radius_gap,
            "inside_rate": torch.tensor(inside_rate, device=device),
            "assign_entropy": torch.tensor(assign_entropy, device=device),
            "r_mean": radii_now.mean(),
            "r_std": radii_now.std(unbiased=False),
            "usage": L_balance.detach(),  # <--- 将未定义的 L_usage 替换为实际计算的 L_balance
        }

    # =========================================================
    # 总损失
    # =========================================================
    def _compute_dualspace_meb_loss(
            self,
            z_sh: torch.Tensor,
            z_sp: torch.Tensor,
            labels: torch.Tensor,
            sample_rel: torch.Tensor = None,
            update_radii: bool = True,
    ):
        ball_dict = self._compute_shared_ball_loss(
            z_sh=z_sh,
            labels=labels,
            sample_rel=sample_rel,
            update_radii=update_radii
        )

        L_ortho = self._compute_ortho_loss(z_sh, z_sp)
        L_var = self._compute_specific_var_loss(z_sp)

        L_total = (
                ball_dict["ball_total"]
                + self.lambda_ortho * L_ortho
                + self.lambda_var * L_var
        )

        return {
            "total": L_total,
            "ball_total": ball_dict["ball_total"],
            "intra": ball_dict["intra"],
            "overlap": ball_dict["overlap"],
            "div": ball_dict["div"],
            "radius": ball_dict["radius"],
            "ortho": L_ortho,
            "var": L_var,
            "center_shift": ball_dict["center_shift"],
            "radius_shift": ball_dict["radius_shift"],
            "radius_gap": ball_dict["radius_gap"],
            "inside_rate": ball_dict["inside_rate"],
            "assign_entropy": ball_dict["assign_entropy"],
            "r_mean": ball_dict["r_mean"],
            "r_std": ball_dict["r_std"],
            "usage": ball_dict["usage"],
        }

    # =========================================================
    # forward：只返回原始 z，不改分类输入
    # =========================================================
    def forward(
            self,
            z: torch.Tensor,
            labels: torch.Tensor = None,
            sample_rel: torch.Tensor = None,
            update_radii: bool = True,
            epoch: int = None  # <--- 新增 epoch 参数
    ):
        z_in = self.dropout(z)
        z_sh, z_sp = self._split_spaces(z_in)

        L_meb_dict = {"total": torch.tensor(0.0, device=z.device)}
        if labels is not None:
            L_meb_dict = self._compute_dualspace_meb_loss(
                z_sh=z_sh, z_sp=z_sp, labels=labels,
                sample_rel=sample_rel, update_radii=update_radii
            )

            # --- 修改的调试判断逻辑 ---
            if self.debug_mode and self.training and epoch is not None:
                # 仅当 epoch 是 10 的倍数，且该 epoch 尚未打印过时触发
                if epoch % self.debug_epoch_interval == 0 and epoch != self.last_debug_epoch:
                    print(
                        f"[MEB-Debug] epoch={epoch} "
                        f"intra={L_meb_dict['intra'].item():.4f} "
                        f"overlap={L_meb_dict['overlap'].item():.4f} "
                        f"div={L_meb_dict['div'].item():.4f} "
                        f"radius={L_meb_dict['radius'].item():.4f} "
                        f"ortho={L_meb_dict['ortho'].item():.4f} "
                        f"var={L_meb_dict['var'].item():.4f} "
                        f"inside={L_meb_dict['inside_rate'].item():.4f} "
                        f"Hq={L_meb_dict['assign_entropy'].item():.4f} "
                        f"r_mean={L_meb_dict['r_mean'].item():.4f} "
                        f"r_std={L_meb_dict['r_std'].item():.4f} "
                        f"r_gap={L_meb_dict['radius_gap'].item():.4f} "
                        f"c_shift={L_meb_dict['center_shift'].item():.4f} "
                        f"r_shift={L_meb_dict['radius_shift'].item():.4f}"
                        f"usage={L_meb_dict['usage'].item():.4f} "
                    )
                    self.last_debug_epoch = epoch  # 锁定当前 epoch
            # ------------------------

        return z, L_meb_dict

    def get_ball_params(self):
        return {
            "centers": self.ball_centers.data,
            "radii": self.ema_radii.data,  # 直接返回 ema_radii
            "ema_radii": self.ema_radii.data
        }




