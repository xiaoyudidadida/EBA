1. AV 子共享分支（SharedSpecificBranch）
新增 proj_sub_a 和 proj_sub_v（维度为 unified_dim // 4），专门从音频/视频特征中提取非语言情绪协同线索
forward 返回值从二元组 (u_dict, s_dict) 扩展为三元组 (u_dict, s_dict, sub_dict)，其中 sub_dict['t'] = None
2. Scorer 升级（ReliabilityScorer）
输入从 [g_bar, q]（2维）升级为三项输入 [sim_ta, sim_tv, q]（3维）
文本看与 audio 和 video 的一致性（g_ta + g_tv）
音视频额外看 AV 子共享一致性（g_ta/g_tv + sim_av_sub）
内置两级门控：alpha_t = (0.55 + 0.30 * g_t).clamp(max=0.95)，alpha_a/v = (0.10 + 0.35 * g_av * w_av).clamp(max=0.95)
3. RRA 损失扩展（_compute_rra_loss）
新增 L_pair：基于 AV 子共享分支的对比对齐损失，专门拉近音频-视频的非语言情绪协同表示
forward 返回值新增 sim_av_sub 和 r_dict，供两级门控使用
4. sample_rel 重定义（model.py）
sample_rel 从三模态平均改为 0.5 * alpha_a + 0.5 * alpha_v，只吃音视频可靠性，避免文本长期偏高导致 MEB 误判
5. Bug 修复
Bug 1（无需修复）：h_tilde_dict = {} 在遍历前已初始化
Bug 2：sim_av_sub 默认值从标量 torch.tensor(0.0) 改为 torch.zeros(N, 1)，与 q_m 维度一致
Bug 3：两级门控逻辑统一移入 ReliabilityScorer.forward，不再在 model.py 中重复计算，确保 RRA 残差融合使用正确的门控权重
Bug 4：L_pair 计算中移除 .detach()，允许梯度回传到 proj_sub_a / proj_sub_v，保证 AV 子共享分支可学习
Bug 5（rra_loss_dict 默认初始化）：forward 开始处添加 rra_loss_dict = {}，防止 RRA 未启用或 N_valid=0 时末尾 UnboundLocalError
