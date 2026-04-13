🚨 泄露是如何发生的？（致命的 4 步）
第 1 步：测试集依然传入了真实的 Label
在 train_IEMOCAP.py 中，无论 train=True 还是 train=False，你都把真实的 label 传给了模型：

Python
# train_IEMOCAP.py 
log_prob, e_i, e_n, e_t, e_l = model(..., labels=flat_label, ...)
第 2 步：model.py 把 Label 直接喂给了 MEB
在 model.py 的前向传播中，只要 labels 不为空，它就直接传给了 MEB，完全没有区分当前是 Training 还是 Eval 阶段：

Python
# model.py 约 675 行 和 705 行
emotions_feat_enhanced, L_meb_dict = self.meb(
    emotions_feat,
    labels=labels,       # <--- 无论训练还是测试，真值都被送进去了！
    guidance=meb_guidance.detach() if meb_guidance is not None else None,
    ...
)
第 3 步：MEB 直接用 Label 提取了正确的“情绪球心”
在 emb.py 中，MEB 调用了 _compute_residual_delta(z_sh, labels)。我们来看看这个函数里面干了什么：

Python
# emb.py 约 420 行
def _compute_residual_delta(self, z_sh: torch.Tensor, labels: torch.Tensor):
    z_norm = F.normalize(z_sh, dim=-1)  
    c_norm = F.normalize(self.ball_centers, dim=-1)  
    labels = labels.long()
    
    # 🚨 致命泄露点：直接用 Ground Truth Label 把当前样本该去哪个球心拿了出来！
    centers_lbl = c_norm[labels]  
    
    sim = torch.einsum('nd,nkd->nk', z_norm, centers_lbl)
    q = F.softmax(sim / self.tau_b, dim=-1)
    center_mix = torch.einsum('nk,nkd->nd', q, centers_lbl)
    
    # 计算向正确球心靠拢的偏移量
    delta_sh = center_mix - z_norm 
    delta_z = self.proj_sh_to_z(delta_sh)
    return delta_z
第 4 步：RRA 的 Guidance 放行了作弊信号
在 emb.py 的 forward 最后，这个带有标准答案信息的偏移量 delta_z 被加回了原特征中，而 RRA 传过来的 guidance 只是调节了作弊的“力度”：

Python
# emb.py 约 658 行
if guidance is not None:
    guide_scale = guidance.clamp(min=0.05, max=1.0).to(z.dtype)
    # 带有真值信息的 delta_z 被加进了特征中送给最后的分类器
    z_out = z + (scale * guide_scale) * torch.tanh(delta_z) 
💡 结论：
RRA 模块本身没有泄露数据，它只是根据多模态的一致性计算了一个置信度 meb_guidance。
但是 MEB 模块在测试阶段“作弊”了。它利用真实的 Labels 算出了一条通往正确答案的捷径（delta_z），而 RRA 的 guidance 不知情地充当了这个作弊信号的权重。