必须严格保证在推理（Inference/Eval）阶段，模型绝对不能利用 labels 来修改特征 z。

修改 emb.py 中 AngularMultiCenterEmotionBall 的 forward 函数（约 645 行）：

修改前：

Python
        L_meb_dict = {"total": torch.tensor(0.0, device=z.device)}
        if labels is not None:
            L_meb_dict = self._compute_dualspace_meb_loss(...)
            delta_z = self._compute_residual_delta(z_sh, labels)
            # ... 下面是加残差的代码 ...
修改后（推荐方案）：
在测试阶段，完全关闭 delta_z 残差调整（或者将其设为 0），仅计算 Loss 供监控使用。

Python
        L_meb_dict = {"total": torch.tensor(0.0, device=z.device)}
        
        # 1. 如果传了 labels，永远可以计算 Loss (因为 Loss 不参与前向传播图)
        if labels is not None:
            L_meb_dict = self._compute_dualspace_meb_loss(
                z_sh=z_sh, z_sp=z_sp, labels=labels,
                guidance=guidance, update_radii=update_radii
            )

        # 2. 🚨 严格隔离：仅在训练阶段，且传了 labels 时，才允许计算和叠加残差！
        if self.training and labels is not None:
            delta_z = self._compute_residual_delta(z_sh, labels)
            scale = torch.clamp(self.residual_scale, min=0.0, max=self.residual_max)
            if guidance is not None:
                guide_scale = guidance.clamp(min=0.05, max=1.0).to(z.dtype)
                z_out = z + (scale * guide_scale) * torch.tanh(delta_z)
            else:
                z_out = z + scale * torch.tanh(delta_z)
        else:
            # 测试阶段，直接原样输出 z，杜绝任何 Ground Truth 的干预
            z_out = z
改完这个地方之后，你在跑一遍测试，看看 F1 掉不掉。如果修复前 F1 异常的高，修复后掉下来了，说明之前的性能很大程度上是沾了这个“作弊残差”的光。