# 无标签优化目标修正（初轮结束后、重跑前冻结）

初轮9次（3版×3seed）保留 outputs/flow_latent_20261007_v* 及不可变代码快照。其 E-step/边缘密度正确，但“每簇样本量缩放ridge＋未包含ridge项的残差协方差”不是同一 MAP 目标的 M-step。因此初轮只能叫交替正则化条件密度拟合，不能声称严格 EM 单调收敛。此修正来自公式核对，与自然AUC无关；不改变图条件、风险方向、来源清单、seed、阈值或选簇。

修正后的模型：非截距系数 C'_k | Σ_k ~ MatrixNormal(0, λ⁻¹ I, Σ_k)，λ=.01×N_fit，固定不随 posterior count 改变；截距平坦先验，π~Dir(1.2)。Σ_k=U_k U_kᵀ+σ²_k I，rank4，σ²>=.05。

忽略与参数无关常数：

J = Σ_t logΣ_k π_k N(x_t; C_k[1,c_t],Σ_k)
    + .2Σ_k logπ_k
    − .5Σ_k[d_c logdetΣ_k + λ tr(Σ_k⁻¹ C'_kᵀ C'_k)].

E-step q_tk ∝ π_k N(x_t|…)。令 N_k=Σ_t q_tk、F=[1,c]、Λ=diag(0,λ,…,λ)：

C_k=(Fᵀ diag(q_k)F+Λ)⁻¹Fᵀ diag(q_k)X；
S_k=[(X−FC_k)ᵀ diag(q_k)(X−FC_k)+C_kᵀΛC_k]/(N_k+d_c)。

对 S_k 特征分解，以尾部平均特征值（下限.05）作为σ²，前4个方向按 max(eigenvalue−σ²,0) 构造 U。π MAP=(N_k+.2)/(N_fit+.8)。这是同一目标上的解析/谱 M-step；每轮检查 J 不下降，仅看数据 logdensity 不要求单调，因为还有先验项。

逐 token q(z_t) 保存4态 posterior；q(u_t|z_t=k)=N(U_kᵀΣ_k⁻¹[x_t−C_kf_t], I−U_kᵀΣ_k⁻¹U_k)，保存每 token/每态4维后验均值及每态协方差。后者不是风险标签。

修正版输出统一为 outputs/flow_latent_20261007_map_v{1,2,3}[\_seed43|\_seed44]。继续同样3个预定图条件，9次拟合，0新8B前向。初轮损失并未直接改变其已保存异常排名，但不将其当修正版同一优化实现。最终报告两轮数据，不删除初轮。
