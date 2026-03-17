import json, numpy as np, matplotlib.pyplot as plt
import main_1 as m1

ckpt = "checkpoints/main1_quick_t6_p80_n80_seed8/ckpt_t6"
hist_path = "checkpoints/main1_quick_t6_p80_n80_seed8/history.json"
out1 = "checkpoints/main1_quick_t6_p80_n80_seed8/rmse_by_stage.png"
out2 = "checkpoints/main1_quick_t6_p80_n80_seed8/beta_fit_t6.png"

# 1) RMSE by stage
hist = json.load(open(hist_path, "r", encoding="utf-8"))
pairs = sorted((int(h["stage"]), float(h["train_rmse"])) for h in hist if isinstance(h, dict) and "stage" in h and "train_rmse" in h)
s = [x for x,_ in pairs]; r = [y for _,y in pairs]
plt.figure(figsize=(6,4)); plt.plot(s,r,marker="o"); plt.xlabel("stage"); plt.ylabel("train_rmse"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out1,dpi=160); plt.close()

# 2) final beta estimate vs true (默认 true_beta_funcs_default)
tr = m1.IncrementalVCMTrainer.load_checkpoint(ckpt)
t = np.linspace(0,6,400)
B = m1.bspline_design_matrix(t, tr.knots, tr.k)
beta_hat = np.column_stack([B @ tr.coef_blocks[p] for p in range(len(tr.coef_blocks))])
signal_idx = [1,2,3,4,5]
beta_true = m1.true_beta_funcs_default()
fig,axs = plt.subplots(3,2,figsize=(10,8),sharex=True); axs=axs.ravel()
for i,p in enumerate(signal_idx):
    axs[i].plot(t, beta_hat[:,p], label=f"hat p={p}")
    axs[i].plot(t, beta_true[i](t), "--", label="true")
    axs[i].grid(alpha=.3); axs[i].legend(fontsize=8)
axs[-1].axis("off")
fig.tight_layout(); fig.savefig(out2,dpi=160); plt.close(fig)
