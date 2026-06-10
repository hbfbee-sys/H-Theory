"""
H-Theory Recall Simulation
==========================

目标：
用“复平面函数计算”模拟 H-Theory 中的回忆机制。

核心思想：
1. 记忆不是持续活动，而是静态 H 型分布编码 MEMORY_Q。
2. 回忆不是精确回放，而是在当前 H 型生态背景 CONTEXT_Q 上，
   由线索 CUE 触发的吸引子重建。
3. 周边 H 型生态改变越大，回忆误差越大。
4. 每次回忆会轻微改写记忆，形成 reconsolidation（再巩固）。

输出：
- 控制台打印回忆过程指标
- recall_output/recall_order_parameters.png
- recall_output/recall_q_trajectory.png
- recall_output/recall_error.png
- recall_output/final_phase_field.png
- recall_output/final_H_decoding.png

运行：
python h_theory_recall_simulation.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 全局参数
# ============================================================

N = 64
STEPS = 900
DT = 0.025
SEED = 42

# 复场动力学参数
ALPHA = 1.8
BETA = 0.45
GAMMA = 0.25
OMEGA = 0.55
D = 1.2
ETA = 0.25
NOISE = 0.12

# 回忆参数
CUE_STRENGTH = 0.18
CUE_FRACTION = 0.12
K_MEMORY = 0.75
K_CONTEXT = 0.25
RECONSOLIDATION_RATE = 0.03

# 意识判定参数
PHI_C = 0.30
S_C = 0.10
RC_BASE = 0.80
RC_SLOPE = 0.125

rng = np.random.default_rng(SEED)


# ============================================================
# 2. H 型基态：四种感受质吸引子
# ============================================================

H_PHASES = np.array([
    0.0,                 # H1_Red
    2 * np.pi / 3,       # H2_Blue
    4 * np.pi / 3,       # H3_Green
    np.pi / 2,           # H4_White
])

H_NAMES = ["H1_Red", "H2_Blue", "H3_Green", "H4_White"]
H_VALUES = np.exp(1j * H_PHASES)


# ============================================================
# 3. 静态记忆编码与当前背景生态
# ============================================================

MEMORY_Q = np.array([0.52, 0.46, 0.01, 0.01], dtype=float)
MEMORY_Q = MEMORY_Q / MEMORY_Q.sum()

CONTEXT_Q = np.array([0.30, 0.20, 0.30, 0.20], dtype=float)
CONTEXT_Q = CONTEXT_Q / CONTEXT_Q.sum()


# ============================================================
# 4. 工具函数
# ============================================================

def normalize_prob(q: np.ndarray) -> np.ndarray:
    q = np.maximum(q, 1e-12)
    return q / q.sum()


def sample_h_field(q: np.ndarray, size: int) -> np.ndarray:
    flat = rng.choice(4, size=size * size, p=q)
    return flat.reshape(size, size)


def h_index_to_complex(idx: np.ndarray) -> np.ndarray:
    return H_VALUES[idx]


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4 * field
    )


def nearest_H_decode(field: np.ndarray):
    phase_z = np.exp(1j * np.angle(field))
    distances = np.abs(phase_z[..., None] - H_VALUES[None, None, :])
    idx = np.argmin(distances, axis=-1)
    counts = np.array([(idx == k).sum() for k in range(4)], dtype=float)
    q = counts / counts.sum()
    return idx, q


def qualia_entropy(q: np.ndarray) -> float:
    q = normalize_prob(q)
    return float(-np.sum(q * np.log2(q + 1e-12)))


def dynamic_Rc(Hq: float) -> float:
    return float(max(0.25, RC_BASE - RC_SLOPE * Hq))


def compute_R(field: np.ndarray) -> float:
    phase = np.angle(field)
    return float(np.abs(np.sum(np.exp(1j * phase))) / field.size)


def compute_S(field: np.ndarray) -> float:
    amp = np.abs(field)
    return float(np.mean(amp > 0.55))


def compute_phi(strong_edges: np.ndarray, polarity: np.ndarray) -> float:
    rho_strong = float(np.mean(strong_edges))

    p0 = float(np.mean(polarity == 0))
    p1 = float(np.mean(polarity == 1))

    H_pol = 0.0
    for p in (p0, p1):
        if p > 0:
            H_pol += -p * np.log2(p)

    return float(rho_strong * H_pol)


def recall_score(q_current: np.ndarray, q_memory: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.sum(np.abs(q_current - q_memory)))


def memory_error(q_current: np.ndarray, q_memory: np.ndarray) -> float:
    return float(np.linalg.norm(q_current - q_memory))


def consciousness(R: float, Phi: float, S: float, Hq: float) -> int:
    Rc = dynamic_Rc(Hq)
    return int(R > Rc and Phi > PHI_C and S > S_C)


def q_to_complex_mean(q: np.ndarray) -> complex:
    q = normalize_prob(q)
    return complex(np.sum(q * H_VALUES))


# ============================================================
# 5. 初始化网络
# ============================================================

initial_idx = sample_h_field(CONTEXT_Q, N)
psi = h_index_to_complex(initial_idx)

psi = (0.8 + 0.25 * rng.random((N, N))) * psi
psi *= np.exp(1j * 0.3 * rng.normal(size=(N, N)))

strong_edges = rng.random((N, N)) < 0.45
polarity = rng.integers(0, 2, size=(N, N))

cue_mask = rng.random((N, N)) < CUE_FRACTION
memory_complex = q_to_complex_mean(MEMORY_Q)
context_complex = q_to_complex_mean(CONTEXT_Q)


# ============================================================
# 6. 记录变量
# ============================================================

history_R = []
history_Phi = []
history_S = []
history_Hq = []
history_Rc = []
history_C = []
history_score = []
history_err = []
history_q = []


# ============================================================
# 7. 主仿真循环
# ============================================================

for t in range(STEPS):
    decoded_idx, q_current = nearest_H_decode(psi)

    R = compute_R(psi)
    Phi = compute_phi(strong_edges, polarity)
    S = compute_S(psi)
    Hq = qualia_entropy(q_current)
    Rc_t = dynamic_Rc(Hq)
    C = consciousness(R, Phi, S, Hq)

    score = recall_score(q_current, MEMORY_Q)
    err = memory_error(q_current, MEMORY_Q)

    history_R.append(R)
    history_Phi.append(Phi)
    history_S.append(S)
    history_Hq.append(Hq)
    history_Rc.append(Rc_t)
    history_C.append(C)
    history_score.append(score)
    history_err.append(err)
    history_q.append(q_current.copy())

    current_complex = q_to_complex_mean(q_current)

    memory_force = K_MEMORY * (memory_complex - current_complex)
    context_force = K_CONTEXT * (context_complex - current_complex)

    cue_term = np.zeros_like(psi)
    cue_term[cue_mask] = CUE_STRENGTH * (memory_complex - psi[cue_mask])

    global_memory_term = memory_force * psi
    global_context_term = context_force * psi

    dpsi = (
        (ALPHA + 1j * OMEGA) * psi
        - (BETA + 1j * GAMMA) * (np.abs(psi) ** 2) * psi
        + D * laplacian(psi)
        + ETA * R * psi
        + global_memory_term
        + global_context_term
        + cue_term
    )

    psi = psi + DT * dpsi
    psi *= np.exp(1j * NOISE * rng.normal(size=(N, N)))

    phase = np.angle(psi)
    diff = np.abs(np.angle(np.exp(1j * (phase - np.roll(phase, -1, axis=1)))))
    anti_phase = np.abs(diff - np.pi) < 0.10
    strong_edges = np.logical_or(strong_edges, anti_phase)

    if t % 100 == 0:
        print(
            f"step={t:04d} "
            f"R={R:.3f} Rc={Rc_t:.3f} Phi={Phi:.3f} S={S:.3f} "
            f"Hq={Hq:.3f} C={C} score={score:.3f} "
            f"q={np.round(q_current, 3)}"
        )


# ============================================================
# 8. 回忆后的再巩固
# ============================================================

q_final = history_q[-1]
MEMORY_AFTER_RECALL = normalize_prob(
    (1.0 - RECONSOLIDATION_RATE) * MEMORY_Q
    + RECONSOLIDATION_RATE * q_final
)

print("\n=== Final Recall Result ===")
print("Original MEMORY_Q:      ", np.round(MEMORY_Q, 4))
print("Current recalled q:     ", np.round(q_final, 4))
print("Memory after recall:    ", np.round(MEMORY_AFTER_RECALL, 4))
print(f"Final Recall Score:     {history_score[-1]:.4f}")
print(f"Final Memory Error:     {history_err[-1]:.4f}")
print(f"Conscious steps:        {sum(history_C)}/{STEPS} = {sum(history_C)/STEPS:.2%}")


# ============================================================
# 9. 作图输出
# ============================================================

out_dir = Path("recall_output")
out_dir.mkdir(exist_ok=True)

history_q = np.array(history_q)

plt.figure(figsize=(10, 6))
plt.plot(history_R, label="R synchronization")
plt.plot(history_Rc, label="Rc(Hq) dynamic threshold", linestyle="--")
plt.plot(history_Phi, label="Phi integration")
plt.plot(history_S, label="S qualia-field")
plt.plot(history_Hq, label="Hq qualia entropy")
plt.plot(history_C, label="C consciousness", linestyle=":")
plt.xlabel("time step")
plt.ylabel("value")
plt.title("Recall Simulation: Order Parameters")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "recall_order_parameters.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
for i, name in enumerate(H_NAMES):
    plt.plot(history_q[:, i], label=name)
plt.xlabel("time step")
plt.ylabel("H-type proportion")
plt.title("Recall Simulation: H-Type Distribution Trajectory")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "recall_q_trajectory.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history_score, label="Recall score")
plt.plot(history_err, label="Memory error")
plt.xlabel("time step")
plt.ylabel("value")
plt.title("Recall Accuracy and Distortion")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "recall_error.png", dpi=220)
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(np.angle(psi), cmap="twilight", interpolation="nearest")
plt.colorbar(label="phase")
plt.title("Final Phase Field After Recall")
plt.axis("off")
plt.tight_layout()
plt.savefig(out_dir / "final_phase_field.png", dpi=220)
plt.close()

decoded_final, _ = nearest_H_decode(psi)
plt.figure(figsize=(7, 6))
plt.imshow(decoded_final, interpolation="nearest")
plt.colorbar(ticks=[0, 1, 2, 3], label="decoded H type")
plt.title("Final Decoded H-Type Field")
plt.axis("off")
plt.tight_layout()
plt.savefig(out_dir / "final_H_decoding.png", dpi=220)
plt.close()

print(f"\nFigures saved to: {out_dir.resolve()}")
