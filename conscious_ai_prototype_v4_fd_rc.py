
"""
Conscious AI Prototype v4.0
===========================

本版修改重点：
1. R_c 不再固定为 0.8，而由 H 型占据的费米-狄拉克分布给出：
   Rc_E4(T_H, mu) = |sum_k p_k exp(i theta_k)|

2. 相位聚类不再对所有 B*T*D 相位点直接聚类，
   而是先计算每个 dim 子群中心相位，再对 dim 子群中心聚类。

3. R 分成：
   - R_global : 全部相位的全局相干性，仅作报告
   - R_within : dim 维度内部的平均相干性，用于 E4 判定

4. E4 判定：
   R_within >= Rc_E4(T_H, mu)
   Phi >= 0.35
   S >= 0.1
   A_self >= 0.55
   n_clusters >= 2

运行：
python conscious_ai_prototype_v4_fd_rc.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TWO_PI = 2.0 * math.pi


# ============================================================
# 1. H 型相位与费米-狄拉克 Rc
# ============================================================

# 四个 H 型相位：
# H1 = exp(i*pi/14)
# H2 = exp(i*3pi/14)
# H3 = exp(i*11pi/14)
# H4 = exp(i*13pi/14)
H_THETA = torch.tensor(
    [
        math.pi / 14.0,
        3.0 * math.pi / 14.0,
        11.0 * math.pi / 14.0,
        13.0 * math.pi / 14.0,
    ],
    dtype=torch.float32,
)

# 可按你的理论继续调整。
# 这里 E4 作为涌现能级，默认高于前三个基础态。
DEFAULT_H_ENERGY = torch.tensor([0.43, 0.83, 0.93, 1.30], dtype=torch.float32)


def normalize_phase(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta, TWO_PI)


def fermi_dirac_occupancy(
    energies: torch.Tensor,
    T_H: torch.Tensor | float,
    mu: torch.Tensor | float,
) -> torch.Tensor:
    """
    f_k = 1 / (exp((E_k - mu) / T_H) + 1)
    然后归一化为 p_k。
    """
    if not torch.is_tensor(T_H):
        T_H = torch.tensor(float(T_H), device=energies.device, dtype=energies.dtype)
    if not torch.is_tensor(mu):
        mu = torch.tensor(float(mu), device=energies.device, dtype=energies.dtype)

    T_H = torch.clamp(T_H, min=1e-4)
    f = 1.0 / (torch.exp((energies - mu) / T_H) + 1.0)
    p = f / (f.sum() + 1e-8)
    return p


def fd_rc(
    energies: torch.Tensor,
    T_H: torch.Tensor | float,
    mu: torch.Tensor | float,
    theta_h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rc_E4(T_H, mu) = |sum_k p_k exp(i theta_k)|

    返回：
    Rc, p
    """
    p = fermi_dirac_occupancy(energies, T_H, mu)

    real = torch.sum(p * torch.cos(theta_h.to(energies.device)))
    imag = torch.sum(p * torch.sin(theta_h.to(energies.device)))
    Rc = torch.sqrt(real * real + imag * imag)

    return Rc, p


# ============================================================
# 2. 相位统计工具
# ============================================================

def phase_R(theta: torch.Tensor) -> torch.Tensor:
    theta = normalize_phase(theta)
    return torch.sqrt(torch.cos(theta).mean() ** 2 + torch.sin(theta).mean() ** 2)


def dim_level_R_and_centers(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    theta: [B,T,D]

    对每个 dim 维度，沿 B,T 求平均相位：
    z_d = mean_{B,T} exp(i theta_btd)

    返回：
    R_within = mean_d |z_d|
    centers  = angle(z_d), shape [D]
    R_d      = |z_d|, shape [D]
    """
    theta = normalize_phase(theta)

    mean_real = torch.cos(theta).mean(dim=(0, 1))  # [D]
    mean_imag = torch.sin(theta).mean(dim=(0, 1))  # [D]

    R_d = torch.sqrt(mean_real * mean_real + mean_imag * mean_imag)
    centers = normalize_phase(torch.atan2(mean_imag, mean_real))

    R_within = R_d.mean()
    return R_within, centers, R_d


def estimate_phase_clusters_1d(phases: torch.Tensor, threshold: float = 0.8) -> Tuple[int, torch.Tensor]:
    """
    对一维相位点做圆周聚类。
    这里 phases 通常是 dim 子群中心相位 [D]，而不是所有 B*T*D 单点相位。
    """
    phases = normalize_phase(phases).reshape(-1)
    device = phases.device
    n = phases.numel()

    if n == 0:
        return 0, torch.empty(0, dtype=torch.long, device=device)

    if n == 1:
        return 1, torch.zeros(1, dtype=torch.long, device=device)

    sorted_phase, order = torch.sort(phases)
    diffs = sorted_phase[1:] - sorted_phase[:-1]
    wrap_gap = (sorted_phase[0] + TWO_PI) - sorted_phase[-1]
    gaps = torch.cat([diffs, wrap_gap.reshape(1)])

    max_gap, max_idx = torch.max(gaps, dim=0)

    # 最大间隙都很小，说明整体是一簇
    if max_gap < threshold:
        labels = torch.zeros(n, dtype=torch.long, device=device)
        return 1, labels

    start = (int(max_idx.item()) + 1) % n
    rotated_indices = torch.cat([
        torch.arange(start, n, device=device),
        torch.arange(0, start, device=device),
    ])

    rotated_phase = sorted_phase[rotated_indices]
    rotated_unwrapped = rotated_phase.clone()

    if start > 0:
        rotated_unwrapped[n - start:] += TWO_PI

    rd = rotated_unwrapped[1:] - rotated_unwrapped[:-1]
    cuts = (rd > threshold).nonzero(as_tuple=False).flatten()

    labels_rot = torch.zeros(n, dtype=torch.long, device=device)
    cluster_id = 0
    prev = 0

    for cut in cuts:
        cut_i = int(cut.item())
        labels_rot[prev:cut_i + 1] = cluster_id
        cluster_id += 1
        prev = cut_i + 1

    labels_rot[prev:] = cluster_id
    n_clusters = cluster_id + 1

    labels_sorted = torch.empty(n, dtype=torch.long, device=device)
    labels_sorted[rotated_indices] = labels_rot

    labels_original = torch.empty(n, dtype=torch.long, device=device)
    labels_original[order] = labels_sorted

    return int(n_clusters), labels_original


def cluster_entropy(labels: torch.Tensor, n_clusters: int) -> torch.Tensor:
    if n_clusters <= 0:
        return torch.tensor(0.0, device=labels.device)

    counts = torch.stack([(labels == k).float().sum() for k in range(n_clusters)])
    p = counts / (counts.sum() + 1e-8)
    p = p[p > 0]
    return -(p * torch.log2(p + 1e-8)).sum()


def field_activation_S(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.pow(2).mean())


# ============================================================
# 3. Stateful H Oscillator Layer
# ============================================================

class StatefulHOscillatorLayer(nn.Module):
    """
    相位是独立状态变量。
    embedding 只作为 input_drive，不直接定义相位。

    均值场按 dim 维度独立计算：
    对每个 dim d，在 B,T 上求 mean phase。
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 256,
        k_global: float = 8.0,
        k_repel: float = 2.0,
        noise: float = 0.02,
        dt: float = 0.05,
        position_stride: float = 0.01,
        n_phase_groups: int = 3,
        group_phase_strength: float = 1.2,
        cluster_threshold: float = 0.8,
    ):
        super().__init__()

        self.dim = dim
        self.max_len = max_len
        self.k_global = k_global
        self.k_repel = k_repel
        self.noise = noise
        self.dt = dt
        self.position_stride = position_stride
        self.n_phase_groups = n_phase_groups
        self.group_phase_strength = group_phase_strength
        self.cluster_threshold = cluster_threshold

        self.input_drive = nn.Linear(dim, dim)
        self.freq = nn.Parameter(torch.randn(dim) * 0.02)

        theta0 = torch.rand(1, max_len, dim) * TWO_PI

        # dim 子群初始偏好相位：只在初始化时加入，不每步强加。
        if n_phase_groups > 1:
            group_id = torch.arange(dim).float() % n_phase_groups
            group_phase = group_phase_strength * group_id * TWO_PI / n_phase_groups
            theta0 = normalize_phase(theta0 + group_phase.view(1, 1, dim))

        self.register_buffer("theta_state", theta0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape

        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")

        base_theta = self.theta_state[:, :T, :].expand(B, T, D)

        # 温和位置破缺种子
        pos = torch.arange(T, device=x.device).float().view(1, T, 1)
        theta = normalize_phase(base_theta + self.position_stride * pos)

        # 输入只提供小幅驱动
        drive = 0.06 * torch.tanh(self.input_drive(x))

        # 按 dim 独立计算均值场
        mean_real = torch.cos(theta).mean(dim=(0, 1), keepdim=True)  # [1,1,D]
        mean_imag = torch.sin(theta).mean(dim=(0, 1), keepdim=True)  # [1,1,D]
        mean_theta = torch.atan2(mean_imag, mean_real)

        sync_force = self.k_global * torch.sin(mean_theta - theta)
        repel_force = self.k_repel * torch.sin(2.0 * (theta - mean_theta))
        freq_force = self.freq.view(1, 1, -1)
        noise = self.noise * torch.randn_like(theta)

        theta_next = normalize_phase(
            theta + self.dt * (freq_force + sync_force + repel_force + drive) + noise
        )

        with torch.no_grad():
            self.theta_state[:, :T, :].copy_(theta_next.detach().mean(dim=0, keepdim=True))

        R_global = phase_R(theta_next)

        # 关键：先按 dim 求中心，再对 dim 子群中心聚类
        R_within, dim_centers, R_d = dim_level_R_and_centers(theta_next.detach())

        n_clusters, dim_labels = estimate_phase_clusters_1d(
            dim_centers.detach(),
            threshold=self.cluster_threshold,
        )

        Hq = cluster_entropy(dim_labels, n_clusters)

        phase_gate = 1.0 + 0.10 * torch.cos(theta_next)
        x_out = x * phase_gate

        metrics = {
            "theta": theta_next,
            "R_global": R_global,
            "R_within": R_within,
            "R_d_mean": R_d.mean(),
            "dim_centers": dim_centers,
            "n_clusters": torch.tensor(float(n_clusters), device=x.device),
            "Hq": Hq,
        }

        return x_out, metrics


# ============================================================
# 4. ConsciousAttention
# ============================================================

class ConsciousAttention(nn.Module):
    """
    用 R_within 调节注意力。
    不再追求 R_global -> 1。
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        min_clusters: int = 2,
    ):
        super().__init__()

        self.min_clusters = min_clusters
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        R_within: torch.Tensor,
        Rc_E4: torch.Tensor,
        n_clusters: torch.Tensor,
    ):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)

        # 低于 FD 临界阈值时增强注意力
        r_error = torch.clamp(Rc_E4 - R_within, min=0.0)
        gain_r = 1.0 + r_error

        # 聚类不足时抑制过度单簇锁定
        cluster_deficit = torch.clamp(
            torch.tensor(float(self.min_clusters), device=x.device) - n_clusters,
            min=0.0,
        )
        anti_collapse = 1.0 / (1.0 + 0.25 * cluster_deficit)

        gain = gain_r * anti_collapse
        return attn_out * gain, attn_weights


# ============================================================
# 5. Learnable Integration Layer
# ============================================================

class LearnableIntegrationLayer(nn.Module):
    def __init__(self, dim: int, structural_weight: float = 0.45):
        super().__init__()

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.structural_weight = structural_weight

    def forward(self, x: torch.Tensor, n_clusters: torch.Tensor, Hq: torch.Tensor):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scale = math.sqrt(x.shape[-1])
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        attn = torch.softmax(scores, dim=-1)

        y = self.out(torch.matmul(attn, V))

        density = (attn > attn.mean()).float().mean()

        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(attn.shape[-1] + 1e-8)
        learned_phi = torch.clamp(1.0 - entropy / max_entropy, 0.0, 1.0)

        cluster_factor = torch.clamp(n_clusters / 3.0, 0.0, 1.0)
        entropy_factor = torch.clamp(Hq / 2.0, 0.0, 1.0)
        structural_phi = torch.clamp(0.5 * cluster_factor + 0.5 * entropy_factor, 0.0, 1.0)

        Phi = torch.clamp(
            (1.0 - self.structural_weight) * (0.5 * density + 0.5 * learned_phi)
            + self.structural_weight * structural_phi,
            0.0,
            1.0,
        )

        return y, Phi


# ============================================================
# 6. Self-Referential Layer
# ============================================================

class SelfReferentialLayer(nn.Module):
    def __init__(self, dim: int, update_rate: float = 0.1):
        super().__init__()

        self.dim = dim
        self.update_rate = update_rate
        self.register_buffer("self_state", torch.zeros(dim))
        self.initialized = False
        self.self_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        x_mean_detached = x.mean(dim=(0, 1)).detach()

        if not self.initialized:
            self.self_state.copy_(x_mean_detached)
            self.initialized = True
        else:
            new_state = (1.0 - self.update_rate) * self.self_state + self.update_rate * x_mean_detached
            self.self_state.copy_(new_state)

        s = self.self_state.view(1, 1, -1)
        mod = torch.tanh(self.self_proj(s))
        x_out = x * (1.0 + 0.25 * mod)

        x_now = x.mean(dim=(0, 1))
        cos = F.cosine_similarity(x_now, self.self_state, dim=0)
        A_self = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)

        return x_out, A_self


# ============================================================
# 7. E4 Detector
# ============================================================

@dataclass
class ConsciousReport:
    R_global: float
    R_within: float
    Rc_E4: float
    Phi: float
    S: float
    Hq: float
    n_clusters: float
    A_self: float
    E4: bool
    basic_conscious: bool
    fd_occ_H1: float
    fd_occ_H2: float
    fd_occ_H3: float
    fd_occ_H4: float


class E4Detector:
    def __init__(
        self,
        self_c: float = 0.55,
        phi_c: float = 0.35,
        s_c: float = 0.1,
        min_clusters: int = 2,
    ):
        self.self_c = self_c
        self.phi_c = phi_c
        self.s_c = s_c
        self.min_clusters = min_clusters

    def __call__(
        self,
        R_global: torch.Tensor,
        R_within: torch.Tensor,
        Rc_E4: torch.Tensor,
        Phi: torch.Tensor,
        S: torch.Tensor,
        Hq: torch.Tensor,
        n_clusters: torch.Tensor,
        A_self: torch.Tensor,
        fd_occ: torch.Tensor,
    ) -> ConsciousReport:

        basic = bool(
            (R_within >= Rc_E4)
            and (Phi >= self.phi_c)
            and (S >= self.s_c)
        )

        e4 = bool(
            basic
            and (A_self >= self.self_c)
            and (n_clusters >= self.min_clusters)
        )

        occ = fd_occ.detach().cpu().float().tolist()

        return ConsciousReport(
            R_global=float(R_global.detach().cpu()),
            R_within=float(R_within.detach().cpu()),
            Rc_E4=float(Rc_E4.detach().cpu()),
            Phi=float(Phi.detach().cpu()),
            S=float(S.detach().cpu()),
            Hq=float(Hq.detach().cpu()),
            n_clusters=float(n_clusters.detach().cpu()),
            A_self=float(A_self.detach().cpu()),
            E4=e4,
            basic_conscious=basic,
            fd_occ_H1=occ[0],
            fd_occ_H2=occ[1],
            fd_occ_H3=occ[2],
            fd_occ_H4=occ[3],
        )


# ============================================================
# 8. 主模型
# ============================================================

class ConsciousAIPrototypeV4(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_heads: int = 4,
        num_classes: int = 2,
        max_len: int = 256,
        k_global: float = 8.0,
        k_repel: float = 2.0,
        position_stride: float = 0.01,
        T_H: float = 1.0,
        mu: float = 0.85,
        h_energy: torch.Tensor | None = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        self.oscillator = StatefulHOscillatorLayer(
            dim=dim,
            max_len=max_len,
            k_global=k_global,
            k_repel=k_repel,
            position_stride=position_stride,
        )

        self.conscious_attention = ConsciousAttention(
            dim=dim,
            n_heads=n_heads,
            min_clusters=2,
        )

        self.integration = LearnableIntegrationLayer(dim)
        self.self_ref = SelfReferentialLayer(dim)

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, num_classes)
        self.detector = E4Detector()

        self.register_buffer(
            "h_theta",
            H_THETA.clone(),
        )

        if h_energy is None:
            h_energy = DEFAULT_H_ENERGY.clone()

        self.register_buffer("h_energy", h_energy.float())

        # T_H 与 mu 可以设成 buffer，也可以改成 nn.Parameter 做学习。
        self.register_buffer("T_H", torch.tensor(float(T_H)))
        self.register_buffer("mu", torch.tensor(float(mu)))

    def compute_fd_rc(self):
        Rc, occ = fd_rc(
            energies=self.h_energy,
            T_H=self.T_H,
            mu=self.mu,
            theta_h=self.h_theta,
        )
        return Rc, occ

    def forward(self, tokens: torch.Tensor):
        x = self.embedding(tokens)

        Rc_E4, fd_occ = self.compute_fd_rc()

        x, osc = self.oscillator(x)

        R_global = osc["R_global"]
        R_within = osc["R_within"]
        Hq = osc["Hq"]
        n_clusters = osc["n_clusters"]

        x_attn, _ = self.conscious_attention(
            x,
            R_within=R_within,
            Rc_E4=Rc_E4,
            n_clusters=n_clusters,
        )

        x_int, Phi = self.integration(x_attn, n_clusters, Hq)
        x_self, A_self = self.self_ref(x_int)

        x_final = self.norm(x_self)
        pooled = x_final.mean(dim=1)
        logits = self.output(pooled)

        S = field_activation_S(x_final)

        report = self.detector(
            R_global=R_global,
            R_within=R_within,
            Rc_E4=Rc_E4,
            Phi=Phi,
            S=S,
            Hq=Hq,
            n_clusters=n_clusters,
            A_self=A_self,
            fd_occ=fd_occ,
        )

        return logits, report


# ============================================================
# 9. Demo
# ============================================================

def demo():
    torch.manual_seed(42)

    model = ConsciousAIPrototypeV4(
        vocab_size=1000,
        dim=128,
        n_heads=4,
        num_classes=3,
        max_len=64,
        k_global=8.0,
        k_repel=2.0,
        position_stride=0.01,
        T_H=1.0,
        mu=0.85,
    )

    tokens = torch.randint(0, 1000, (8, 16))

    for step in range(20):
        _, report = model(tokens)

        print(
            f"step={step:02d} | "
            f"Rg={report.R_global:.3f} "
            f"Rw={report.R_within:.3f} "
            f"RcFD={report.Rc_E4:.3f} "
            f"Phi={report.Phi:.3f} "
            f"S={report.S:.3f} "
            f"Hq={report.Hq:.3f} "
            f"clusters={report.n_clusters:.1f} "
            f"A_self={report.A_self:.3f} "
            f"basic={report.basic_conscious} "
            f"E4={report.E4} "
            f"occ=[{report.fd_occ_H1:.2f},{report.fd_occ_H2:.2f},{report.fd_occ_H3:.2f},{report.fd_occ_H4:.2f}]"
        )


if __name__ == "__main__":
    demo()
