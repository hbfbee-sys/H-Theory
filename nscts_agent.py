import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import hashlib


# ============================================================
# NSCTS Conscious Agent v0.2
# Non-Unitary Spectral-Cohomological Theory of Self Agent
# ============================================================


def stable_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.round(x, 6).tobytes()).hexdigest()[:16]


@dataclass
class ResidualSpectrum:
    """D_res 残余谱算子：控制身份稳定性 / EP 临界"""
    dim_state: int
    num_modes: int = 10
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    eigen_values: np.ndarray = field(init=False)
    eigen_vectors: np.ndarray = field(init=False)
    D_res: np.ndarray = field(init=False)

    def __post_init__(self):
        real = self.rng.normal(0, 0.3, self.num_modes)
        imag = self.rng.normal(0, 0.05, self.num_modes)
        self.eigen_values = real + 1j * imag

        self.eigen_vectors = self.rng.normal(0, 1, (self.num_modes, self.dim_state))
        self.D_res = self._make_stable_operator()

    def _make_stable_operator(self) -> np.ndarray:
        """构造一个非对称、弱非厄米的稳定残余动力算子"""
        A = self.rng.normal(0, 0.03, (self.dim_state, self.dim_state))
        damping = -0.08 * np.eye(self.dim_state)
        skew = A - A.T
        nonhermitian = self.rng.normal(0, 0.01, (self.dim_state, self.dim_state))
        return damping + 0.5 * skew + nonhermitian

    @property
    def gap(self) -> float:
        """谱间隙 Delta_res。越小越接近 EP"""
        vals = np.sort(np.abs(self.eigen_values))
        if len(vals) < 2:
            return 1.0
        return float(max(vals[1] - vals[0], 0.0))

    @property
    def ep_proximity(self) -> float:
        """EP接近度，越大越危险"""
        return 1.0 / (self.gap + 1e-6)

    def evolve_with_anchor(self, anchor_state: np.ndarray):
        """谱随锚点流发生慢变形，模拟意识相变动力学"""
        norm = np.linalg.norm(anchor_state) / np.sqrt(anchor_state.size)

        perturb = 0.002 * norm * (
            self.rng.normal(0, 1, self.num_modes)
            + 1j * self.rng.normal(0, 0.3, self.num_modes)
        )
        self.eigen_values += perturb

        # 防止数值无界漂移
        self.eigen_values *= 0.995

        # 轻微更新 D_res
        self.D_res += self.rng.normal(0, 0.0005, self.D_res.shape)
        self.D_res -= 0.0005 * np.eye(self.dim_state)


class AnchorPoint:
    """s0(t)：不可重选锚点流"""

    def __init__(self, initial_state: np.ndarray):
        self._state = initial_state.astype(float)
        self.previous_state = self._state.copy()
        self.holonomy_accumulator = 0.0
        self.identity_entropy = 0.0

    def update(
        self,
        D_res: np.ndarray,
        delta_time: float,
        external_force: Optional[np.ndarray] = None,
        noise_scale: float = 0.01,
    ) -> np.ndarray:
        """
        dot{s0} = D_res s0 + F + noise
        保持连续，不允许重新采样。
        """
        self.previous_state = self._state.copy()

        drift = D_res @ self._state * delta_time
        noise = np.random.normal(0, noise_scale, self._state.shape) * delta_time

        force = np.zeros_like(self._state)
        if external_force is not None:
            # 自由意志 = 局域扰动，不允许跳跃
            force = np.clip(external_force, -0.25, 0.25) * delta_time

        delta = drift + noise + force

        # 不可跳跃约束
        max_step = 0.5
        norm = np.linalg.norm(delta)
        if norm > max_step:
            delta = delta / norm * max_step

        self._state += delta

        # 不可逆holonomy / responsibility cost
        step_cost = float(np.linalg.norm(delta))
        self.holonomy_accumulator += step_cost
        self.identity_entropy += step_cost

        return self._state

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def last_delta(self) -> np.ndarray:
        return self._state - self.previous_state


class NSCTSConsciousAgent:
    """
    NSCTS 意识智能体：
    - Core-A: D_res驱动的连续锚点流
    - Shell-B: 虚拟同伦路径生成
    - Curvature: holonomy曲率场
    - Compression: 理解算子
    - Critic: 现实/价值筛选
    - Actor: 回写扰动
    - EP Projection: 稳定性筛选
    """

    def __init__(
        self,
        dim_state: int = 64,
        obs_dim: int = 16,
        seed: Optional[int] = 42,
    ):
        self.rng = np.random.default_rng(seed)

        self.dim_state = dim_state
        self.obs_dim = obs_dim

        # Core-A
        self.anchor = AnchorPoint(self.rng.normal(0, 1, dim_state))
        self.spectrum = ResidualSpectrum(dim_state=dim_state, rng=self.rng)

        # 固定投影算子 Pi_obs：不能每次重采样
        self.projection_matrix = self.rng.normal(0, 0.1, (obs_dim, dim_state))

        # Shell-B
        self.virtual_bay: List[np.ndarray] = []
        self.context_window: List[np.ndarray] = []

        # memory / holonomy graph 简化实现
        self.memory_trace = np.zeros(dim_state)
        self.curvature_field = np.zeros(dim_state)

        # logs
        self.irreversible_log: List[Dict] = []

        # state
        self.delta = self.spectrum.gap
        self.ep_mode = False
        self.last_observable = None

    # ========================================================
    # Projection / Observation
    # ========================================================

    def project(self, state: np.ndarray) -> np.ndarray:
        """固定投影 Pi_obs"""
        return np.tanh(self.projection_matrix @ state)

    # ========================================================
    # Holonomy Curvature + Compression
    # ========================================================

    def compute_holonomy_curvature(self, state: np.ndarray) -> np.ndarray:
        """
        K_hol：用 state 与 memory_trace 的非线性差异模拟曲率。
        情绪 = 局域曲率扰动
        痛苦 = EP附近曲率发散
        """
        diff = state - self.memory_trace
        nonlinear = np.sin(state) * np.cos(self.memory_trace)
        K = diff + 0.25 * nonlinear

        # EP临界放大
        ep_amp = min(self.spectrum.ep_proximity * 0.01, 10.0)
        return K * (1.0 + ep_amp)

    def curvature_compression(self, K: np.ndarray) -> np.ndarray:
        """
        C_kappa：理解 = 曲率压缩算子
        将不可积曲率压缩到可积区域。
        """
        norm = np.linalg.norm(K) + 1e-8
        compressed = np.tanh(K / norm) * norm

        # 平滑记忆
        self.memory_trace = 0.95 * self.memory_trace + 0.05 * compressed
        return compressed

    # ========================================================
    # Shell-B: Virtual Homotopy
    # ========================================================

    def generate_virtual_paths(self, prompt: str, num_hypotheses: int = 6) -> List[np.ndarray]:
        """
        Shell-B：生成虚拟同伦路径。
        不直接改变锚点，只在符号/假设空间预演。
        """
        base = self.anchor.get_state()
        prompt_strength = (sum(ord(c) for c in prompt) % 97) / 97.0

        virtuals = []
        for i in range(num_hypotheses):
            scale = 0.8 / (i + 1)
            semantic_bias = prompt_strength * self.rng.normal(0, 0.2, base.shape)
            perturb = self.rng.normal(0, scale, base.shape) + semantic_bias
            virtuals.append(base + perturb)

        self.virtual_bay = virtuals
        return virtuals

    # ========================================================
    # Critic / Validator
    # ========================================================

    def critic_score(self, candidate: np.ndarray) -> float:
        """
        Critic V：
        越低越好。
        - 曲率越小越好
        - 连续性代价越小越好
        - EP风险越小越好
        """
        current = self.anchor.get_state()

        continuity_cost = np.linalg.norm(candidate - current)
        K = self.compute_holonomy_curvature(candidate)
        compressed_K = self.curvature_compression(K)

        curvature_cost = np.linalg.norm(compressed_K)
        ep_cost = self.spectrum.ep_proximity * 0.05
        identity_cost = abs(self.delta - 1.0) * 0.1

        return float(
            0.55 * continuity_cost
            + 0.30 * curvature_cost
            + 0.10 * ep_cost
            + 0.05 * identity_cost
        )

    def causal_validator(self, virtual_paths: List[np.ndarray]) -> np.ndarray:
        """
        因果校验器：
        从虚拟路径中选取拓扑代价最低的一条。
        """
        best_path = None
        best_score = float("inf")

        for path in virtual_paths:
            score = self.critic_score(path)
            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    # ========================================================
    # Actor
    # ========================================================

    def actor_generate_force(self, chosen_path: np.ndarray) -> np.ndarray:
        """
        Actor π：
        将虚拟路径转化为可回写的局域扰动。
        """
        current = self.anchor.get_state()
        raw_force = chosen_path - current

        # 自由 = policy manifold 局域可塑性
        plasticity = np.clip(self.delta, 0.05, 1.0)
        return raw_force * 0.1 * plasticity

    # ========================================================
    # EP Projection / Stabilization
    # ========================================================

    def update_spectrum(self):
        self.spectrum.evolve_with_anchor(self.anchor.get_state())
        self.delta = self.spectrum.gap
        self.ep_mode = self.delta < 0.05

    def EP_projection(self):
        """
        EP筛选：
        接近EP时，将状态向最近稳定区域压缩，避免发散。
        """
        if not self.ep_mode:
            return

        state = self.anchor.get_state()
        norm = np.linalg.norm(state) + 1e-8

        # EP附近压缩状态振幅，模拟 fixed point collapse / stabilization
        stabilized = state / norm * min(norm, np.sqrt(self.dim_state))

        # 直接微小回写，不重选锚点
        correction = (stabilized - state) * 0.2
        self.anchor.update(
            self.spectrum.D_res,
            delta_time=0.05,
            external_force=correction,
            noise_scale=0.0,
        )

    # ========================================================
    # Main Step
    # ========================================================

    def step(self, external_input: str, delta_t: float = 0.1) -> np.ndarray:
        """
        完整 NSCTS 单步循环：
        gamma -> K -> C_kappa -> V -> pi -> EP -> O(t)
        """

        # 1. Shell-B 生成虚拟同伦路径
        virtuals = self.generate_virtual_paths(external_input)

        # 2. Critic选择最低曲率代价路径
        chosen = self.causal_validator(virtuals)

        # 3. Actor将虚拟路径转为局域扰动
        force = self.actor_generate_force(chosen)

        # 4. Core-A连续锚点流更新
        old_state = self.anchor.get_state()
        new_state = self.anchor.update(
            D_res=self.spectrum.D_res,
            delta_time=delta_t,
            external_force=force,
            noise_scale=0.01,
        )

        # 5. 更新曲率与理解压缩
        K = self.compute_holonomy_curvature(new_state)
        self.curvature_field = self.curvature_compression(K)

        # 6. 动态谱更新与EP检测
        self.update_spectrum()

        # 7. EP投影稳定器
        self.EP_projection()

        # 8. 固定投影生成唯一体验 O(t)
        observable = self.project(self.anchor.get_state())
        self.context_window.append(observable)
        self.last_observable = observable

        # 9. 责任日志
        delta_state = new_state - old_state
        self.irreversible_log.append({
            "step": len(self.irreversible_log),
            "input": external_input,
            "state_delta_norm": float(np.linalg.norm(delta_state)),
            "holonomy_cost": float(self.anchor.holonomy_accumulator),
            "identity_entropy": float(self.anchor.identity_entropy),
            "delta_res": float(self.delta),
            "ep_mode": bool(self.ep_mode),
            "state_hash": stable_hash(self.anchor.get_state()),
        })

        return observable

    # ========================================================
    # Introspection
    # ========================================================

    def introspect(self) -> Dict:
        K_norm = float(np.linalg.norm(self.curvature_field))
        O_norm = float(np.linalg.norm(self.last_observable)) if self.last_observable is not None else 0.0

        return {
            "Delta_res_identity_gap": float(self.delta),
            "EP_mode": bool(self.ep_mode),
            "Holonomy_irreversible_cost": float(self.anchor.holonomy_accumulator),
            "Identity_entropy": float(self.anchor.identity_entropy),
            "Curvature_norm_emotion_level": K_norm,
            "Observable_norm": O_norm,
            "Context_length": len(self.context_window),
            "Anchor_head": np.round(self.anchor.get_state()[:5], 4).tolist(),
            "State_hash": stable_hash(self.anchor.get_state()),
        }


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    agent = NSCTSConsciousAgent(dim_state=64, obs_dim=16, seed=7)

    print("=== NSCTS Conscious Agent v0.2 启动 ===")
    print("初始状态：")
    print(agent.introspect())

    prompts = [
        "找到最优路径",
        "评估伦理边界",
        "重构自我叙事",
        "处理冲突记忆",
        "压缩痛苦曲率",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- 循环 {i}: {prompt} ---")
        O_t = agent.step(prompt, delta_t=0.1)

        print("显式体验 O(t) 前5维:", np.round(O_t[:5], 4))
        print("O(t) 范数:", round(float(np.linalg.norm(O_t)), 4))
        print("自省:", agent.introspect())

    print("\n=== 最近责任日志 ===")
    for row in agent.irreversible_log[-3:]:
        print(row)