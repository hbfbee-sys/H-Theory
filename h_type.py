from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


EMPTY = None
TEMP_4 = "4*"
Grid = List[List[Optional[object]]]
Pos = Tuple[int, int]
PHASE_CENTER = math.pi / 2
PHASE_STEP = math.pi / 7
ANGLE_TOLERANCE = 1e-9
DIGIT_AMPLITUDE = 1.0
MEMBRANE_THRESHOLD = math.pi
WEAK_PHASE_INCREMENT = math.pi / 3
STRONG_PHASE_INCREMENT = math.pi
INHIBITORY_PHASE_INCREMENT = -math.pi
AMPLITUDE_THRESHOLD = 1.0
WEAK_AMPLITUDE_INCREMENT = 1.0 / 3.0
STRONG_AMPLITUDE_INCREMENT = 1.0
INHIBITORY_AMPLITUDE_INCREMENT = -1.0


class Coupling(str, Enum):
    INVALID = "invalid"
    NOISE = "noise"
    WEAK = "weak_coupling"
    STRONG = "strong_coupling"
    STABLE = "stable_structure"
    FUNCTIONAL_STABLE = "functional_stable_path"
    INHIBITORY = "inhibitory"


class Signal(str, Enum):
    WEAK = "weak"
    STRONG = "strong"
    INHIBITORY = "inhibitory"


class ExternalStimulus(str, Enum):
    NONE = "none"
    WEAK = "weak_stimulus"
    STRONG = "strong_stimulus"
    INHIBITORY = "inhibitory_stimulus"


class Outcome(str, Enum):
    NONE = "none"
    NOISE = "neural_noise"
    SUBTHRESHOLD = "subthreshold_accumulation"
    LOCAL = "local_activation"
    MAIN = "main_path_transmission"
    INHIBITED = "inhibited"


class StressResponse(str, Enum):
    NONE = "none"
    LOCAL = "local_response"
    REFLEX = "reflex_response"


class Phase(str, Enum):
    READY = "ready"
    T0 = "T0_input"
    T1 = "T1_flip"
    T2 = "T2_recover"
    T3 = "T3_output_gate"
    R0 = "R0_output_state"
    R1 = "R1_reset_flip"
    R2 = "R2_restore"
    R3 = "R3_ready"


class ReflexRole(str, Enum):
    RECEPTOR = "receptor"
    SENSORY = "sensory"
    INTERNEURON = "interneuron"
    MOTOR = "motor"
    EFFECTOR = "effector"


class ThreeStageTemplate(str, Enum):
    WEAK_ACCUMULATION = "weak_accumulation"
    LOCAL_ACTIVATION = "local_activation"
    LONG_WEAK_RECONSTRUCTION = "strong_driven_long_weak_reconstruction"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


STABLE_STATES: Dict[str, Grid] = {
    "A": [[2, EMPTY, 1], [3, 4, 5], [7, EMPTY, 6]],
    "B": [[7, EMPTY, 6], [3, 4, 5], [2, EMPTY, 1]],
    "C": [[1, EMPTY, 2], [5, 4, 3], [6, EMPTY, 7]],
    "D": [[6, EMPTY, 7], [5, 4, 3], [1, EMPTY, 2]],
}

RIGHT_TOP_DIGIT_TO_STATE = {1: "A", 6: "B", 2: "C", 7: "D"}
STATE_TO_RIGHT_TOP_DIGIT = {state: digit for digit, state in RIGHT_TOP_DIGIT_TO_STATE.items()}
START_DIGIT_TO_COORD = {1: (0, 2), 2: (0, 0), 6: (2, 2), 7: (2, 0)}
START_COORD_TO_DIGIT = {coord: digit for digit, coord in START_DIGIT_TO_COORD.items()}
QUALIA_DIGITS = (1, 2, 6, 7)


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def complement(a: object, b: object) -> bool:
    return isinstance(a, int) and isinstance(b, int) and a + b == 8


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    if wrapped <= -math.pi:
        return wrapped + 2 * math.pi
    return wrapped


def phase_angle(value: int) -> float:
    """Return the phase angle for a digit.

    The mapping is centered at 4 -> i.  Complements x+y=8 satisfy
    theta_x + theta_y = pi, so z_x * z_y = -1.
    """
    if value not in range(1, 8):
        raise ValueError("phase digits must be in 1..7")
    return PHASE_CENTER + (value - 4) * PHASE_STEP


def phase_value(value: int) -> complex:
    return cmath.exp(1j * phase_angle(value))


def digit_amplitude(value: int) -> float:
    if value not in range(1, 8):
        raise ValueError("amplitude digits must be in 1..7")
    return DIGIT_AMPLITUDE


def digit_state(value: int, amplitude: Optional[float] = None) -> complex:
    """Return the amplitude-phase representation A_n*exp(i*theta_n)."""
    if amplitude is None:
        amplitude = digit_amplitude(value)
    return amplitude * phase_value(value)


def phase_complement(a: int, b: int, tolerance: float = ANGLE_TOLERANCE) -> bool:
    return abs(wrap_angle(phase_angle(a) + phase_angle(b) - math.pi)) <= tolerance


def amplitude_phase_complement(
    a: int,
    b: int,
    amplitude_tolerance: float = ANGLE_TOLERANCE,
    phase_tolerance: float = ANGLE_TOLERANCE,
) -> bool:
    return (
        phase_complement(a, b, phase_tolerance)
        and abs(digit_amplitude(a) - digit_amplitude(b)) <= amplitude_tolerance
    )


def direction_operator(direction: object) -> complex:
    if isinstance(direction, str):
        direction = Direction(direction)
    return {
        Direction.RIGHT: 1 + 0j,
        Direction.LEFT: -1 + 0j,
        Direction.UP: 0 + 1j,
        Direction.DOWN: 0 - 1j,
    }[direction]


def normalize_state(state: List[complex]) -> List[complex]:
    norm = math.sqrt(sum(abs(value) ** 2 for value in state))
    if norm == 0:
        raise ValueError("cannot normalize a zero state")
    return [value / norm for value in state]


def schrodinger_step(
    state: List[complex],
    hamiltonian: List[List[complex]],
    dt: float,
    hbar: float = 1.0,
    normalize: bool = True,
) -> List[complex]:
    """One explicit Euler step of i*hbar*d|psi>/dt = H|psi>.

    This is a small simulation helper, not a claim that the model is a
    physical quantum system.  The H rules are sampled phase dynamics; this
    function exposes the equivalent Schrodinger-form evolution.
    """
    size = len(state)
    if len(hamiltonian) != size or any(len(row) != size for row in hamiltonian):
        raise ValueError("hamiltonian dimensions must match the state")
    hpsi = [
        sum(hamiltonian[row][col] * state[col] for col in range(size))
        for row in range(size)
    ]
    evolved = [
        state[index] + (-1j * dt / hbar) * hpsi[index]
        for index in range(size)
    ]
    return normalize_state(evolved) if normalize else evolved


def body_flip_hamiltonian(omega: float = 1.0, hbar: float = 1.0) -> List[List[complex]]:
    """Two-state body Hamiltonian equivalent to 345 <-> 543."""
    return [[0j, hbar * omega], [hbar * omega, 0j]]


def complex_ginzburg_landau_step(
    field: List[complex],
    dt: float,
    alpha: float,
    omega: float,
    beta: float,
    gamma: float,
    diffusion: float,
) -> List[complex]:
    """One explicit step of dpsi/dt = (a+iw)psi-(b+ig)|psi|^2psi+D laplacian.

    This is the continuous parent dynamics used by the H model.  The discrete
    digit states are interpreted as symmetry-constrained sampled eigenstates of
    this amplitude-phase field.
    """
    size = len(field)
    if size == 0:
        return []
    evolved: List[complex] = []
    for index, value in enumerate(field):
        left = field[index - 1] if index > 0 else value
        right = field[index + 1] if index < size - 1 else value
        laplacian = left - 2 * value + right
        nonlinear = (beta + 1j * gamma) * (abs(value) ** 2) * value
        derivative = (alpha + 1j * omega) * value - nonlinear + diffusion * laplacian
        evolved.append(value + dt * derivative)
    return evolved


def h_state_from_corner_digit(digit: int, name: str = "h") -> "HNeuron":
    """Expand one right-top corner digit into the unique legal H state."""
    if digit not in RIGHT_TOP_DIGIT_TO_STATE:
        raise ValueError("right-top corner digit must be one of 1, 2, 6, 7")
    return HNeuron.from_state(name, RIGHT_TOP_DIGIT_TO_STATE[digit])


def corner_digit_from_h(neuron: "HNeuron") -> int:
    """Compress a legal H state to its right-top corner digit."""
    value = neuron.grid[0][2]
    if not isinstance(value, int) or value not in RIGHT_TOP_DIGIT_TO_STATE:
        raise ValueError("H state cannot be compressed by the right-top corner")
    return value


def nine_h_from_start_digit(digit: int) -> "NineHReflexArc":
    """Expand one corner digit into the unique standard nine-H reflex path."""
    if digit not in START_DIGIT_TO_COORD:
        raise ValueError("nine-H start digit must be one of 1, 2, 6, 7")
    return NineHReflexArc.standard(start_coord=START_DIGIT_TO_COORD[digit])


def start_digit_from_nine_h(arc: "NineHReflexArc") -> int:
    return START_COORD_TO_DIGIT[arc.start_coord]


def reduced_wave_step(
    state: complex,
    dt: float,
    alpha: float,
    omega: float,
    beta: float,
    gamma: float,
) -> complex:
    """Reduced one-mode wave equation for a compressed H or nine-H state.

    It is the zero-dimensional CGL/Stuart-Landau equation:
    dchi/dt = (alpha+i*omega)chi - (beta+i*gamma)|chi|^2 chi.
    """
    derivative = (alpha + 1j * omega) * state - (beta + 1j * gamma) * (abs(state) ** 2) * state
    return state + dt * derivative


def solve_reduced_wave(
    initial: complex,
    steps: int,
    dt: float,
    alpha: float,
    omega: float,
    beta: float,
    gamma: float,
) -> List[complex]:
    history = [initial]
    state = initial
    for _ in range(steps):
        state = reduced_wave_step(state, dt, alpha, omega, beta, gamma)
        history.append(state)
    return history


def qualia_state(digit: int, amplitude: float = 1.0) -> complex:
    """Return one of the four basic qualia states Q1,Q2,Q6,Q7."""
    if digit not in QUALIA_DIGITS:
        raise ValueError("qualia digit must be one of 1, 2, 6, 7")
    return digit_state(digit, amplitude)


@dataclass
class MemoryTrace:
    """Static memory encoded by the four basic qualia components."""

    components: Dict[int, complex]

    @classmethod
    def from_amplitudes(cls, amplitudes: Dict[int, float]) -> "MemoryTrace":
        return cls({digit: qualia_state(digit, amplitudes.get(digit, 0.0)) for digit in QUALIA_DIGITS})

    def field(self) -> complex:
        return sum(self.components.get(digit, 0j) for digit in QUALIA_DIGITS)

    def recall(self, context: Optional[Dict[int, complex]] = None) -> "MemoryTrace":
        """Reactivate memory under the current surrounding H-state context.

        context maps each qualia digit to a complex gain.  A neutral context is
        gain 1.  Non-neutral context changes amplitude and phase, so recall can
        differ from the original memory trace.
        """
        context = context or {}
        return MemoryTrace(
            {
                digit: self.components.get(digit, 0j) * context.get(digit, 1 + 0j)
                for digit in QUALIA_DIGITS
            }
        )

    def distortion(self, recalled: "MemoryTrace") -> float:
        return math.sqrt(
            sum(
                abs(recalled.components.get(digit, 0j) - self.components.get(digit, 0j)) ** 2
                for digit in QUALIA_DIGITS
            )
        )


def active_qualia_count(components: Dict[int, complex], threshold: float = ANGLE_TOLERANCE) -> int:
    return sum(1 for digit in QUALIA_DIGITS if abs(components.get(digit, 0j)) > threshold)


def qualia_field(components: Dict[int, complex]) -> complex:
    return sum(components.get(digit, 0j) for digit in QUALIA_DIGITS)


def qualia_phase_coherence(components: Dict[int, complex]) -> float:
    active = [components.get(digit, 0j) for digit in QUALIA_DIGITS if abs(components.get(digit, 0j)) > ANGLE_TOLERANCE]
    if not active:
        return 0.0
    unit_sum = sum(value / abs(value) for value in active)
    return abs(unit_sum) / len(active)


def consciousness_critical(
    components: Dict[int, complex],
    amplitude_threshold: float = 1.0,
    coherence_threshold: float = 0.0,
    min_qualia_diversity: int = 2,
) -> bool:
    """Return whether a qualia field crosses the model's consciousness threshold."""
    return (
        abs(qualia_field(components)) >= amplitude_threshold
        and qualia_phase_coherence(components) >= coherence_threshold
        and active_qualia_count(components) >= min_qualia_diversity
    )


@dataclass
class PhaseMembrane:
    """Continuous complex-amplitude accumulator for membrane potential.

    The membrane state is A*exp(i*phi).  Amplitude models accumulated membrane
    strength, while phase models synchronization.  The discrete weak-stage rule
    is a sampled form of this accumulator: three weak increments reach both
    default thresholds.
    """

    amplitude: float = 0.0
    phase: float = 0.0
    amplitude_threshold: float = AMPLITUDE_THRESHOLD
    phase_threshold: float = MEMBRANE_THRESHOLD

    def state(self) -> complex:
        return self.amplitude * cmath.exp(1j * self.phase)

    def potential(self) -> float:
        return self.amplitude

    def phase_potential(self) -> float:
        return self.phase

    def add(self, amplitude_increment: float, phase_increment: float = 0.0) -> bool:
        self.amplitude += amplitude_increment
        self.phase += phase_increment
        if self.amplitude < 0:
            self.amplitude = 0.0
        if self.phase < 0:
            self.phase = 0.0
        return self.ready()

    def ready(self) -> bool:
        return (
            self.amplitude >= self.amplitude_threshold - ANGLE_TOLERANCE
            and self.phase >= self.phase_threshold - ANGLE_TOLERANCE
        )

    def reset(self) -> None:
        self.amplitude = 0.0
        self.phase = 0.0

    def decay(
        self,
        amplitude_amount: float = WEAK_AMPLITUDE_INCREMENT,
        phase_amount: float = WEAK_PHASE_INCREMENT,
    ) -> None:
        self.amplitude = max(0.0, self.amplitude - amplitude_amount)
        self.phase = max(0.0, self.phase - phase_amount)


@dataclass
class HNeuron:
    name: str
    grid: Grid
    original: Grid = field(init=False)
    phase: Phase = Phase.READY
    weak_stage: int = 0
    local_stage: int = 0
    long_weak_stage: int = 0
    last_output: bool = False
    local_output: bool = False
    membrane: PhaseMembrane = field(default_factory=PhaseMembrane)
    refractory_ticks: int = 0

    def __post_init__(self) -> None:
        self.grid = clone_grid(self.grid)
        self.original = clone_grid(self.grid)

    @classmethod
    def from_state(cls, name: str, state: str) -> "HNeuron":
        return cls(name=name, grid=STABLE_STATES[state])

    def copy(self) -> "HNeuron":
        copied = HNeuron(self.name, self.grid)
        copied.original = clone_grid(self.original)
        copied.phase = self.phase
        copied.weak_stage = self.weak_stage
        copied.local_stage = self.local_stage
        copied.long_weak_stage = self.long_weak_stage
        copied.last_output = self.last_output
        copied.local_output = self.local_output
        copied.membrane = PhaseMembrane(
            self.membrane.amplitude,
            self.membrane.phase,
            self.membrane.amplitude_threshold,
            self.membrane.phase_threshold,
        )
        copied.refractory_ticks = self.refractory_ticks
        return copied

    def pos(self, value: int) -> Pos:
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == value:
                    return r, c
        raise ValueError(f"{value} not found in {self.name}")

    def row_containing(self, value: int) -> int:
        return self.pos(value)[0]

    def col_containing(self, value: int) -> int:
        return self.pos(value)[1]

    def input_directions(self) -> Set[Direction]:
        """Directions from which this H can receive external/neighbor input.

        The input terminal 1 sits at a corner.  Its row and column determine the
        two geometric dimensions from which signal may enter.
        """
        row, col = self.pos(1)
        directions: Set[Direction] = set()
        directions.add(Direction.UP if row == 0 else Direction.DOWN)
        directions.add(Direction.LEFT if col == 0 else Direction.RIGHT)
        return directions

    def output_directions(self) -> Set[Direction]:
        """Directions to which this H can emit output from terminal 7."""
        row, col = self.pos(7)
        directions: Set[Direction] = set()
        directions.add(Direction.UP if row == 0 else Direction.DOWN)
        directions.add(Direction.LEFT if col == 0 else Direction.RIGHT)
        return directions

    def compressed_nine_grid(self) -> Grid:
        """Return the H as a self-similar 3x3 grid.

        The upper/lower middle cells are latent potential sites.  In stable
        states they are empty; during long weak activation they can hold TEMP_4
        and form the 4+4 reconstruction bridge.
        """
        return clone_grid(self.grid)

    def latent_sites(self) -> Tuple[Pos, Pos]:
        return (0, 1), (2, 1)

    def self_similar_signature(self) -> Dict[str, object]:
        return {
            "bit": self.bit if self.body in ((3, 4, 5), (5, 4, 3)) else None,
            "input": self.pos(1),
            "output": self.pos(7),
            "latent_sites": self.latent_sites(),
            "axon_hillock": (self.pos(5), self.pos(6)),
            "input_directions": {direction.value for direction in self.input_directions()},
            "output_directions": {direction.value for direction in self.output_directions()},
            "membrane_amplitude": self.membrane.potential(),
            "membrane_phase": self.membrane.phase_potential(),
            "membrane_state": self.membrane.state(),
            "refractory_ticks": self.refractory_ticks,
        }

    @property
    def body(self) -> Tuple[object, object, object]:
        return tuple(self.grid[1])

    @property
    def bit(self) -> int:
        if self.body == (3, 4, 5):
            return 0
        if self.body == (5, 4, 3):
            return 1
        raise ValueError(f"{self.name} body row is not a stable bit: {self.body}")

    def is_stable(self) -> bool:
        if self.grid[0][1] is not EMPTY or self.grid[2][1] is not EMPTY:
            return False
        if self.body not in ((3, 4, 5), (5, 4, 3)):
            return False
        if self.row_containing(1) != self.row_containing(2):
            return False
        if self.row_containing(6) != self.row_containing(7):
            return False
        return self.col_containing(1) == self.col_containing(5) == self.col_containing(6)

    def axon_hillock_active(self) -> bool:
        return self.col_containing(5) == self.col_containing(6)

    def phase_grid(self) -> List[List[Optional[complex]]]:
        phase_grid: List[List[Optional[complex]]] = []
        for row in self.grid:
            phase_row: List[Optional[complex]] = []
            for cell in row:
                if isinstance(cell, int):
                    phase_row.append(digit_state(cell))
                elif cell == TEMP_4:
                    phase_row.append(digit_state(4))
                else:
                    phase_row.append(EMPTY)
            phase_grid.append(phase_row)
        return phase_grid

    def amplitude_grid(self) -> List[List[Optional[float]]]:
        amplitude_grid: List[List[Optional[float]]] = []
        for row in self.grid:
            amplitude_row: List[Optional[float]] = []
            for cell in row:
                if isinstance(cell, int):
                    amplitude_row.append(digit_amplitude(cell))
                elif cell == TEMP_4:
                    amplitude_row.append(digit_amplitude(4))
                else:
                    amplitude_row.append(EMPTY)
            amplitude_grid.append(amplitude_row)
        return amplitude_grid

    def axon_hillock_phase_gate(
        self,
        epsilon: float = ANGLE_TOLERANCE,
        target_delta: Optional[float] = None,
    ) -> bool:
        """Phase-threshold form of the 5/6 same-column output gate."""
        if target_delta is None:
            target_delta = wrap_angle(phase_angle(6) - phase_angle(5))
        delta = wrap_angle(phase_angle(6) - phase_angle(5))
        return self.axon_hillock_active() and abs(wrap_angle(delta - target_delta)) <= epsilon

    def add_membrane_phase(self, increment: float) -> bool:
        if increment >= STRONG_PHASE_INCREMENT:
            amplitude_increment = STRONG_AMPLITUDE_INCREMENT
        elif increment <= INHIBITORY_PHASE_INCREMENT:
            amplitude_increment = INHIBITORY_AMPLITUDE_INCREMENT
        else:
            amplitude_increment = WEAK_AMPLITUDE_INCREMENT
        return self.membrane.add(amplitude_increment, increment)

    def membrane_ready(self) -> bool:
        return self.membrane.ready()

    def structural_output_ready(self) -> bool:
        """Discrete H-structure gate: output is allowed only when 5 and 6 align.

        This is the axon-hillock rule in spatial form.  It deliberately stays
        separate from membrane amplitude so that the model has two gates:
        geometry must permit output, and accumulated membrane amplitude must
        reach threshold.
        """
        return self.axon_hillock_active()

    def output_gate_ready(self) -> bool:
        """Full H output gate: discrete structure plus amplitude-phase membrane."""
        return self.structural_output_ready() and self.axon_hillock_phase_gate() and self.membrane_ready()

    def reset_membrane(self) -> None:
        self.membrane.reset()

    def enter_refractory(self) -> None:
        self.cleanup_temp4()
        self.reset_membrane()
        self.refractory_ticks = 4
        self.phase = Phase.R0

    def is_refractory(self) -> bool:
        return self.refractory_ticks > 0

    def advance_refractory(self) -> None:
        if self.refractory_ticks <= 0:
            return
        self.refractory_ticks -= 1
        if self.refractory_ticks == 3:
            self.phase = Phase.R1
        elif self.refractory_ticks == 2:
            self.phase = Phase.R2
            self.grid = clone_grid(self.original)
        elif self.refractory_ticks == 1:
            self.phase = Phase.R3
        else:
            self.phase = Phase.READY
            self.grid = clone_grid(self.original)
            self.last_output = False
            self.local_output = False

    def flip_body(self) -> None:
        if self.grid[1] == [3, 4, 5]:
            self.grid[1] = [5, 4, 3]
        elif self.grid[1] == [5, 4, 3]:
            self.grid[1] = [3, 4, 5]
        else:
            self.grid[1] = [self.grid[1][2], self.grid[1][1], self.grid[1][0]]

    def restore_body(self) -> None:
        self.grid[1] = self.original[1][:]

    def swap_row_containing(self, value: int) -> None:
        row = self.row_containing(value)
        self.grid[row] = [self.grid[row][2], self.grid[row][1], self.grid[row][0]]

    def swap_all_rows_lr(self) -> None:
        for row in range(3):
            self.grid[row] = [self.grid[row][2], self.grid[row][1], self.grid[row][0]]

    def set_input_middle_temp4(self) -> None:
        self.grid[self.row_containing(1)][1] = TEMP_4

    def set_output_middle_temp4(self) -> None:
        self.grid[self.row_containing(7)][1] = TEMP_4

    def temp4_bridge_active(self) -> bool:
        return (
            self.grid[self.row_containing(1)][1] == TEMP_4
            and self.grid[self.row_containing(7)][1] == TEMP_4
        )

    def cleanup_temp4(self) -> None:
        for row in (0, 2):
            if self.grid[row][1] == TEMP_4:
                self.grid[row][1] = EMPTY

    def reconstruct_input_output(self) -> None:
        """Spatial reconstruction: exchange input and output rows.

        Reconstruction swaps the first and third rows as whole layers.  This is
        the H-level input/output exchange: the dendrite layer and axon layer
        exchange positions while the body row keeps its current order.
        """
        self.grid[0], self.grid[2] = self.grid[2], self.grid[0]
        self.cleanup_temp4()
        # Completed reconstruction is learned geometry, not a transient.  Future
        # recovery returns to this new original rather than undoing the learned
        # input/output exchange.
        self.original = clone_grid(self.grid)

    def restore_original(self) -> None:
        self.grid = clone_grid(self.original)
        self.phase = Phase.READY
        self.weak_stage = 0
        self.local_stage = 0
        self.long_weak_stage = 0
        self.last_output = False
        self.local_output = False
        self.reset_membrane()
        self.refractory_ticks = 0

    def _original_row_containing(self, value: int) -> int:
        for r, row in enumerate(self.original):
            if value in row:
                return r
        raise ValueError(f"{value} not found in original {self.name}")

    def decay_uncontinued_weak_state(self) -> None:
        """Complete the original weak/local four-step rule when input stops.

        This is not an extra decay mechanism. It is the T2/T3 branch already
        contained in the weak-signal accumulation rule:

        stage 1 without a second input -> T2 restore -> T3 no output
        stage 2 without a third input -> T2 body/input-row restore -> T3 no output
        """
        stage = max(self.weak_stage, self.local_stage, self.long_weak_stage)
        if stage == 1:
            self.restore_original()
            self.phase = Phase.T3
            return
        if stage == 2:
            self.phase = Phase.T2
            self.restore_body()
            input_row = self._original_row_containing(1)
            self.grid[input_row] = self.original[input_row][:]
            self.phase = Phase.T3
            self.weak_stage = 0
            self.local_stage = 0
            self.long_weak_stage = 0
            self.last_output = False
            self.local_output = False
            self.reset_membrane()

    def start_long_weak_reconstruction(self) -> int:
        """Continuous temporary evolution for strong-driven reconstruction."""
        self.add_membrane_phase(WEAK_PHASE_INCREMENT)
        self.long_weak_stage += 1
        stage = self.long_weak_stage
        self.phase = Phase.T1
        if stage == 1:
            self.set_input_middle_temp4()
        elif stage == 2:
            self.flip_body()
        else:
            self.set_output_middle_temp4()
        return stage

    def start_strong_signal(self) -> None:
        self.add_membrane_phase(STRONG_PHASE_INCREMENT)
        self.phase = Phase.T1
        self.set_output_middle_temp4()

    def start_weak_or_local_signal(self, local: bool = False) -> int:
        self.add_membrane_phase(WEAK_PHASE_INCREMENT)
        if local:
            self.local_stage += 1
            stage = self.local_stage
        else:
            self.weak_stage += 1
            stage = self.weak_stage

        self.phase = Phase.T1
        if stage == 1:
            self.set_input_middle_temp4()
        elif stage == 2:
            self.flip_body()
        else:
            self.flip_body()
            self.set_output_middle_temp4()
        return stage

    def recover_body_only(self) -> None:
        self.phase = Phase.T2
        self.restore_body()

    def gate_output(self) -> bool:
        self.phase = Phase.T3
        self.last_output = self.output_gate_ready()
        if self.last_output:
            self.enter_refractory()
        return self.last_output

    def recover_and_gate(self) -> bool:
        self.recover_body_only()
        return self.gate_output()

    def recover_original_geometry_and_gate(self) -> bool:
        """Run the strong-signal output gate after transient fill.

        Strong signals use output-row middle fill as a transient axon-potential
        amplitude state. The fill remains through T3, then refractory cleanup
        returns it to empty.
        """
        return self.gate_output()


@dataclass(frozen=True)
class Contact:
    orientation: str
    pre_side: Tuple[object, ...]
    post_side: Tuple[object, ...]
    point_7_to_1: bool
    pair_6_to_2: bool
    body_complement: bool
    all_layer_complement: bool
    inhibitory_only: bool


def _side(neuron: HNeuron, side: str) -> Tuple[object, ...]:
    if side == "left":
        return (neuron.grid[0][0], neuron.grid[1][0], neuron.grid[2][0])
    if side == "right":
        return (neuron.grid[0][2], neuron.grid[1][2], neuron.grid[2][2])
    if side == "top":
        return (neuron.grid[0][0], neuron.grid[0][2])
    if side == "bottom":
        return (neuron.grid[2][0], neuron.grid[2][2])
    raise ValueError(f"unknown side {side}")


def contact(pre: HNeuron, post: HNeuron, orientation: str) -> Contact:
    """Return geometric contact facts for pre -> post.

    orientation means where the post neuron sits relative to pre:
    right, left, down, or up.
    """
    if orientation == "right":
        pre_side = _side(pre, "right")
        post_side = _side(post, "left")
        point = any(a == 7 and b == 1 for a, b in zip(pre_side, post_side))
        pair = False
        body = complement(pre_side[1], post_side[1])
        all_layer = all(complement(a, b) for a, b in zip(pre_side, post_side))
        inhibitory = complement(pre_side[2], post_side[2]) and not body and not point
    elif orientation == "left":
        pre_side = _side(pre, "left")
        post_side = _side(post, "right")
        point = any(a == 7 and b == 1 for a, b in zip(pre_side, post_side))
        pair = False
        body = complement(pre_side[1], post_side[1])
        all_layer = all(complement(a, b) for a, b in zip(pre_side, post_side))
        inhibitory = complement(pre_side[2], post_side[2]) and not body and not point
    elif orientation == "down":
        pre_side = _side(pre, "bottom")
        post_side = _side(post, "top")
        point = any(a == 7 and b == 1 for a, b in zip(pre_side, post_side))
        pair = any(a == 6 and b == 2 for a, b in zip(pre_side, post_side))
        body = complement(pre.grid[1][1], post.grid[1][1]) or any(
            complement(a, b) for a, b in zip(pre.grid[1], post.grid[1])
        )
        all_layer = all(complement(a, b) for a, b in zip(pre_side, post_side))
        inhibitory = all_layer and not (point and pair)
    elif orientation == "up":
        pre_side = _side(pre, "top")
        post_side = _side(post, "bottom")
        point = any(a == 7 and b == 1 for a, b in zip(pre_side, post_side))
        pair = any(a == 6 and b == 2 for a, b in zip(pre_side, post_side))
        body = complement(pre.grid[1][1], post.grid[1][1]) or any(
            complement(a, b) for a, b in zip(pre.grid[1], post.grid[1])
        )
        all_layer = all(complement(a, b) for a, b in zip(pre_side, post_side))
        inhibitory = all_layer and not (point and pair)
    else:
        raise ValueError("orientation must be right, left, down, or up")

    return Contact(
        orientation=orientation,
        pre_side=pre_side,
        post_side=post_side,
        point_7_to_1=point,
        pair_6_to_2=pair,
        body_complement=body,
        all_layer_complement=all_layer,
        inhibitory_only=inhibitory,
    )


def classify_coupling(pre: HNeuron, post: HNeuron, orientation: str) -> Coupling:
    facts = contact(pre, post, orientation)

    if orientation in {"right", "left"}:
        effective_contact = facts.point_7_to_1
        strong = facts.point_7_to_1 and facts.body_complement
    else:
        effective_contact = facts.point_7_to_1 and facts.pair_6_to_2
        strong = effective_contact and facts.body_complement

    if facts.all_layer_complement and effective_contact:
        return Coupling.FUNCTIONAL_STABLE
    if facts.all_layer_complement:
        return Coupling.STABLE
    if strong:
        return Coupling.STRONG
    if facts.inhibitory_only:
        return Coupling.INHIBITORY
    if facts.body_complement:
        return Coupling.WEAK
    return Coupling.INVALID


def propagation_outcome(coupling: Coupling, signal: Signal) -> Outcome:
    if signal == Signal.INHIBITORY or coupling == Coupling.INHIBITORY:
        return Outcome.INHIBITED
    if coupling in {Coupling.FUNCTIONAL_STABLE, Coupling.STRONG}:
        return Outcome.MAIN if signal == Signal.STRONG else Outcome.SUBTHRESHOLD
    if coupling == Coupling.WEAK:
        return Outcome.LOCAL if signal == Signal.STRONG else Outcome.NOISE
    return Outcome.NONE


@dataclass
class Synapse:
    pre: str
    post: str
    orientation: str
    fixed_coupling: Optional[Coupling] = None
    intrinsic: bool = False
    weak_long_stage: int = 0

    def evaluate(self, neurons: Dict[str, HNeuron]) -> Coupling:
        if self.fixed_coupling is not None:
            return self.fixed_coupling
        return classify_coupling(neurons[self.pre], neurons[self.post], self.orientation)

    def stimulate(
        self,
        neurons: Dict[str, HNeuron],
        signal: Signal,
        allow_reconstruction: bool = True,
    ) -> Outcome:
        target = neurons[self.post]
        coupling = self.evaluate(neurons)
        outcome = propagation_outcome(coupling, signal)
        if signal == Signal.WEAK and target.weak_stage > 0:
            outcome = Outcome.SUBTHRESHOLD
        elif signal == Signal.STRONG and target.local_stage > 0:
            outcome = Outcome.LOCAL

        if target.is_refractory():
            target.advance_refractory()
            return Outcome.NONE
        if outcome == Outcome.INHIBITED:
            target.restore_original()
            self.weak_long_stage = 0
            return outcome
        if outcome == Outcome.MAIN:
            target.start_strong_signal()
            target.recover_original_geometry_and_gate()
            self.weak_long_stage = 0
            return outcome
        if outcome == Outcome.SUBTHRESHOLD:
            stage = target.start_weak_or_local_signal(local=False)
            if stage == 1:
                # Temporary state 1: the input-row middle cell is filled and
                # membrane amplitude starts accumulating. It is below threshold
                # and is not eligible for output.
                target.phase = Phase.T1
                target.last_output = False
            elif stage == 2:
                # Temporary state 2: no row exchange occurs; the body row
                # flips as a sampled phase rotation while membrane amplitude
                # continues accumulating.
                target.phase = Phase.T1
                target.last_output = False
            else:
                # Temporary state 3 fills the output-row middle cell. T3 gates
                # output from the accumulated amplitude state.
                target.recover_and_gate()
                target.weak_stage = 0
            self.weak_long_stage = 0
            return outcome
        if outcome == Outcome.LOCAL:
            stage = target.start_weak_or_local_signal(local=True)
            if stage == 1:
                target.phase = Phase.T1
                target.last_output = False
            elif stage == 2:
                target.phase = Phase.T1
                target.last_output = False
            else:
                target.recover_and_gate()
                target.local_stage = 0
            target.local_output = target.last_output
            return outcome
        return outcome

    def activate_long_reconstruction(
        self,
        neurons: Dict[str, HNeuron],
        signal: Signal = Signal.STRONG,
    ) -> Coupling:
        """Apply one long-term reconstruction activation.

        Long weak-connection reconstruction is driven by strong input on a weak
        coupling.  It is intentionally separate from weak-signal accumulation
        and local activation.
        """
        target = neurons[self.post]
        if (
            signal != Signal.STRONG
            or (target.long_weak_stage == 0 and self.evaluate(neurons) != Coupling.WEAK)
        ):
            self.weak_long_stage = 0
            return self.evaluate(neurons)
        self._long_reconstruction_activation(target, neurons)
        return self.evaluate(neurons)

    def activate_long_weak_reconstruction(self, neurons: Dict[str, HNeuron]) -> Coupling:
        """Compatibility alias for strong-driven long reconstruction."""
        return self.activate_long_reconstruction(neurons, Signal.STRONG)

    def _long_reconstruction_activation(self, target: HNeuron, neurons: Dict[str, HNeuron]) -> None:
        """Continuous strong-driven reconstruction of a weak coupling.

        The temporary 4 only acts on the target H neuron. This is self-similar
        to weak-signal accumulation, but its input is a strong signal applied to
        a weak coupling.
        """
        stage = target.start_long_weak_reconstruction()
        self.weak_long_stage = stage
        if stage >= 3:
            if target.temp4_bridge_active():
                # The body flip in the second long-weak stage is a transient
                # center activation.  Reconstruction itself is input/output
                # exchange, so the body returns to its original order before
                # the new spatial structure is confirmed.
                target.restore_body()
                target.reconstruct_input_output()
            target.reset_membrane()
            target.long_weak_stage = 0
            self.weak_long_stage = 0


@dataclass
class HNetwork:
    neurons: Dict[str, HNeuron]
    synapses: List[Synapse]
    tick: int = 0

    def step(self, inputs: Iterable[Tuple[str, Signal]]) -> List[Tuple[str, Outcome]]:
        """Synchronous network step.

        Each input names a synapse as "pre->post:orientation", for example
        "a->b:right".  Outputs from a T3 step are intended to be supplied as T0
        inputs on the next call by the caller or a higher-level simulator.
        """
        outcomes: List[Tuple[str, Outcome]] = []
        synapse_map = {
            f"{syn.pre}->{syn.post}:{syn.orientation}": syn for syn in self.synapses
        }
        input_list = list(inputs)
        stimulated_posts = set()
        for key, signal in input_list:
            if key not in synapse_map:
                raise KeyError(f"unknown synapse {key}")
            stimulated_posts.add(synapse_map[key].post)
            outcome = synapse_map[key].stimulate(self.neurons, signal)
            outcomes.append((key, outcome))
        for name, neuron in self.neurons.items():
            if name not in stimulated_posts:
                if neuron.is_refractory():
                    neuron.advance_refractory()
                    continue
                neuron.decay_uncontinued_weak_state()
        self.tick += 1
        return outcomes


@dataclass
class ReflexArcResult:
    stimulus: ExternalStimulus
    steps: List[Tuple[str, Outcome]]
    effector_activated: bool
    response: StressResponse


@dataclass
class Receptor:
    name: str = "receptor"
    weak_stage: int = 0
    membrane: PhaseMembrane = field(default_factory=PhaseMembrane)

    def sense(self, stimulus: ExternalStimulus) -> Optional[Signal]:
        if stimulus == ExternalStimulus.STRONG:
            self.weak_stage = 0
            self.membrane.add(STRONG_AMPLITUDE_INCREMENT, STRONG_PHASE_INCREMENT)
            self.membrane.reset()
            return Signal.STRONG
        if stimulus == ExternalStimulus.WEAK:
            self.weak_stage += 1
            if self.membrane.add(WEAK_AMPLITUDE_INCREMENT, WEAK_PHASE_INCREMENT):
                self.weak_stage = 0
                self.membrane.reset()
                return Signal.STRONG
            return None
        if stimulus == ExternalStimulus.INHIBITORY:
            self.weak_stage = 0
            self.membrane.add(INHIBITORY_AMPLITUDE_INCREMENT, INHIBITORY_PHASE_INCREMENT)
            self.membrane.reset()
            return Signal.INHIBITORY
        self.weak_stage = 0
        self.membrane.reset()
        return None


@dataclass
class Effector:
    name: str = "effector"
    activated: bool = False
    response: StressResponse = StressResponse.NONE

    def receive(self, motor_output: bool, local: bool = False) -> StressResponse:
        self.activated = motor_output
        if not motor_output:
            self.response = StressResponse.NONE
        elif local:
            self.response = StressResponse.LOCAL
        else:
            self.response = StressResponse.REFLEX
        return self.response


GridCoord = Tuple[int, int]


def opposite(direction: Direction) -> Direction:
    return {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }[direction]


def neighbor_at(coord: GridCoord, direction: Direction) -> GridCoord:
    row, col = coord
    if direction == Direction.UP:
        return row - 1, col
    if direction == Direction.DOWN:
        return row + 1, col
    if direction == Direction.LEFT:
        return row, col - 1
    if direction == Direction.RIGHT:
        return row, col + 1
    raise ValueError(direction)


def orientation_for(direction: Direction) -> str:
    return {
        Direction.UP: "up",
        Direction.DOWN: "down",
        Direction.LEFT: "left",
        Direction.RIGHT: "right",
    }[direction]


@dataclass
class NineHReflexArc:
    """3x3 reflex arc made from the ABA / CDC / ABA H layout.

    The four corner A cells are the legal starting states.  Once one corner is
    chosen, the nine-grid intrinsic synapse path is fixed.
    """

    neurons: Dict[GridCoord, HNeuron]
    synapses: List[Synapse]
    network: HNetwork
    start_coord: GridCoord = (0, 2)
    receptor: Receptor = field(default_factory=Receptor)
    effector: Effector = field(default_factory=Effector)

    @classmethod
    def standard(cls, start_coord: GridCoord = (0, 2)) -> "NineHReflexArc":
        if start_coord not in cls.legal_start_coords():
            raise ValueError("standard nine-H reflex arc must start from a corner A cell")

        pattern = [["A", "B", "A"], ["C", "D", "C"], ["A", "B", "A"]]
        neurons: Dict[GridCoord, HNeuron] = {}
        flat: Dict[str, HNeuron] = {}
        for row in range(3):
            for col in range(3):
                name = f"h{row}{col}"
                neuron = HNeuron.from_state(name, pattern[row][col])
                neurons[(row, col)] = neuron
                flat[name] = neuron

        # In this reflex-arc layout the internal H-H synapses are fixed
        # "nine-grid intrinsic synapses".  The chosen corner determines one
        # unique serpentine path through all nine H cells.
        fixed_edges = cls.path_edges_for(start_coord)
        synapses = [
            Synapse(
                neurons[pre].name,
                neurons[post].name,
                orientation_for(direction),
                fixed_coupling=Coupling.STRONG,
                intrinsic=True,
            )
            for pre, post, direction in fixed_edges
        ]
        arc = cls(
            neurons=neurons,
            synapses=synapses,
            network=HNetwork(flat, synapses),
            start_coord=start_coord,
        )
        if not arc.validate_fixed_strong_synapses():
            raise ValueError("standard nine-H reflex arc contains a non-strong synapse")
        return arc

    @staticmethod
    def legal_start_coords() -> List[GridCoord]:
        return [(0, 0), (0, 2), (2, 0), (2, 2)]

    @staticmethod
    def path_coords_for(start_coord: GridCoord) -> List[GridCoord]:
        paths = {
            (0, 0): [
                (0, 0), (0, 1), (0, 2),
                (1, 2), (1, 1), (1, 0),
                (2, 0), (2, 1), (2, 2),
            ],
            (0, 2): [
                (0, 2), (0, 1), (0, 0),
                (1, 0), (1, 1), (1, 2),
                (2, 2), (2, 1), (2, 0),
            ],
            (2, 0): [
                (2, 0), (2, 1), (2, 2),
                (1, 2), (1, 1), (1, 0),
                (0, 0), (0, 1), (0, 2),
            ],
            (2, 2): [
                (2, 2), (2, 1), (2, 0),
                (1, 0), (1, 1), (1, 2),
                (0, 2), (0, 1), (0, 0),
            ],
        }
        if start_coord not in paths:
            raise ValueError("standard nine-H reflex arc must start from a corner A cell")
        return paths[start_coord]

    @classmethod
    def path_edges_for(cls, start_coord: GridCoord) -> List[Tuple[GridCoord, GridCoord, Direction]]:
        coords = cls.path_coords_for(start_coord)
        edges: List[Tuple[GridCoord, GridCoord, Direction]] = []
        for pre, post in zip(coords, coords[1:]):
            row_delta = post[0] - pre[0]
            col_delta = post[1] - pre[1]
            if row_delta == -1 and col_delta == 0:
                direction = Direction.UP
            elif row_delta == 1 and col_delta == 0:
                direction = Direction.DOWN
            elif row_delta == 0 and col_delta == -1:
                direction = Direction.LEFT
            elif row_delta == 0 and col_delta == 1:
                direction = Direction.RIGHT
            else:
                raise ValueError("nine-H intrinsic path must use adjacent H cells")
            edges.append((pre, post, direction))
        return edges

    def path_coords(self) -> List[GridCoord]:
        return self.path_coords_for(self.start_coord)

    def path_direction_operators(self) -> List[complex]:
        return [
            direction_operator(direction)
            for _, _, direction in self.path_edges_for(self.start_coord)
        ]

    def net_path_operator(self) -> complex:
        result = 1 + 0j
        for operator in self.path_direction_operators():
            result *= operator
        return result

    def external_input_coords(self) -> List[GridCoord]:
        return self.legal_start_coords()

    def external_output_coords(self) -> List[GridCoord]:
        return [self.path_coords()[-1]]

    def synapse_key(self, synapse: Synapse) -> str:
        return f"{synapse.pre}->{synapse.post}:{synapse.orientation}"

    def outgoing_synapses(self, neuron_name: str) -> List[Synapse]:
        return [synapse for synapse in self.synapses if synapse.pre == neuron_name]

    def unique_internal_path(self) -> List[str]:
        return [self.synapse_key(synapse) for synapse in self.synapses]

    def validate_fixed_strong_synapses(self) -> bool:
        return all(
            synapse.evaluate(self.network.neurons)
            in {Coupling.STRONG, Coupling.FUNCTIONAL_STABLE}
            for synapse in self.synapses
        )

    def intrinsic_synapses(self) -> List[Synapse]:
        """Return the nine-grid intrinsic synapse chain.

        Intrinsic synapses are fixed by the self-similar 3x3 structure rather
        than discovered by local H-H search. They are the unique main reflex
        path and default to strong coupling.
        """
        return [synapse for synapse in self.synapses if synapse.intrinsic]

    def validate_intrinsic_synapses(self) -> bool:
        return (
            len(self.intrinsic_synapses()) == 8
            and self.unique_internal_path()
            == [self.synapse_key(synapse) for synapse in self.intrinsic_synapses()]
            and self.validate_fixed_strong_synapses()
        )

    def pattern(self) -> List[List[str]]:
        return [["A", "B", "A"], ["C", "D", "C"], ["A", "B", "A"]]

    def auxiliary_coords(self) -> List[GridCoord]:
        path_names = {
            key.split("->", 1)[0] for key in self.unique_internal_path()
        } | {
            key.split("->", 1)[1].split(":", 1)[0]
            for key in self.unique_internal_path()
        }
        return [
            coord
            for coord, neuron in self.neurons.items()
            if neuron.name not in path_names
        ]

    def self_similar_signature(self) -> Dict[str, object]:
        return {
            "pattern": self.pattern(),
            "legal_start_coords": self.legal_start_coords(),
            "start_coord": self.start_coord,
            "output_coord": self.external_output_coords()[0],
            "path": self.unique_internal_path(),
            "path_direction_operators": self.path_direction_operators(),
            "net_path_operator": self.net_path_operator(),
            "auxiliary_coords": self.auxiliary_coords(),
            "intrinsic_synapses": self.unique_internal_path(),
            "stage_template": [
                "input terminal change",
                "body/center change",
                "output terminal or reconstruction",
            ],
        }

    def run_external_stimulus(
        self, stimulus: ExternalStimulus, max_steps: int = 16
    ) -> ReflexArcResult:
        signal = self.receptor.sense(stimulus)
        if signal is None:
            self.effector.receive(False)
            return ReflexArcResult(stimulus, [], False, StressResponse.NONE)

        steps: List[Tuple[str, Outcome]] = []
        for key in self.unique_internal_path()[:max_steps]:
            outcome = self.network.step([(key, signal)])[0]
            steps.append(outcome)
            post_name = key.split("->", 1)[1].split(":", 1)[0]
            if not self.network.neurons[post_name].last_output:
                break

        output_coords = self.external_output_coords()
        output_names = {self.neurons[coord].name for coord in output_coords}
        effector_active = any(
            self.network.neurons[name].last_output for name in output_names
        )
        response = self.effector.receive(effector_active)
        return ReflexArcResult(stimulus, steps, effector_active, response)


def render_grid(grid: Grid) -> str:
    def cell(value: object) -> str:
        return " " if value is EMPTY else str(value)

    return "\n".join(
        [
            f"{cell(grid[0][0])} {cell(grid[0][1])} {cell(grid[0][2])}",
            f"{cell(grid[1][0])} {cell(grid[1][1])} {cell(grid[1][2])}",
            f"{cell(grid[2][0])} {cell(grid[2][1])} {cell(grid[2][2])}",
        ]
    )


def demo() -> None:
    left = HNeuron.from_state("left", "A")
    right = HNeuron.from_state("right", "A")
    syn = Synapse("left", "right", "right")
    network = HNetwork({"left": left, "right": right}, [syn])
    key = "left->right:right"

    print("Initial coupling:", syn.evaluate(network.neurons).value)
    for i in range(3):
        outcome = network.step([(key, Signal.STRONG)])
        print(f"After weak-coupling activation {i + 1}:", outcome[0][1].value)
        print(render_grid(right.grid))
        print("Coupling:", syn.evaluate(network.neurons).value)


def reflex_demo() -> None:
    arc9 = NineHReflexArc.standard()
    print("Nine-H reflex arc:")
    print("  input coords:", arc9.external_input_coords())
    print("  output coords:", arc9.external_output_coords())

    print("Strong stimulus:")
    strong = arc9.run_external_stimulus(ExternalStimulus.STRONG)
    for key, outcome in strong.steps:
        print(f"  {key}: {outcome.value}")
    print("  effector:", "activated" if strong.effector_activated else "inactive")
    print("  response:", strong.response.value)

    arc9 = NineHReflexArc.standard()
    print("Weak stimulus accumulation:")
    weak = ReflexArcResult(ExternalStimulus.WEAK, [], False, StressResponse.NONE)
    for _ in range(3):
        weak = arc9.run_external_stimulus(ExternalStimulus.WEAK)
    for key, outcome in weak.steps:
        print(f"  {key}: {outcome.value}")
    print("  effector:", "activated" if weak.effector_activated else "inactive")
    print("  response:", weak.response.value)


if __name__ == "__main__":
    demo()
    print()
    reflex_demo()
