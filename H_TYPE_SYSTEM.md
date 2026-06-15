# H-Type Complex Phase Reflex Arc System

## 0. Continuous Parent Dynamics

The discrete H model is interpreted as a sampled eigenstate set of a continuous
complex amplitude-phase dynamics under H-type symmetry constraints.

The parent field equation is:

```text
dpsi/dt = (alpha + i omega) psi
        - (beta + i gamma) |psi|^2 psi
        + D laplacian(psi)
```

Meanings:

```text
psi              = complex amplitude-phase state
alpha            = linear excitation / growth
omega            = intrinsic phase rotation
beta             = nonlinear amplitude saturation
gamma            = amplitude-dependent phase shift
D laplacian(psi) = spatial coupling / diffusion
```

The discrete model arises by imposing:

```text
H-shaped spatial constraint
1..7 energy-level labels
complement phase quantization x+y=8
four legal mirror-complement H eigenstates
T0/T1/T2/T3 sampled time sections
R0/R1/R2/R3 refractory recovery sections
```

Thus:

```text
digits 1..7 = energy-level labels
345/543     = degenerate ground-state pair
x+y=8       = phase quantization condition
A,B,C,D     = symmetry-constrained H eigenstates
```

Because one legal corner digit uniquely determines a full H state, the field can
also be reduced to a one-mode compressed wave variable:

```text
chi_d = A_d exp(i theta_d)
```

The reduced wave equation is the single-site parent dynamics:

```text
dchi/dt = (alpha + i omega) chi - (beta + i gamma) |chi|^2 chi
```

This is the parent equation after the H symmetry constraint removes the explicit
spatial Laplacian.  The full H structure can then be recovered by expanding the
corner digit.

## 1. Core Digit Space

The digit set is:

```text
D = {1,2,3,4,5,6,7}
```

The complement involution is:

```text
k(n) = 8 - n
```

So:

```text
1 <-> 7
2 <-> 6
3 <-> 5
4 <-> 4
```

## 2. Stable H Neuron

A stable H neuron is a 3 by 3 grid with the upper-middle and lower-middle sites empty:

```text
a _ b
c d e
f _ g
```

The stable constraints are:

```text
body row in {(3,4,5), (5,4,3)}
row(1) = row(2)
row(6) = row(7)
col(1) = col(5) = col(6)
```

The four legal stable states are:

```text
A = 2 _ 1 / 3 4 5 / 7 _ 6
B = 7 _ 6 / 3 4 5 / 2 _ 1
C = 1 _ 2 / 5 4 3 / 6 _ 7
D = 6 _ 7 / 5 4 3 / 1 _ 2
```

They are symmetric complement-mirror states.

Using the right-top corner as the reduced coordinate:

```text
1 -> A
6 -> B
2 -> C
7 -> D
```

Thus one corner digit can encode the entire legal H state.

## 3. Bit and Time Sequence

The body row stores one bit:

```text
345 = 0
543 = 1
```

The four-step timing is:

```text
T0 input
T1 flip
T2 recover
T3 output gate
```

Output is permitted only when the axon hillock gate is active:

```text
col(5) = col(6)
```

## 4. Weak Signal Accumulation

Weak signal accumulation is a continuous three-stage temporary evolution:

```text
W1: input-row middle empty -> filled 4, no output
W2: no row exchange, body row flips, membrane amplitude/phase continues accumulating, no output
W3: output-row middle empty -> filled 4, then T3 output gate
```

If the next weak input does not arrive inside the four-step timing window, the temporary state decays back to the original state without output.

In the complex representation this is also a continuous membrane-potential accumulator:

```text
z_mem(t) = A(t) exp(i phi(t))
```

Amplitude models membrane strength. Phase models synchronization. The default increments are:

```text
weak input   -> A += 1/3, phi += pi/3
strong input -> A += 1,   phi += pi
threshold    -> A >= 1 and phi >= pi
```

Thus three weak inputs are equivalent to one threshold-reaching strong activation:

```text
3 * (1/3) = 1
3 * (pi/3) = pi
```

After successful output, the membrane accumulator resets:

```text
A -> 0
phi -> 0
```

The filled middle-column 4 sites also return to empty after output:

```text
filled 4 -> empty
```

If the weak sequence is not continued, the four-step timing restores the temporary state and clears the pending membrane accumulation.

Strong signal is the one-step equivalent:

```text
S1: output-row middle empty -> filled 4
S2: output gate
S3: filled 4 -> empty, refractory recovery begins
```

The refractory period is:

```text
amplitude reset + R0/R1/R2/R3 timing recovery
```

During refractory recovery, new incoming stimulation is ignored by that H until it returns to READY.

When output succeeds, the T3 judgment immediately enters R0:

```text
T3 success -> R0
```

So T3 is a decision instant, not a full held state.

## 5. Coupling Rules

For adjacent H neurons, layer complement means:

```text
x + y = 8
```

Connection classes:

```text
all-layer complement                  -> stable structure
all-layer complement + 7->1 contact   -> functional stable path
body-layer complement                 -> weak coupling
axon-only complement                  -> inhibitory coupling
no complement                         -> invalid coupling
```

Signal outcome:

```text
strong coupling + strong signal -> main path
strong coupling + weak signal   -> subthreshold accumulation
weak coupling   + strong signal -> local activation
weak coupling   + weak signal   -> neural noise
```

Strong inhibition clears unfinished weak/local states, but does not cancel a main-path signal already in T3 output judgment.

Long weak-connection reconstruction is not weak-signal accumulation. It is
triggered by strong input acting on a weak coupling. The mechanism is
self-similar to weak accumulation, but its input class is strong:

```text
weak coupling + repeated strong input -> long reconstruction stages
```

After reconstruction, the input and output layers are exchanged by swapping the
first and third rows:

```text
row 1 <-> row 3
row 2 unchanged
```

For example:

```text
1 _ 2        6 _ 7
5 4 3   ->   5 4 3
6 _ 7        1 _ 2
```

After this row exchange, the original weak connection automatically becomes a
functional stable path in the reverse direction:

```text
pre -> post : weak coupling
post -> pre : functional stable path
```

So reconstruction both satisfies the strong-connection requirement and reverses
signal transmission direction.

## 6. Nine-H Reflex Arc

The standard reflex arc core is:

```text
A B A
C D C
A B A
```

The four legal starting states are the four corners:

```text
(0,0), (0,2), (2,0), (2,2)
```

Using the same reduced corner digit idea:

```text
1 -> (0,2)
2 -> (0,0)
6 -> (2,2)
7 -> (2,0)
```

So one start digit determines the unique standard nine-H path.

Middle-layer H cells cannot be starting states because they model the reflex-arc core layer. Once a corner is chosen, the intrinsic synapse path is unique and passes through all nine H cells.

The default start is the upper-right corner:

```text
(0,2) -> (0,1) -> (0,0)
      -> (1,0) -> (1,1) -> (1,2)
      -> (2,2) -> (2,1) -> (2,0)
```

Its complex direction product is:

```text
D_path = -1
```

This is a net pi phase flip across the whole reflex path.  It is consistent
with the single-H input response:

```text
345 <-> 543
```

The nine-H path-level phase inversion and the single-H body flip are the same
symmetry action at different scales.

Weak external stimulus accumulates only in the receptor. The third weak stimulus becomes a strong signal entering the nine-H main path.

## 7. Complex Amplitude-Phase Equivalent Form

Digit states have an equivalent complex amplitude-phase representation:

```text
z_n = A_n exp(i theta_n)
theta_n = pi/2 + (n - 4) pi/7
```

By default:

```text
A_n = 1
```

So:

```text
4 -> i
```

Complement is equivalent to:

```text
x + y = 8
theta_x + theta_y = pi mod 2pi
z_x z_y = -1
```

With amplitudes included, complement means phase complement plus amplitude match:

```text
theta_x + theta_y = pi mod 2pi
A_x = A_y
```

Digit flips and amplitude-phase rotations are equivalent descriptions of the same rule.

## 8. Direction Operators

Signal direction is represented in the complex plane:

```text
right = +1
left  = -1
up    = +i
down  = -i
```

Horizontal propagation is real-axis propagation. Vertical propagation is imaginary-axis propagation.

For the default upper-right path, the direction sequence is:

```text
(-1, -1, -i, +1, +1, -i, -1, -1)
```

The net path operator is:

```text
-1
```

This represents a net complement-phase inversion across the reflex path.

## 9. Schrodinger-Form Evolution

The complex phase form can be written as:

```text
i hbar d|psi>/dt = H |psi>
```

For the body bit:

```text
|0> = |345>
|1> = |543>
```

The body flip Hamiltonian is:

```text
H_body = hbar omega [[0,1],[1,0]]
```

This is a mathematical representation of phase evolution, not a claim that the biological system is physically quantum.

## 10. Implemented Checks

The Python tests cover:

```text
four legal H stable states
complex complement z_x z_y = -1
4 -> i
5/6 column gate equals phase gate
weak stimulus as continuous membrane amplitude/phase accumulation
weak signal outputs only on the third pulse
long weak reconstruction is geometric, not forced
four legal nine-H start corners
unique path for each start corner
non-corner starts rejected
complex direction operators for path propagation
Schrodinger-form step preserves normalized state
```

## 11. Qualia, Conscious Field, and Memory

The four legal nine-H boundary start digits decode into four basic qualia:

```text
G_1 -> Q_1
G_2 -> Q_2
G_6 -> Q_6
G_7 -> Q_7
```

Each qualia is a complex amplitude-phase state:

```text
Q_d = A_d exp(i theta_d)
d in {1,2,6,7}
```

The complement pairs are:

```text
Q_1 * Q_7 = -1
Q_2 * Q_6 = -1
```

Amplitude is qualia intensity.  Phase is qualia type.  Neighboring phases can
interfere or resonate; complement phases form opponent or suppressive pairs.

Conditioned reflex drives sensation emergence:

```text
external stimulus
-> receptor accumulation
-> nine-H reflex path
-> qualia Q_d
-> repeated strong activation on weak coupling
-> reconstruction / changed future response
```

Consciousness is modeled as a dynamic integrated sensation field:

```text
Psi(t) = sum_d Q_d(t)
```

A conscious field emerges when the integrated qualia field has enough amplitude,
phase coherence, qualia diversity, and duration:

```text
|Psi| >= A_c
coherence(Psi) >= R_c
active_qualia_count(Psi) >= 2
duration >= T_c
```

The diversity condition is essential.  A single qualia channel, even with high
amplitude, is modeled as a strong isolated sensation rather than a conscious
field.  Consciousness requires at least two distinct qualia components so that
integration, contrast, interference, and field structure can occur.

Memory is not the dynamic field itself.  Memory is a static structure encoded by
the four qualia components:

```text
M = c_1 Q_1 + c_2 Q_2 + c_6 Q_6 + c_7 Q_7
```

Recall is not exact readout.  Recall is reactivation under the current
surrounding H-state context:

```text
M_recall = R(M, H_context)
M_recall = M + Delta M
```

The distortion is:

```text
distortion = ||M_recall - M||
```

So recollection can differ from the original memory because neighboring H states
change amplitude, phase, inhibition, refractory state, reconstruction state, and
available propagation paths.
