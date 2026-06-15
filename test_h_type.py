import math
import unittest

from h_type import (
    Coupling,
    ExternalStimulus,
    HNetwork,
    HNeuron,
    MemoryTrace,
    NineHReflexArc,
    Outcome,
    Phase,
    Receptor,
    Signal,
    STRONG_PHASE_INCREMENT,
    Synapse,
    amplitude_phase_complement,
    body_flip_hamiltonian,
    classify_coupling,
    complex_ginzburg_landau_step,
    consciousness_critical,
    contact,
    corner_digit_from_h,
    digit_state,
    direction_operator,
    h_state_from_corner_digit,
    nine_h_from_start_digit,
    normalize_state,
    phase_complement,
    phase_value,
    qualia_state,
    active_qualia_count,
    reduced_wave_step,
    schrodinger_step,
    start_digit_from_nine_h,
)


class HTypeTests(unittest.TestCase):
    def test_stable_states_and_bits(self) -> None:
        expected_bits = {"A": 0, "B": 0, "C": 1, "D": 1}
        for state, bit in expected_bits.items():
            neuron = HNeuron.from_state(state, state)
            self.assertTrue(neuron.is_stable(), state)
            self.assertEqual(neuron.bit, bit)
            self.assertTrue(neuron.axon_hillock_active())
            self.assertEqual(neuron.latent_sites(), ((0, 1), (2, 1)))
            self.assertIsNone(neuron.compressed_nine_grid()[0][1])
            self.assertIsNone(neuron.compressed_nine_grid()[2][1])

    def test_corner_digit_reduces_and_expands_h_state(self) -> None:
        expected = {1: "A", 6: "B", 2: "C", 7: "D"}
        for digit, state in expected.items():
            neuron = h_state_from_corner_digit(digit, name=state)
            self.assertEqual(neuron.grid, HNeuron.from_state(state, state).grid)
            self.assertEqual(corner_digit_from_h(neuron), digit)

    def test_start_digit_reduces_and_expands_nine_h_arc(self) -> None:
        expected = {1: (0, 2), 2: (0, 0), 6: (2, 2), 7: (2, 0)}
        for digit, coord in expected.items():
            arc = nine_h_from_start_digit(digit)
            self.assertEqual(arc.start_coord, coord)
            self.assertEqual(start_digit_from_nine_h(arc), digit)

    def test_four_qualia_states_form_two_complement_pairs(self) -> None:
        self.assertLess(abs(qualia_state(1) * qualia_state(7) + 1), 1e-9)
        self.assertLess(abs(qualia_state(2) * qualia_state(6) + 1), 1e-9)

    def test_memory_is_static_qualia_combination_and_recall_can_distort(self) -> None:
        memory = MemoryTrace.from_amplitudes({1: 1.0, 2: 0.5, 6: 0.25, 7: 0.0})
        neutral = memory.recall()
        shifted = memory.recall({1: 1 + 0j, 2: 0.8 + 0.2j, 6: 1.2 + 0j, 7: 1 + 0j})

        self.assertAlmostEqual(memory.distortion(neutral), 0.0)
        self.assertGreater(memory.distortion(shifted), 0.0)

    def test_consciousness_requires_qualia_diversity(self) -> None:
        single = {1: qualia_state(1, 2.0), 2: 0j, 6: 0j, 7: 0j}
        diverse = {1: qualia_state(1, 1.0), 2: qualia_state(2, 1.0), 6: 0j, 7: 0j}

        self.assertEqual(active_qualia_count(single), 1)
        self.assertEqual(active_qualia_count(diverse), 2)
        self.assertFalse(consciousness_critical(single, amplitude_threshold=1.0))
        self.assertTrue(consciousness_critical(diverse, amplitude_threshold=1.0))

    def test_complex_phase_representation_matches_complement_rules(self) -> None:
        self.assertAlmostEqual(digit_state(4).real, 0.0)
        self.assertAlmostEqual(digit_state(4).imag, 1.0)
        for a, b in ((1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)):
            self.assertTrue(phase_complement(a, b))
            self.assertTrue(amplitude_phase_complement(a, b))
            self.assertLess(abs(phase_value(a) * phase_value(b) + 1), 1e-9)
            self.assertLess(abs(digit_state(a) * digit_state(b) + 1), 1e-9)

    def test_digit_arrangement_has_amplitude_phase_grid(self) -> None:
        neuron = HNeuron.from_state("a", "A")
        phase_grid = neuron.phase_grid()
        amplitude_grid = neuron.amplitude_grid()

        self.assertEqual(amplitude_grid[0][1], None)
        self.assertEqual(amplitude_grid[2][1], None)
        self.assertAlmostEqual(abs(phase_grid[0][0]), 1.0)
        self.assertAlmostEqual(abs(phase_grid[1][1]), 1.0)
        self.assertAlmostEqual(phase_grid[0][2], digit_state(1))

    def test_phase_gate_is_equivalent_to_axon_hillock_column_rule(self) -> None:
        stable = HNeuron.from_state("stable", "A")
        transient = HNeuron.from_state("transient", "A")
        transient.flip_body()

        self.assertTrue(stable.axon_hillock_active())
        self.assertTrue(stable.axon_hillock_phase_gate())
        self.assertFalse(transient.axon_hillock_active())
        self.assertFalse(transient.axon_hillock_phase_gate())

    def test_output_gate_requires_structure_and_membrane_amplitude(self) -> None:
        neuron = HNeuron.from_state("gate", "A")

        self.assertFalse(neuron.output_gate_ready())
        neuron.add_membrane_phase(STRONG_PHASE_INCREMENT)
        self.assertTrue(neuron.output_gate_ready())

        neuron.flip_body()
        self.assertFalse(neuron.structural_output_ready())
        self.assertFalse(neuron.output_gate_ready())

    def test_receptor_weak_stimulus_is_continuous_membrane_accumulation(self) -> None:
        receptor = Receptor()

        self.assertIsNone(receptor.sense(ExternalStimulus.WEAK))
        self.assertAlmostEqual(receptor.membrane.potential(), 1.0 / 3.0)
        self.assertAlmostEqual(receptor.membrane.phase_potential(), math.pi / 3)

        self.assertIsNone(receptor.sense(ExternalStimulus.WEAK))
        self.assertAlmostEqual(receptor.membrane.potential(), 2.0 / 3.0)
        self.assertAlmostEqual(receptor.membrane.phase_potential(), 2 * math.pi / 3)

        self.assertEqual(receptor.sense(ExternalStimulus.WEAK), Signal.STRONG)
        self.assertAlmostEqual(receptor.membrane.potential(), 0.0)
        self.assertAlmostEqual(receptor.membrane.phase_potential(), 0.0)

    def test_horizontal_weak_and_functional_stable(self) -> None:
        a = HNeuron.from_state("a", "A")
        aa = HNeuron.from_state("aa", "A")
        d = HNeuron.from_state("d", "D")
        c = HNeuron.from_state("c", "C")

        self.assertEqual(classify_coupling(a, aa, "right"), Coupling.WEAK)
        self.assertEqual(classify_coupling(a, HNeuron.from_state("b", "B"), "right"), Coupling.STABLE)
        self.assertEqual(classify_coupling(d, c, "right"), Coupling.FUNCTIONAL_STABLE)

    def test_vertical_complete_axodendritic_pairing(self) -> None:
        a = HNeuron.from_state("a", "A")
        c = HNeuron.from_state("c", "C")
        facts = contact(a, c, "down")

        self.assertTrue(facts.point_7_to_1)
        self.assertTrue(facts.pair_6_to_2)
        self.assertEqual(classify_coupling(a, c, "down"), Coupling.FUNCTIONAL_STABLE)

    def test_strong_signal_outputs_when_hillock_active(self) -> None:
        source = HNeuron.from_state("source", "D")
        target = HNeuron.from_state("target", "C")
        syn = Synapse("source", "target", "right")
        neurons = {"source": source, "target": target}

        self.assertIn(syn.evaluate(neurons), {Coupling.STABLE, Coupling.STRONG, Coupling.FUNCTIONAL_STABLE})
        syn.stimulate(neurons, Signal.STRONG)
        self.assertTrue(target.last_output)
        self.assertTrue(target.axon_hillock_active())
        self.assertEqual(target.grid, target.original)

    def test_weak_connection_reconstruction_is_geometric_not_forced(self) -> None:
        source = HNeuron.from_state("source", "A")
        target = HNeuron.from_state("target", "A")
        syn = Synapse("source", "target", "right")
        neurons = {"source": source, "target": target}

        self.assertEqual(syn.evaluate(neurons), Coupling.WEAK)
        original_top = target.grid[0][:]
        original_body = target.grid[1][:]
        original_bottom = target.grid[2][:]

        syn.activate_long_reconstruction(neurons, Signal.STRONG)
        self.assertEqual(syn.weak_long_stage, 1)
        self.assertEqual(target.grid[target.row_containing(1)][1], "4*")
        self.assertEqual(target.long_weak_stage, 1)

        syn.activate_long_reconstruction(neurons, Signal.STRONG)
        self.assertEqual(syn.weak_long_stage, 2)
        self.assertEqual(target.long_weak_stage, 2)
        self.assertEqual(target.grid[target.row_containing(1)][1], "4*")
        self.assertNotEqual(target.body, tuple(target.original[1]))

        syn.activate_long_reconstruction(neurons, Signal.STRONG)
        self.assertEqual(syn.weak_long_stage, 0)
        self.assertEqual(target.long_weak_stage, 0)
        self.assertIsNone(target.grid[0][1])
        self.assertIsNone(target.grid[2][1])
        learned_grid = [row[:] for row in target.grid]
        target.restore_original()
        self.assertEqual(target.grid, learned_grid)

        # The rule only swaps input/output geometry.  Coupling is then recomputed
        # from structure; this assertion protects against a forced label upgrade.
        self.assertEqual(syn.evaluate(neurons), classify_coupling(source, target, "right"))
        reverse_syn = Synapse("target", "source", "left")
        self.assertEqual(reverse_syn.evaluate(neurons), Coupling.FUNCTIONAL_STABLE)
        self.assertEqual(target.grid[0], original_bottom)
        self.assertEqual(target.grid[1], original_body)
        self.assertEqual(target.grid[2], original_top)

    def test_nine_h_reflex_arc_has_four_legal_start_states(self) -> None:
        arc = NineHReflexArc.standard()

        self.assertEqual(arc.external_input_coords(), [(0, 0), (0, 2), (2, 0), (2, 2)])
        self.assertEqual(arc.external_output_coords(), [(2, 0)])
        self.assertEqual(len(arc.unique_internal_path()), 8)
        self.assertEqual(
            arc.unique_internal_path(),
            [
                "h02->h01:left",
                "h01->h00:left",
                "h00->h10:down",
                "h10->h11:right",
                "h11->h12:right",
                "h12->h22:down",
                "h22->h21:left",
                "h21->h20:left",
            ],
        )
        self.assertTrue(arc.validate_fixed_strong_synapses())
        self.assertTrue(arc.validate_intrinsic_synapses())
        self.assertEqual(len(arc.intrinsic_synapses()), 8)
        self.assertTrue(all(synapse.intrinsic for synapse in arc.synapses))
        self.assertEqual(arc.auxiliary_coords(), [])

    def test_nine_h_start_corner_determines_unique_path(self) -> None:
        expected_paths = {
            (0, 0): [
                "h00->h01:right",
                "h01->h02:right",
                "h02->h12:down",
                "h12->h11:left",
                "h11->h10:left",
                "h10->h20:down",
                "h20->h21:right",
                "h21->h22:right",
            ],
            (0, 2): [
                "h02->h01:left",
                "h01->h00:left",
                "h00->h10:down",
                "h10->h11:right",
                "h11->h12:right",
                "h12->h22:down",
                "h22->h21:left",
                "h21->h20:left",
            ],
            (2, 0): [
                "h20->h21:right",
                "h21->h22:right",
                "h22->h12:up",
                "h12->h11:left",
                "h11->h10:left",
                "h10->h00:up",
                "h00->h01:right",
                "h01->h02:right",
            ],
            (2, 2): [
                "h22->h21:left",
                "h21->h20:left",
                "h20->h10:up",
                "h10->h11:right",
                "h11->h12:right",
                "h12->h02:up",
                "h02->h01:left",
                "h01->h00:left",
            ],
        }
        for start_coord, path in expected_paths.items():
            arc = NineHReflexArc.standard(start_coord=start_coord)
            self.assertEqual(arc.start_coord, start_coord)
            self.assertEqual(arc.unique_internal_path(), path)
            self.assertEqual(arc.external_output_coords(), [NineHReflexArc.path_coords_for(start_coord)[-1]])
            self.assertTrue(arc.validate_intrinsic_synapses())

    def test_nine_h_path_has_complex_direction_operator(self) -> None:
        arc = NineHReflexArc.standard(start_coord=(0, 2))
        self.assertEqual(
            arc.path_direction_operators(),
            [
                direction_operator("left"),
                direction_operator("left"),
                direction_operator("down"),
                direction_operator("right"),
                direction_operator("right"),
                direction_operator("down"),
                direction_operator("left"),
                direction_operator("left"),
            ],
        )
        self.assertLess(abs(arc.net_path_operator() + 1), 1e-9)

    def test_nine_h_path_phase_flip_matches_single_h_body_flip(self) -> None:
        arc = NineHReflexArc.standard(start_coord=(0, 2))
        neuron = HNeuron.from_state("a", "A")
        before = neuron.bit

        neuron.flip_body()

        self.assertEqual(arc.net_path_operator(), -1 + 0j)
        self.assertEqual(before, 0)
        self.assertEqual(neuron.bit, 1)

    def test_nine_h_reflex_arc_rejects_non_corner_start(self) -> None:
        with self.assertRaises(ValueError):
            NineHReflexArc.standard(start_coord=(1, 1))

    def test_nine_h_reflex_arc_strong_stimulus_reaches_effector(self) -> None:
        arc = NineHReflexArc.standard()
        result = arc.run_external_stimulus(ExternalStimulus.STRONG)

        self.assertTrue(result.effector_activated)
        self.assertEqual(result.response.value, "reflex_response")
        self.assertEqual([outcome for _, outcome in result.steps], [Outcome.MAIN] * 8)

    def test_nine_h_weak_stimulus_accumulates_only_in_receptor(self) -> None:
        arc = NineHReflexArc.standard()

        first = arc.run_external_stimulus(ExternalStimulus.WEAK)
        self.assertFalse(first.effector_activated)
        self.assertEqual(first.steps, [])

        second = arc.run_external_stimulus(ExternalStimulus.WEAK)
        self.assertFalse(second.effector_activated)
        self.assertEqual(second.steps, [])

        third = arc.run_external_stimulus(ExternalStimulus.WEAK)
        self.assertTrue(third.effector_activated)
        self.assertEqual(third.response.value, "reflex_response")
        self.assertTrue(all(outcome == Outcome.MAIN for _, outcome in third.steps))

    def test_nine_h_receptor_none_stimulus_does_not_activate_effector(self) -> None:
        arc = NineHReflexArc.standard()
        result = arc.run_external_stimulus(ExternalStimulus.NONE)

        self.assertFalse(result.effector_activated)
        self.assertEqual(result.response.value, "none")
        self.assertEqual(result.steps, [])

    def test_weak_stimulus_outputs_only_on_third_pulse(self) -> None:
        source = HNeuron.from_state("source", "D")
        target = HNeuron.from_state("target", "C")
        syn = Synapse("source", "target", "right")
        neurons = {"source": source, "target": target}

        syn.stimulate(neurons, Signal.WEAK)
        self.assertFalse(target.last_output)
        self.assertTrue(target.axon_hillock_active())
        self.assertEqual(target.phase, Phase.T1)
        self.assertAlmostEqual(target.membrane.potential(), 1.0 / 3.0)
        self.assertAlmostEqual(target.membrane.phase_potential(), math.pi / 3)
        self.assertEqual(target.grid[target.row_containing(1)][1], "4*")
        self.assertEqual(target.body, tuple(target.original[1]))
        syn.stimulate(neurons, Signal.WEAK)
        self.assertFalse(target.last_output)
        self.assertFalse(target.axon_hillock_active())
        self.assertEqual(target.phase, Phase.T1)
        self.assertAlmostEqual(target.membrane.potential(), 2.0 / 3.0)
        self.assertAlmostEqual(target.membrane.phase_potential(), 2 * math.pi / 3)
        self.assertEqual(target.grid[target.row_containing(1)][1], "4*")
        self.assertNotEqual(target.body, tuple(target.original[1]))
        syn.stimulate(neurons, Signal.WEAK)
        self.assertTrue(target.last_output)
        self.assertTrue(target.axon_hillock_active())
        self.assertEqual(target.phase, Phase.R0)
        self.assertAlmostEqual(target.membrane.potential(), 0.0)
        self.assertAlmostEqual(target.membrane.phase_potential(), 0.0)
        self.assertIsNone(target.grid[0][1])
        self.assertIsNone(target.grid[2][1])
        self.assertEqual(target.body, tuple(target.original[1]))
        self.assertEqual(target.grid, target.original)

    def test_uncontinued_weak_signal_decays_by_timing(self) -> None:
        source = HNeuron.from_state("source", "D")
        target = HNeuron.from_state("target", "C")
        syn = Synapse("source", "target", "right")
        network = HNetwork({"source": source, "target": target}, [syn])

        network.step([("source->target:right", Signal.WEAK)])
        self.assertNotEqual(target.grid, target.original)

        network.step([])
        self.assertEqual(target.grid, target.original)
        self.assertFalse(target.last_output)
        self.assertEqual(target.phase, Phase.T3)

    def test_schrodinger_form_body_evolution_preserves_normalized_state(self) -> None:
        state = normalize_state([1 + 0j, 0 + 0j])
        evolved = schrodinger_step(state, body_flip_hamiltonian(omega=1.0), dt=0.1)
        probability = sum(abs(value) ** 2 for value in evolved)

        self.assertAlmostEqual(probability, 1.0)
        self.assertGreater(abs(evolved[1]), 0.0)
        self.assertLess(abs(evolved[0]), 1.0)

    def test_complex_ginzburg_landau_parent_dynamics(self) -> None:
        field = [1 + 0j]
        evolved = complex_ginzburg_landau_step(
            field,
            dt=0.1,
            alpha=1.0,
            omega=0.0,
            beta=1.0,
            gamma=0.0,
            diffusion=0.0,
        )
        self.assertAlmostEqual(abs(evolved[0]), 1.0)

        coupled = complex_ginzburg_landau_step(
            [1 + 0j, 0 + 0j],
            dt=0.1,
            alpha=0.0,
            omega=0.0,
            beta=0.0,
            gamma=0.0,
            diffusion=1.0,
        )
        self.assertLess(abs(coupled[0]), 1.0)
        self.assertGreater(abs(coupled[1]), 0.0)

    def test_reduced_wave_equation_matches_single_site_parent_dynamics(self) -> None:
        state = digit_state(1)
        reduced = reduced_wave_step(
            state,
            dt=0.1,
            alpha=1.0,
            omega=0.4,
            beta=1.0,
            gamma=0.0,
        )
        parent = complex_ginzburg_landau_step(
            [state],
            dt=0.1,
            alpha=1.0,
            omega=0.4,
            beta=1.0,
            gamma=0.0,
            diffusion=0.0,
        )[0]

        self.assertAlmostEqual(reduced.real, parent.real)
        self.assertAlmostEqual(reduced.imag, parent.imag)


if __name__ == "__main__":
    unittest.main()
