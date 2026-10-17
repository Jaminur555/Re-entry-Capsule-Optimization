# Re-entry Capsule Shape Optimization

Python code for the shape optimization of a ballistic re-entry capsule, both
deterministic and under uncertainty. The project extends a defended BSc thesis
and is being prepared for submission to Acta Astronautica.

## What it is

- An aerothermodynamic model for axisymmetric sphere-cone capsules using the local
  inclination method: modified Newtonian pressure on the nose, the exact
  Taylor–Maccoll tangent-cone solution on the afterbody, and Prandtl–Meyer on the
  lee side; Sutton–Graves stagnation and Chapman shoulder convective heating; a
  US76 atmosphere with a hypersonic extension.
- A 3-DoF entry trajectory simulation over a rotating Earth with rate-limited
  no-skip bank guidance, and a trim angle-of-attack solver.
- A validation suite that compares the model against digitized Apollo data from
  Dirkx & Mooij, *Aerothermodynamic Design and Assessment of Space Vehicles*
  (Springer, 2017).
- Deterministic multi-objective shape optimization with NSGA-II: volume
  efficiency, integrated heat load, and ground track as objectives, with
  constraints on peak heat rates and load factor.
- An uncertainty quantification layer: uncertain entry state and aerodynamic
  model-form factors, a design of experiments of trajectory simulations,
  Gaussian-process surrogate models with an out-of-fold accuracy gate, and a
  robust moment-based optimization formulation. The deterministic and robust
  results are compared and verified with Monte-Carlo simulation on the full
  trajectory model.
- The journal manuscript (LaTeX, elsarticle) in `paper/manuscript/`.

## Repository layout

```
capsule_opt/        simulation package
  aerodynamics/     pressure laws (Newtonian, Taylor–Maccoll, Prandtl–Meyer)
  heating/          convective heating models
  atmosphere/       US76 + hypersonic extension
  trajectory/       3-DoF entry simulation
  trim/             trim angle-of-attack solver
  geometry/         capsule parameterization
  optimization/     objectives, constraints, metrics
  uncertainty/      uncertainty distributions + sampler
validation/         validation suite vs Dirkx & Mooij (2017)
uq/                 uncertainty quantification pipeline
main.py             deterministic optimization entry point
results/            validation figures
data/cache/         Taylor–Maccoll lookup table
paper/              manuscript + publication assets
```

## Running

Requires Python 3.10 or newer: `pip install -r requirements.txt`

```bash
# validation suite
python -m validation.build_reference_data
python -m validation.aerodynamics.validate_apollo_coefficients
python -m validation.system.validate_apollo_end_to_end

# deterministic shape optimization
python main.py

# uncertainty pipeline (DOE → descriptors → surrogate build)
python -m uq.doe 500
python -m uq.doe_descriptors
python -m uq.build_surrogate
```

## Reference

M. A. Dirkx and E. Mooij, *Aerothermodynamic Design and Assessment of Space
Vehicles*, Springer, 2017 — validation source.
