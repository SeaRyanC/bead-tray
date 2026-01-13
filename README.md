# bead-tray

A bead sorting tray that, when shaken, aligns cylindrical beads coaxially into neat rows. Designed for 3D printing with PLA.

## Overview

The tray features V-shaped grooves that guide beads into aligned positions when the tray is shaken. The design is parametric and defined in OpenSCAD.

## Files

- `tray.scad` - OpenSCAD source file for the tray geometry
- `bead_simulator/` - Rust physics simulator for optimization

## Physical Parameters

- **Tray mass**: 10.45 grams (when printed in PLA)
- **Bead dimensions**: 2.8mm length, 2.5mm outer diameter, 1.5mm inner diameter
- **Bead weight**: 0.01 grams each
- **Friction coefficient**: tan(55°) ≈ 1.43 (bead on PLA surface)

## Optimization Simulator

The `bead_simulator` directory contains a Rust physics simulator that evaluates how well the tray aligns beads when shaken.

### Building

```bash
cd bead_simulator
cargo build --release
```

### Usage

```bash
# Run single simulation with default configuration
./target/release/bead_simulator

# Run full optimization (245 configurations)
./target/release/bead_simulator optimize

# Run quick test (8 configurations)
./target/release/bead_simulator quick-test

# Show help
./target/release/bead_simulator help
```

### Dependencies

- **Rust** (1.70+)
- **OpenSCAD** (for STL generation)
- **Xvfb** (for headless OpenSCAD rendering)

Install on Ubuntu/Debian:
```bash
sudo apt-get install openscad xvfb
```

### Simulation Model

The simulator models:
- 100 cylindrical beads dropped onto the tray
- Shaking motion at 3 Hz with 15mm amplitude
- Tray tilted 10° toward the closed end
- Friction, gravity, bead-bead collisions
- Groove constraint forces

### Optimization Parameters

- `bead_dia` (2.4 - 2.8 mm): Groove diameter in the SCAD file (simulated beads stay at 2.5mm)
- `depth_factor` (0.30 - 0.60): How deep the grooves are relative to groove diameter
- `row_spacing` (0.2 - 0.8 mm): Spacing between groove centers beyond groove diameter

Note: The `bead_dia` parameter affects groove size in the generated tray, but the simulated beads always use the physical bead outer diameter (2.5mm). This allows optimizing groove fit tolerance.

### Output

Results are saved to `optimization_results.json` containing:
- Configuration parameters tested
- Alignment score (0-100%)
- Number of beads settled in grooves
- Average axial alignment
- Simulation time

## Generating STL

```bash
openscad -o tray.stl tray.scad
```

## License

MIT