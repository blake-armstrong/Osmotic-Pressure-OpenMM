# OpenMM Osmotic Pressure Regulator

[![DOI](https://zenodo.org/badge/884747574.svg)](https://doi.org/10.5281/zenodo.15515783)

A Python package for calculating osmotic pressure in molecular dynamics simulations using OpenMM. This package allows you to add osmotic pressure forces to your simulations with various geometries including planes, slabs, cylinders, and spheres.

## Features

- Calculate osmotic pressure during molecular dynamics simulations
- Support for multiple geometry types:
  - Planar membranes
  - Slab geometries
  - Cylindrical containers
  - Spherical containers
- Configurable parameters for pressure calculations
- Integration with OpenMM simulation framework
- Support for restart capabilities
- Real-time pressure reporting

## Installation

The package requires OpenMM. Follow the conda instructions at http://docs.openmm.org/latest/userguide/application/01_getting_started.html#installing-openmm. 

Once installed and the conda environment is activated, use pip to install the osmotic reporter from the main directory.

```bash
pip install .
```

## Usage

### Basic Setup

```python
from opo import OsmoticPressureReporter, OsmoticConfig, Cylinder  # or other geometries

# Define your geometry
geometry = Cylinder(
    kappa=1000,            # Force constant
    species="Ar",          # Particle type to apply force to
    radius=28.19903,       # Cylinder radius
    centre=[45, 45],       # Center coordinates
    axis="y"               # Cylinder axis
)

# Configure osmotic pressure calculation
osmotic_config = OsmoticConfig(
    temperature=300 * unit.kelvin,
    geometry=geometry,
    osmotic_pressure=1,
    report_interval=1000,
    compute_interval=1000,
    sample_length=1000,
    tau=1.0
)

# Add to simulation
osmotic_reporter = OsmoticPressureReporter(
    context=simulation.context,
    topology=pdb.topology,
    config=osmotic_config
)
simulation.reporters.append(osmotic_reporter)
```

### Available Geometries

#### Plane
```python
from opo import Plane

plane = Plane(
    kappa=1000,      # Force constant
    species="Ar",    # Particle type
    pos=45,          # Position along axis
    axis="z",        # Normal axis
    direction="-"    # Solute localised below plane (-), solute localised above plane (+)
)
```

#### Slab
```python
from opo import Slab

slab = Slab(
    kappa=1000,      # Force constant
    species="Ar",    # Particle type
    width=13.86329,  # Slab width
    centre=45,       # Center position
    axis="z"         # Normal axis
)
```

#### Cylinder
```python
from opo import Cylinder

cylinder = Cylinder(
    kappa=1000,            # Force constant
    species="Ar",          # Particle type
    radius=28.19903,       # Cylinder radius
    centre=[45, 45],       # Center coordinates (perpendicular to axis)
    axis="y"               # Cylinder axis
)
```

#### Sphere
```python
from opo import Sphere

sphere = Sphere(
    kappa=1000,                # Force constant
    species="Ar",              # Particle type
    radius=37.62109,           # Sphere radius
    centre=[45, 45, 45]        # Center coordinates
)
```

### Configuration Options

The `OsmoticConfig` class accepts the following parameters:

- `temperature`: System temperature (required)
- `geometry`: Geometry object (required)
- `osmotic_pressure`: Target osmotic pressure (required)
- `tau`: Time constant for pressure coupling (default: 1.0)
- `file`: Output file name (default: "osmotic.out")
- `restart`: Path to restart file (optional)
- `compressibility`: System compressibility (default: 0.01)
- `compute_interval`: Steps between force computations (default: 1000)
- `report_interval`: Steps between output writes (default: 1000)
- `sample_length`: Number of samples for averaging (default: 1000)
- `gcmd`: Enable/disable GCMD algorithm (default: True)

### Output Format

The osmotic pressure calculation outputs the following columns:

```
Time        Pi(t)       <Pi>        Pi(gcmd)    Parm        Area        Volume      Press
```

Where:
- `Time`: Simulation time in ps
- `Pi(t)`: Instantaneous osmotic pressure
- `<Pi>`: Average osmotic pressure
- `Pi(gcmd)`: GCMD-averaged osmotic pressure
- `Parm`: Scaled parameter value
- `Area`: Surface area of the container
- `Volume`: Volume of the container
- `Press`: Predicted pressure

## Implementation Details

The package implements the following key components:

1. `OsmoticPressureReporter`: Main class for pressure calculation and reporting
2. `GCMD`: Grand Canonical Molecular Dynamics algorithm
3. `GeometryInterpreter`: Handles different geometry types and force calculations
4. `OsmoticConfig`: Configuration management for the simulation

## Examples

The package includes example scripts demonstrating usage with different geometries:

1. Cylindrical container simulation
2. Spherical container simulation
3. Slab geometry simulation

Each example shows how to:
- Set up the simulation parameters
- Configure the geometry
- Initialise the osmotic pressure reporter
- Run the simulation

