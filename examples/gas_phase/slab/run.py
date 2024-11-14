import sys
import numpy as np
import logging
import openmm as mm
import openmm.unit as unit
import openmm.app as app
from opo import OsmoticPressureReporter, OsmoticConfig, Slab

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def main():
    # Simulation parameters
    timestep = 0.002 * unit.picosecond
    n_steps = 1e6
    temperature = 300 * unit.kelvin
    thermostat_parameter = 1.0 / unit.picoseconds
    npt = False
    pressure = 1 * unit.bar
    barostat_update = 25
    n_screen = 1000
    n_traj = 10000
    n_file = 10000
    coordinates = "coord.pdb"

    geometry = Slab(kappa=1000, species="Ar", width=13.86329, centre=45, axis="z")

    osmotic_config = OsmoticConfig(
        temperature=temperature,
        geometry=geometry,
        osmotic_pressure=1,
        report_interval=1000,
        compute_interval=1000,
        sample_length=1000,
        tau=1.0,
        restart="restart.out",
    )

    # Read the coordinates
    pdb = app.PDBFile(coordinates)
    forcefield = app.ForceField("../argon.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0 * unit.nanometer)

    for n, f in enumerate(system.getForces()):
        f.setForceGroup(n + 1)

    # Create the integrator object
    integrator = mm.LangevinMiddleIntegrator(
        temperature, thermostat_parameter, timestep
    )

    # Add the barostat for NPT simulation
    if npt:
        system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_update))

    # Create the simulation object
    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
        # mm.Platform.getPlatformByName("OpenCL"),
        # mm.Platform.getPlatformByName("CUDA"),
        # mm.Platform.getPlatformByName("HIP"),
        # {"Precision": "mixed"},
    )

    # Add the velocities to the simulation
    simulation.context.setPositions(pdb.positions)

    # Screen output
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            n_screen,
            totalSteps=int(n_steps),
            separator="\t",
            step=False,
            time=True,
            potentialEnergy=False,
            kineticEnergy=False,
            totalEnergy=False,
            temperature=True,
            volume=False,
            density=True,
            progress=True,
            remainingTime=True,
            speed=True,
            elapsedTime=False,
        )
    )

    # File output
    simulation.reporters.append(
        app.StateDataReporter(
            "md.log",
            n_file,
            separator=",",
            step=False,
            time=True,
            potentialEnergy=True,
            kineticEnergy=False,
            totalEnergy=False,
            temperature=True,
            volume=True,
            density=True,
            progress=False,
            remainingTime=False,
            speed=False,
            elapsedTime=False,
        )
    )

    # Trajectory output
    simulation.reporters.append(app.DCDReporter("traj.dcd", n_traj))
    osmotic_config = OsmoticPressureReporter(
        context=simulation.context, topology=pdb.topology, config=osmotic_config
    )
    simulation.reporters.append(osmotic_config)

    # Energy minimisation
    # e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    # LOG.info("Initial energy: %s", e)
    # simulation.minimizeEnergy(tolerance=0.001)
    # e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    # LOG.info("Energy after minimisation: %s", e)

    # Generate the velocities for the simulation
    r = np.random.randint(1, 99999)
    simulation.context.setVelocitiesToTemperature(temperature, r)

    # Run molecular dynamics
    simulation.step(int(n_steps))


if __name__ == "__main__":
    main()
