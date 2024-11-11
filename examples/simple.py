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
    n_steps = 5000
    temperature = 100 * unit.kelvin
    thermostat_parameter = 1.0 / unit.picoseconds
    npt = False
    pressure = 1 * unit.bar
    barostat_update = 25
    n_screen = 1000
    n_traj = 10000
    n_file = 10000
    coordinates = "coord.pdb"

    geometry = Slab(kappa=1000, species="Kr", axis="z", width=3.5, centre=10.3)

    osmotic_config = OsmoticConfig(
        temperature=temperature,
        geometry=geometry,
        osmotic_pressure=40,
        report_interval=1000,
        compute_interval=1000,
        sample_length=1000,
        tau=1.0,
        # restart="osmotic.0.out",
    )

    # Forcefield paramters for Ar and Kr
    atom_types = [
        {"name": "Ar", "mass": 39.95, "sigma": 0.3405, "epsilon": 0.996015},
        {"name": "Kr", "mass": 39.95, "sigma": 0.3405, "epsilon": 0.996015},
    ]
    atom_names = [x["name"] for x in atom_types]

    # Read the coordinates
    pdb = app.PDBFile(coordinates)

    # Create the system object
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    for a in pdb.topology.atoms():
        idx = atom_names.index(a.name)
        system.addParticle(atom_types[idx]["mass"] * unit.amu)

    # Create the forcefield
    M = len(atom_types)
    epsilonAR_r = np.zeros((M, M), dtype="float64")
    sigmaAR_r = np.zeros((M, M), dtype="float64")

    for i in range(M):
        for j in range(i, M):
            epsilonAR_r[i][j] = np.sqrt(
                atom_types[i]["epsilon"] * atom_types[i]["epsilon"]
            )
            epsilonAR_r[j][i] = epsilonAR_r[i][j]
            sigmaAR_r[i][j] = 0.5 * (atom_types[i]["sigma"] + atom_types[i]["sigma"])
            sigmaAR_r[j][i] = sigmaAR_r[i][j]

    # The Lennard-Jones potential we will create in OpenMM accepts these arrays in list form
    epsilonLST_r = (epsilonAR_r).ravel().tolist()
    sigmaLST_r = (sigmaAR_r).ravel().tolist()

    # Create the force object
    cnbf = mm.CustomNonbondedForce(
        "4*eps*((sig/d)^12-(sig/d)^6); d=max(0.1,r); eps=epsilon(type1, type2); sig=sigma(type1, type2)"
    )
    # cnbf.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    cnbf.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    cnbf.setCutoffDistance(1.0 * unit.nanometer)
    cnbf.addTabulatedFunction("epsilon", mm.Discrete2DFunction(M, M, epsilonLST_r))
    cnbf.addTabulatedFunction("sigma", mm.Discrete2DFunction(M, M, sigmaLST_r))
    cnbf.addPerParticleParameter("type")

    for a in pdb.topology.atoms():
        idx = atom_names.index(a.name)
        cnbf.addParticle([idx])

    # Add the force to the system
    system.addForce(cnbf)

    # list_of_solute_pecies = ["Ar"]
    # atoms_list = [
    #     atom.index
    #     for atom in pdb.topology.atoms()
    #     if atom.name in list_of_solute_pecies
    # ]

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
        # mm.Platform.getPlatformByName("CPU"),
        # mm.Platform.getPlatformByName("OpenCL"),
        mm.Platform.getPlatformByName("CUDA"),
        # mm.Platform.getPlatformByName("HIP"),
        {"Precision": "mixed"},
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
