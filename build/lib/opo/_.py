#!/usr/bin/env python3

### Copyright (C) 2023  Paolo Raiteri

### This program is free software: you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation, either version 3 of the License, or
### (at your option) any later version.

### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.

### You should have received a copy of the GNU General Public License
### along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit


def getAtoms(config, topology):
    IDlst = []
    if "residue" in config and len(config["residue"]) > 0:
        if isinstance(config["residue"], str):
            config["residue"] = config["residue"].split(",")
        for a in topology.atoms():
            if a.residue.name in config["residue"]:
                IDlst.append(a.index)

    if "species" in config and len(config["species"]) > 0:
        if isinstance(config["species"], str):
            config["species"] = config["species"].split(",")
        for iatm in topology.atoms():
            if iatm.name in config["species"]:
                IDlst.append(iatm.index)

    if "indices" in config:
        if isinstance(config["indices"], list):
            if len(config["indices"]) == 0:
                config["indices"] = []
        elif isinstance(config["indices"], str):
            config["indices"] = [int(x) for x in config["indices"].split(",")]
        elif isinstance(config["indices"], int):
            config["indices"] = [config["indices"]]

        IDlst += config["indices"]

    return IDlst


def inputToList(input):
    if type(input) is list:
        return input

    input = str(input)
    result = ""
    for char in input:
        if char not in "[](){}":
            result += char
    return result.split(",")


def positionsRestraint(name, settings, system, topology):
    forceName = name
    print("  restraint name: {}".format(forceName))

    variables = inputToList(settings["par"])
    values = [float(x) for x in inputToList(settings["val"])]

    globalParameters = []
    if "global" in settings:
        for p in settings["global"].split(","):
            idx = variables.index(p)
            globalParameters.append([variables[idx], values[idx]])
            variables.pop(idx)
            values.pop(idx)
    expression = ""
    if "fullexp" in settings:
        expression = settings["fullexp"]
        energy = ""
        distance = ""

    elif "exp" in settings:
        energy = settings["exp"]
        distance = " d = periodicdistance(x,y,z,x0,y0,z0)"

    else:
        energy = "k*d^2"
        distance = " d = periodicdistance(x,y,z,x0,y0,z0)"
        f = "global" in settings and "d0" in settings["global"]
        if "d0" in variables or f:
            energy = energy.replace("d", "(max(0,d-d0))")

    atomsList = getAtoms(settings, topology)
    parameters = [values] * len(atomsList)

    if "fullexp" not in settings:

        if "x0" not in variables:
            distance = distance.replace("x0", "x")
        if "y0" not in variables:
            distance = distance.replace("y0", "y")
        if "z0" not in variables:
            distance = distance.replace("z0", "z")
        expression = energy + " ; " + distance

    print("  expression: {}".format(expression.replace(";", "\n  ")))
    for x, y in zip(variables, values):
        print("    {} = {}".format(x, y))

    if len(globalParameters) > 0:
        for x in globalParameters:
            print("    {} = {}".format(*x))

    if "file" in settings:
        print("  atoms selected from file: {}".format(settings["file"]))
    elif "indices" in settings:
        print("  selected atoms: {}".format(settings["indices"]))
    elif "species" in settings:
        print("  selected species: {}".format(settings["species"]))

    force = mm.CustomExternalForce(expression)

    if len(globalParameters) > 0:
        for x in globalParameters:
            force.addGlobalParameter(*x)

    for x in variables:
        force.addPerParticleParameter(x)

    print("  Number of atoms selected: {}".format(len(atomsList)))

    for i, j in zip(atomsList, parameters):
        if type(j) is tuple:
            j = list(j)
        elif type(j) is not list:
            j = [j]
        force.addParticle(i, j)

    force.setName(forceName)
    system.addForce(force)


class OsmoticPressureReporter(object):
    def __init__(self, runID, config, temperature, topology, context):
        print("osmotic pressure calculation ...")

        self._context = context
        system = self._context.getSystem()
        self._force = "osmoticWall"
        if "sphere" in config["geometry"]:
            self._geometry = "SPHERE"
            val = ",".join(
                [
                    str(config["geometry"]["sphere"]["kappa"]),
                    str(config["geometry"]["sphere"]["radius"]),
                    config["geometry"]["sphere"]["centre"],
                ]
            )
            cmd = {
                self._force: {
                    "global": "k,d0",
                    "par": "k,d0,x0,y0,z0",
                    "species": config["geometry"]["sphere"]["species"],
                    "val": val,
                    "exp": "0.5*k*(max(0,d-d0))^2",
                }
            }

        elif "plane" in config["geometry"]:
            self._geometry = "PLANE"
            val = ",".join(
                [
                    str(config["geometry"]["plane"]["kappa"]),
                    str(config["geometry"]["plane"]["pos"]),
                    "0.0",
                ]
            )
            p = config["geometry"]["plane"]["axis"]
            p0 = p + "0"
            expr = "0.5*k*max(0," + p + "-d0)^2"
            cmd = {
                self._force: {
                    "global": "k,d0",
                    "par": "k,d0," + p0,
                    "species": config["geometry"]["plane"]["species"],
                    "val": val,
                    "fullexp": expr,
                }
            }

        elif "slab" in config["geometry"]:
            self._geometry = "SLAB"
            val = ",".join(
                [
                    str(config["geometry"]["slab"]["kappa"]),
                    str(config["geometry"]["slab"]["width"]),
                    str(config["geometry"]["slab"]["centre"]),
                ]
            )
            p = config["geometry"]["slab"]["axis"]
            p0 = p + "0"
            expr = "0.5*k*(max(0,d-d0))^2;d=sqrt((" + p + "-" + p0 + ")^2)"
            cmd = {
                self._force: {
                    "global": "k,d0",
                    "par": "k,d0," + p0,
                    "species": config["geometry"]["slab"]["species"],
                    "val": val,
                    "fullexp": expr,
                }
            }
        elif "cylinder" in config["geometry"]:
            self._geometry = "CYLINDER"
            val = ",".join(
                [
                    str(config["geometry"]["cylinder"]["kappa"]),
                    str(config["geometry"]["cylinder"]["radius"]),
                    str(config["geometry"]["cylinder"]["centre"]),
                ]
            )

            p = config["geometry"]["cylinder"]["axis"]
            if p.upper() == "X":
                expr = "0.5*k*(max(0,d-d0))^2;d=sqrt((y-y0)^2+(z-z0)^2)"
                p0 = "y0,z0"
            elif p.upper() == "Y":
                expr = "0.5*k*(max(0,d-d0))^2;d=sqrt((x-x0)^2+(z-z0)^2)"
                p0 = "x0,z0"
            elif p.upper() == "Z":
                expr = "0.5*k*(max(0,d-d0))^2;d=sqrt((x-x0)^2+(y-y0)^2)"
                p0 = "x0,y0"
            else:
                raise RuntimeError()

            cmd = {
                self._force: {
                    "global": "k,d0",
                    "par": "k,d0," + p0,
                    "species": config["geometry"]["cylinder"]["species"],
                    "val": val,
                    "fullexp": expr,
                }
            }

        else:
            raise ValueError("No geometry.")

        # Add restraint, need to reinitialise the context
        positionsRestraint(self._force, cmd[self._force], system, topology)
        self._context.reinitialize(preserveState=True)

        print("  {:40s} = {}".format("force name", self._force))

        self._forceGroup = set()
        for force in system.getForces():
            if force.getName() == self._force:
                self._forceGroup.add(force.getForceGroup())
                self._restraintForce = force
                break
        if not self._forceGroup:
            raise ValueError("Found no force for osmotic calculation.")

        f = "osmotic.{}.out".format(runID)
        if "file" in config:
            f = config["output"]
        self._out = open(f, "w")
        print("  {:40s} = {}".format("output file", f))

        self._computeInterval = 1000
        if "computeInterval" in config:
            self._computeInterval = config["computeInterval"]

        self._reportInterval = self._computeInterval
        if "reportInterval" in config:
            self._reportInterval = config["reportInterval"]

        self._sampleLength = 1000
        if "sampleLength" in config:
            self._sampleLength = config["sampleLength"]

        assert (
            self._reportInterval >= self._computeInterval
        ), "reportInterval should be larger than computeInterval"
        assert (
            self._reportInterval % self._computeInterval == 0
        ), "reportInterval should be a multiple of computeInterval"

        print("  {:40s} = {}".format("output frequency", self._computeInterval))

        print("  {:40s} = {}".format("geometry", self._geometry.lower()))

        if self._geometry.upper() != "SPHERE":

            try:
                self._dir = config["geometry"][self._geometry.lower()]["axis"]
            except:
                raise Exception(
                    "Missing axis for {} geometry".format(self._geometry.upper())
                )

            if self._dir.upper() == "X":
                self._dirs = (2, 1, 0)
            elif self._dir.upper() == "Y":
                self._dirs = (0, 2, 1)
            elif self._dir.upper() == "Z":
                self._dirs = (0, 1, 2)
            else:
                raise Exception(
                    "Unknown crystallographic direction for {} ({})".format(
                        self._geometry.upper(), self._dir
                    )
                )

            print(
                "  {:40s} = {}".format(
                    "direction normal to the surface ", self._dir.upper()
                )
            )

        self._localCounter = 0
        self._averagePressure = 0.0

        # For spherical restraint
        self._T = temperature
        self._kt = (unit.MOLAR_GAS_CONSTANT_R * self._T).in_units_of(
            unit.kilojoules_per_mole
        )
        self._theta = (
            2 * self._kt / (self._context.getParameter("k") * unit.kilojoules_per_mole)
        )
        # self._theta = self._theta_value

        self._gcmd_parm = "d0"
        self._format = (
            "{:12.1f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
        )
        header = "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}\n".format(
            "#Time", "Pi(t)", "<Pi>", "Pi(gcmd)", "Parm", "Volume", "Press"
        )

        # For spherical restraint
        self._T = temperature
        self._kt = (unit.MOLAR_GAS_CONSTANT_R * self._T).in_units_of(
            unit.kilojoules_per_mole
        )
        self._factor_0 = self._kt / (
            self._context.getParameter("k") * unit.kilojoules_per_mole
        )
        self._factor_1 = np.sqrt(np.pi * self._factor_0 / 2)

        if "gcmd" not in config:
            self._osmoticPressure = np.zeros(self._sampleLength)
            self._out.write(header)
            self._out.flush()
            self._gcmd = False
            return

        self._gcmd = config["gcmd"]

        defaults = {
            "pext": 0.0,
            "K": 0.01,
            "tau": 1.0,
            "sample": 100,
        }
        for k, v in self._gcmd.items():
            try:
                defaults[k] = v
            except:
                raise Exception("Unknown GCMD parameter")

        self._gcmd_pext = float(defaults["pext"])
        self._gcmd_compressibility = 0.01  # <- this can be changed
        self._gcmd_tau = float(defaults["tau"])
        self._gcmd_sample = int(defaults["sample"])
        self._gcmd_dt = (
            context.getIntegrator().getStepSize().value_in_unit(unit.picosecond)
        )

        self._osmoticPressure = np.full(self._sampleLength, self._gcmd_pext)

        print("#--- Constant Osmotic Pressure --------#")
        print("  {:40s} = {}".format("Target Pi (bar)           ", self._gcmd_pext))
        print("  {:40s} = {}".format("Restraint global parameter", self._gcmd_parm))
        print(
            "  {:40s} = {}".format(
                "Compressibility (1/bar)   ", self._gcmd_compressibility
            )
        )
        print("  {:40s} = {}".format("Timestep (ps)             ", self._gcmd_dt))
        print(
            "  {:40s} = {}".format("Sampling frequency        ", self._computeInterval)
        )
        print("  {:40s} = {}".format("Average length            ", self._gcmd_sample))

        if "restart" in self._gcmd:
            if not os.path.exists(self._gcmd["restart"]):
                raise Exception(
                    "GCMD: missing restart file ({})".format(self._gcmd["restart"])
                )
            with open(self._gcmd["restart"], "r") as f:
                lines = f.readlines()
            if lines:
                values = lines[-1].split()
            else:
                raise Exception(
                    "GCMD: cannot restart from file ({})".format(self._gcmd["restart"])
                )

            print("  {:40s} = {}".format("Restarting CGMD from", self._gcmd["restart"]))
            print("  {:40s} = {}".format("Membrane parameter", float(values[3])))
            print("  {:40s} = {}".format("Average osmotic pressure", float(values[4])))

            self._localCounter = self._gcmd_sample
            self._context.setParameter(self._gcmd_parm, float(values[4]))
            self._gcmd_press = float(values[3])

        self._out.write(header)
        self._out.flush()

        return

    def __del__(self):
        if hasattr(self, "_out"):
            self._out.close()

    def describeNextReport(self, simulation):
        steps = self._computeInterval - simulation.currentStep % self._computeInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        self._localCounter += 1
        stime = state.getTime().value_in_unit(unit.picosecond)

        # extract forces
        forces = simulation.context.getState(
            getForces=True, groups=self._forceGroup
        ).getForces(asNumpy=True)
        forces = forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        f = np.sum(np.sqrt(np.einsum("ij,ij->i", forces, forces)))
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        i, j, k = self._dirs
        # spherical restraining potential
        radius = self._context.getParameter(self._gcmd_parm)
        # spherical restraining potential
        if self._geometry.upper() == "SPHERE":
            area = (
                4
                * np.pi
                * (radius**2 + 2 * radius * self._factor_1 + 2 * self._factor_0)
            )
            volume = (
                4
                * np.pi
                * (
                    radius**3 / 3
                    + (radius**2 + self._factor_0) * self._factor_1
                    + radius * 2 * self._factor_0
                )
            )

        # # Harmonic restraint
        # elif self._geometry.upper() == "HARMONIC":
        #     area = box[i][i] * box[j][j]
        #     volume = area * self._factor_1

        # flat restraining potential (flat bottom well)
        elif self._geometry.upper() == "SLAB":
            area = box[i][i] * box[j][j]
            volume = area * 2 * (radius + self._factor_1)
            area *= 2

        # planar restraining potential
        elif self._geometry.upper() == "PLANE":
            area = box[i][i] * box[j][j]
            volume = 0

        # Cyclindrical flat bottom restraining potential
        elif self._geometry.upper() == "CYLINDER":
            rtmp = box[k][k] * 2 * np.pi
            area = rtmp * (radius + self._factor_1)
            volume = rtmp * (0.5 * radius**2 + radius * self._factor_1 + self._factor_0)
        else:
            raise RuntimeError()

        # convert 'kJ/mole/nm^3' to 'bar'
        opress = f / area * 16.605388
        self._osmoticPressure[(self._localCounter - 1) % self._sampleLength] = opress

        # Global average of the pressure
        self._averagePressure = (
            self._averagePressure * (self._localCounter - 1) + opress
        ) / self._localCounter

        # Window average of the pressure
        n = min(self._localCounter, self._sampleLength)
        self._gcmd_press = np.mean(self._osmoticPressure[:n])

        # GCMD
        if volume != 0:
            conc = (
                self._restraintForce.getNumParticles() / volume / 0.6022
            )  # atm/nm^3 -> mol/L
        else:
            conc = 0
        PI = conc * self._kt.value_in_unit(unit.kilojoule_per_mole) * 10  # kJ/L -> bar

        # Berendsen-like barostat
        # \mu = 1 - \frac{\kappa_T \delta t}{\tau_\Pi} (\Pi_0 -\Pi)
        # \kappa = 0.01 (bar^{-1}) (~ isothermal compressibilitty)
        # \delta t = simulation timestep
        # \tau_\Pi = 1.0 ps
        mu = 1.0
        if self._gcmd:
            mu = 1.0 - self._gcmd_compressibility * self._gcmd_dt / self._gcmd_tau * (
                self._gcmd_pext - self._gcmd_press
            )
        parm = mu * self._context.getParameter(self._gcmd_parm)
        if self._gcmd:
            self._context.setParameter(self._gcmd_parm, parm)

        if simulation.currentStep % self._reportInterval != 0:
            return

        self._out.write(
            self._format.format(
                stime,
                opress,
                self._averagePressure,
                self._gcmd_press,
                parm,
                volume,
                PI,
            )
        )
        self._out.flush()


# Simulation parameters
config = {
    "timestep": 0.002,
    "numberOfSteps": 3e8,
    "temperature": 100 * unit.kelvin,
    "thermostatParameter": 1.0 / unit.picoseconds,
    "NPT": False,
    "pressure": 1 * unit.bar,
    "barostatUpdate": 25,
    "screenReport": 10000,
    "trajReport": 10000,
    "fileReport": 10000,
    "coordinatesFile": "coord.pdb",
}

osmoticConfig = {
    "reportInterval": 1000,
    "computeInterval": 1000,
    "sampleLength": 10,
    "geometry": {
        "slab": {
            "axis": "z",
            "width": 3.5,
            "centre": 10.3,
            "kappa": 1000,
            "species": "Kr",
        }
    },
    "gcmd": {
        "parm": "d0",
        "pext": float(sys.argv[1]),
        "tau": 1.0,
    },
}

# Forcefield paramters for Ar and Kr
atomTypes = [
    {"name": "Ar", "mass": 39.95, "sigma": 0.3405, "epsilon": 0.996015},
    {"name": "Kr", "mass": 39.95, "sigma": 0.3405, "epsilon": 0.996015},
    # {"name":"Kr","mass":83.80,"sigma":0.3670,"epsilon":1.388420},
]
atomNames = [x["name"] for x in atomTypes]

# Read the coordinates
pdb = app.PDBFile(config["coordinatesFile"])

# Create the system object
system = mm.System()
system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
for a in pdb.topology.atoms():
    idx = atomNames.index(a.name)
    system.addParticle(atomTypes[idx]["mass"] * unit.amu)

# Create the forcefield
M = len(atomTypes)
epsilonAR_r = np.zeros((M, M), dtype="float64")
sigmaAR_r = np.zeros((M, M), dtype="float64")

for i in range(M):
    for j in range(i, M):
        epsilonAR_r[i][j] = np.sqrt(atomTypes[i]["epsilon"] * atomTypes[i]["epsilon"])
        epsilonAR_r[j][i] = epsilonAR_r[i][j]
        sigmaAR_r[i][j] = 0.5 * (atomTypes[i]["sigma"] + atomTypes[i]["sigma"])
        sigmaAR_r[j][i] = sigmaAR_r[i][j]

# The Lennard-Jones potential we will create in OpenMM accepts these arrays in list form
epsilonLST_r = (epsilonAR_r).ravel().tolist()
sigmaLST_r = (sigmaAR_r).ravel().tolist()

# Create the force object
customNonbondedForce = mm.CustomNonbondedForce(
    "4*eps*((sig/d)^12-(sig/d)^6); d=max(0.1,r); eps=epsilon(type1, type2); sig=sigma(type1, type2)"
)
# customNonbondedForce.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
customNonbondedForce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
customNonbondedForce.setCutoffDistance(1.0 * unit.nanometer)
customNonbondedForce.addTabulatedFunction(
    "epsilon", mm.Discrete2DFunction(M, M, epsilonLST_r)
)
customNonbondedForce.addTabulatedFunction(
    "sigma", mm.Discrete2DFunction(M, M, sigmaLST_r)
)
customNonbondedForce.addPerParticleParameter("type")

for a in pdb.topology.atoms():
    idx = atomNames.index(a.name)
    customNonbondedForce.addParticle([idx])

# Add the force to the system
system.addForce(customNonbondedForce)

listOfSoluteSpecies = ["Ar"]
atomsList = [
    atom.index for atom in pdb.topology.atoms() if atom.name in listOfSoluteSpecies
]

for n, f in enumerate(system.getForces()):
    f.setForceGroup(n + 1)

# Create the integrator object
integrator = mm.LangevinMiddleIntegrator(
    config["temperature"], config["thermostatParameter"], config["timestep"]
)

# Add the barostat for NPT simulation
if config["NPT"]:
    system.addForce(
        mm.MonteCarloBarostat(
            config["pressure"], config["temperature"], config["barostatUpdate"]
        )
    )

# Create the simulation object
simulation = app.Simulation(
    pdb.topology,
    system,
    integrator,
    # mm.Platform.getPlatformByName("OpenCL"),
    # mm.Platform.getPlatformByName("CUDA"),
    mm.Platform.getPlatformByName("HIP"),
    {"Precision": "mixed"},
)

# Add the velocities to the simulation
simulation.context.setPositions(pdb.positions)

# Screen output
simulation.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        config["screenReport"],
        totalSteps=int(config["numberOfSteps"]),
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
        config["fileReport"],
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
simulation.reporters.append(app.DCDReporter("traj.dcd", config["trajReport"]))
osmotic_config = OsmoticPressureReporter(
    runID=0,
    config=osmoticConfig,
    temperature=config["temperature"],
    topology=pdb.topology,
    context=simulation.context,
)
simulation.reporters.append(osmotic_config)

# Energy minimisation
e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(f"Initial energy {e}")
simulation.minimizeEnergy(tolerance=0.001)
e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(f"Energy after minimisation {e}")

# Generate the velocities for the simulation
r = np.random.randint(1, 99999)
simulation.context.setVelocitiesToTemperature(config["temperature"], r)

# Run molecular dynamics
simulation.step(int(config["numberOfSteps"]))
