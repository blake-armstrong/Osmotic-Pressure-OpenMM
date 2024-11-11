### Copyright (C) 2024  Blake I. Armstrong, Paolo Raiteri

from abc import abstractmethod
import os
import sys
import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

import logging
import parse
from dataclasses import dataclass, astuple
from collections import OrderedDict
from typing import Optional, List, Set, Tuple, Any

LOG = logging.getLogger(__name__)


class File:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, "w", encoding="utf-8")
        LOG.debug("Sucessfully writing to file: '%s'", self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.file:
            return
        LOG.debug("Finished writing to file: '%s'", self.filename)
        self.file.close()
        LOG.debug("Closing file: '%s'", self.filename)

    def write_line(self, text, flush=True):
        if not self.file:
            return
        self.file.write(text + "\n")
        if flush:
            self.file.flush()

    def get_last_line(self) -> Optional[str]:
        if not self.file:
            return None
        lines = self.file.readlines()
        if not lines:
            raise RuntimeError(f"Could not read from file: '{self.filename}")
        return lines[-1]


class Axis:
    def __init__(self, axis: str):
        self.axis = str(axis).upper()
        self.axes = ["X", "Y", "Z"]
        if self.axis not in self.axes:
            raise ValueError(
                f"Input axis '{self.axis}' not an available axis: {*self.axes,}"
            )

    def orthog(self) -> list:
        return [axis for axis in self.axes if axis != self.axis]

    def directions(self) -> tuple:
        dirs = {"X": (2, 1, 0), "Y": (0, 2, 1), "Z": (0, 1, 2)}
        return dirs[self.axis]


@dataclass
class GeomData:
    box: np.ndarray
    radius: float
    factor0: float
    factor1: float


@dataclass
class Geometry:
    kappa: float
    species: str | List[str]
    axis: str
    # _axis: Optional[Axis]

    def __post_init__(self):
        self.axis_obj = Axis(self.axis)
        self.dirs = self.axis_obj.directions()
        if not isinstance(self.species, list):
            self.species = [self.species]

    @abstractmethod
    def get_area(self, geom_data: GeomData) -> float:
        pass

    @abstractmethod
    def get_volume(self, geom_data: GeomData) -> float:
        pass


@dataclass
class Plane(Geometry):
    pos: float
    axis: str

    def get_area(self, geom_data: GeomData) -> float:
        i, j, _ = self.dirs
        return geom_data.box[i][i] * geom_data.box[j][j]

    def get_volume(self, geom_data: GeomData) -> float:
        _ = geom_data
        return 0.0


@dataclass
class Slab(Geometry):
    width: float
    centre: float
    axis: str

    def get_area(self, geom_data: GeomData) -> float:
        i, j, _ = self.dirs
        return geom_data.box[i][i] * geom_data.box[j][j] * 2

    def get_volume(self, geom_data: GeomData) -> float:
        return self.get_area(geom_data) * (geom_data.radius + geom_data.factor0)


@dataclass
class Cylinder(Geometry):
    """
    Ordering of elements in centre is important. The order corresponds to the alphabetical
    ordering of x, y and z. For instance if axis is "y" then centre[0] will be for x and
    centre[1] will be for z.
    """

    radius: float
    centre: List[float]
    axis: str

    def __post_init__(self):
        super().__post_init__()
        if len(self.centre) != 2:
            raise ValueError("centre arg for Cylinder should contain two elements.")

    def get_area(self, geom_data: GeomData) -> float:
        _, _, k = self.dirs
        return geom_data.box[k][k] * 2 * np.pi * (geom_data.radius + geom_data.factor1)

    def get_volume(self, geom_data: GeomData) -> float:
        _, _, k = self.dirs
        return (
            geom_data.box[k][k]
            * 2
            * np.pi
            * (
                0.5 * geom_data.radius**2
                + geom_data.radius * geom_data.factor1
                + geom_data.factor0
            )
        )


@dataclass
class Sphere(Geometry):
    """
    Centre order is x, y and z.
    """

    radius: float
    centre: List[float]
    axis: str = "None"

    def __post_init__(self):
        if len(self.centre) != 3:
            raise ValueError("centre arg for Cylinder should contain three elements.")

    def get_area(self, geom_data: GeomData) -> float:
        return (
            4
            * np.pi
            * (
                geom_data.radius**2
                + 2 * geom_data.radius * geom_data.factor1
                + 2 * geom_data.factor0
            )
        )

    def get_volume(self, geom_data: GeomData) -> float:
        return (
            4
            * np.pi
            * (
                geom_data.radius**3 / 3
                + (geom_data.radius**2 + geom_data.factor0) * geom_data.factor1
                + geom_data.radius * 2 * geom_data.factor0
            )
        )


@dataclass
class OsmForceInfo:
    expression: str
    d0: float
    global_parms: dict


GeometryType = Plane | Slab | Cylinder | Sphere


@dataclass
class OsmoticConfig:
    temperature: float
    geometry: GeometryType
    osmotic_pressure: float
    tau: float = 1.0
    file: str = "osmotic.out"
    restart: Optional[str] = None
    compressibility: float = 0.01
    compute_interval: int = 1000
    report_interval: int = 1000
    sample_length: int = 1000
    gcmd: bool = True

    def __post_init__(self):
        if self.report_interval >= self.compute_interval:
            raise ValueError(
                f"Report interval ({self.report_interval}) should be equal to"
                " or larger than compute interval ({self.compute_interval})"
            )
        if self.report_interval % self.compute_interval == 0:
            raise ValueError(
                "Report interval should be a multiple of compute interval."
            )
        LOG.info("Osmotic config has the following options:")
        LOG.info("  temperature: %s", str(self.temperature))
        LOG.info("  geometry: %s", str(self.geometry))
        LOG.info("  osmotic pressure: %s", str(self.osmotic_pressure))
        LOG.info("  tau: %s", str(self.tau))
        LOG.info("  compressibility: %s", str(self.compressibility))
        LOG.info("  restart: %s", str(self.restart))
        LOG.info("  output file: %s", self.file)
        LOG.info("  compute interval: %s", str(self.compute_interval))
        LOG.info("  report interval: %s", str(self.report_interval))
        LOG.info("  sample length: %s", str(self.sample_length))
        LOG.info("  gcmd: %s", str(self.gcmd))


class GeometryInterpreter:
    def __init__(
        self, geometry: Plane | Slab | Cylinder | Sphere, topology: app.Topology
    ) -> None:
        self.geometry = geometry
        self.atoms = self._get_atoms(topology)
        self.processors = {
            Plane: self._process_plane,
            Slab: self._process_slab,
            Cylinder: self._process_cylinder,
            Sphere: self._process_sphere,
        }
        if not isinstance(self.geometry, tuple(self.processors.keys())):
            raise ValueError(f"Unsupported geometry type: {type(self.geometry)}")

        self.process = self.processors[type(geometry)]

    def _get_atoms(self, topology: app.Topology) -> list:
        atoms = list(topology.atoms())
        atom_index_list = []
        for atom in atoms:
            if atom.name not in self.geometry.species:
                continue
            atom_index_list.append(atom.index)
        if not atom_index_list:
            raise RuntimeError(
                f"Could not find atoms with name(s): {*self.geometry.species,}"
            )
        return atom_index_list

    def _process_plane(self, plane: Plane) -> OsmForceInfo:
        expression = f"0.5*k*(max(0,{plane.axis_obj.axis}-{GCMD.D0}))^2)"
        d0 = plane.pos
        global_parms = dict({})
        return OsmForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_slab(self, slab: Slab) -> OsmForceInfo:
        axis = slab.axis_obj.axis.lower()
        centre = f"{axis}0"
        expression = f"0.5*k*(max(0,d-{GCMD.D0}))^2; " f"d=sqrt(({axis}-{centre})^2)"
        d0 = slab.width
        global_parms = dict({centre: slab.centre})
        return OsmForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_cylinder(self, cylinder: Cylinder) -> OsmForceInfo:
        try:
            a0, a1 = cylinder.axis_obj.orthog()
        except Exception as e:
            raise RuntimeError(
                f"Could not get plane orthogonal to axis '{cylinder.axis_obj.axis}'"
            ) from e
        g0, g1 = f"{a0}0", f"{a1}0"
        expression = (
            f"0.5*k*(max(0,d-{GCMD.D0}))^2; " f"d=sqrt(({a0}-{g0})^2+({a1}-{g1})^2)"
        )
        centre = cylinder.centre
        global_parms = dict({g0: centre[0], g1: centre[1]})
        d0 = cylinder.radius
        return OsmForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_sphere(self, sphere: Sphere) -> OsmForceInfo:
        x0, y0, z0 = "x0", "y0", "z0"
        expression = (
            f"0.5*k*(max(0,d-{GCMD.D0}))^2; "
            f"d=periodicdistance(x,y,z,{x0},{y0},{z0})"
        )
        centre = sphere.centre
        global_parms = dict({x0: centre[0], y0: centre[1], z0: centre[2]})
        d0 = sphere.radius
        return OsmForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_geometry(self) -> mm.CustomExternalForce:
        osm_info = self.process(self.geometry)
        osm_force = mm.CustomExternalForce(osm_info.expression)
        osm_force.addGlobalParameter("k", self.geometry.kappa)
        osm_force.addPerParticleParameter(f"{GCMD.D0}")
        for p_name, p_val in osm_info.global_parms.items():
            osm_force.addGlobalParameter(p_name, p_val)
        for atom_index in self.atoms:
            osm_force.addParticle(atom_index, [osm_info.d0])
        try:
            osm_force.setName(str(self.geometry))
            LOG.info("Osmotic membrane created as: %s", str(self.geometry))
        except Exception as e:
            raise RuntimeError(
                "Could not set force name for osmotic pressure force. Perhaps a duplicate?"
            ) from e
        return osm_force

    def process_geometry(self) -> mm.CustomExternalForce:
        try:
            return self._process_geometry()
        except Exception as e:
            raise RuntimeError("Could not create osmotic pressure force object.") from e


class GCMD:
    PRESS_CF0 = 16.605388  # <- convert 'kJ/mole/nm^3' to 'bar'
    PRESS_CF1 = 10  # <- convert 'kJ/L' to 'bar'
    CONC_CF = 0.16606  # <- convert 'atm/nm^3' to 'mol/L'
    D0 = "d0"

    @dataclass
    class Output:
        surface_pressure: float
        average_pressure: float
        gcmd_press: float
        scaled_parm: float
        area: float
        volume: float
        predicted_press: float

    def __init__(
        self, config: OsmoticConfig, num_particles: int, timestep: float
    ) -> None:
        self.config = config
        # self.file = file
        self.num_particles = num_particles
        self.timestep = timestep
        self.geometry = self.config.geometry
        self.temperature = self.config.temperature
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).in_units_of(
            unit.kilojoules_per_mole
        )
        self.factor0 = self.kt / (self.geometry.kappa * unit.kilojoules_per_mole)
        self.factor1 = np.sqrt(np.pi * self.factor0 / 2)
        self.counter: int = 1
        self.average_pressure: float = 0.0
        self.osmotic_pressure: np.ndarray = np.zeros(self.config.sample_length)
        if self.config.restart:
            self.restart()
        # self.file.write_line(GCMD.HEADER)

    @staticmethod
    def berendsen(
        timestep: float,
        tau: float,
        external_press: float,
        gcmd_press: float,
        compressibility: float = 0.01,
    ):
        return 1.0 - compressibility * timestep / tau * (external_press - gcmd_press)

    def restart(self) -> None:
        restart_file = File(self.config.restart)
        last_line = restart_file.get_last_line()
        parsed = parse.parse(OsmoticPressureReporter.FMT, last_line)
        if not parsed:
            raise ValueError(
                f"Could not parse last line of restart file '{restart_file.filename}'"
            )
        # parsed = parsed.fixed
        # TODO: test!!!

        return

    def report(self, forces: np.ndarray, box: np.ndarray, radius: float):
        f = np.sum(np.sqrt(np.einsum("ij,ij->i", forces, forces)))
        geom_data = GeomData(
            box=box, radius=radius, factor0=self.factor0, factor1=self.factor1
        )
        area = self.config.geometry.get_area(geom_data)
        volume = self.config.geometry.get_volume(geom_data)
        surface_pressure = f / area * GCMD.PRESS_CF0
        self.osmotic_pressure[(self.counter - 1) % self.config.sample_length] = (
            surface_pressure
        )
        # Global average of the pressure
        self.average_pressure = (
            self.average_pressure * (self.counter - 1) + surface_pressure
        ) / self.counter
        # Window average of the pressure
        gcmd_press = np.mean(
            self.osmotic_pressure[: min(self.counter, self.config.sample_length)]
        )
        conc = 0
        if not isinstance(self.config.geometry, Plane):
            conc = self.num_particles / volume * GCMD.CONC_CF
        predicted_press = conc * self.kt * GCMD.PRESS_CF1
        mu = 1.0
        if self.config.gcmd:
            mu = self.berendsen(
                self.timestep,
                self.config.tau,
                self.config.osmotic_pressure,
                gcmd_press,
                self.config.compressibility,
            )
        self.counter += 1

        return GCMD.Output(
            surface_pressure=surface_pressure,
            average_pressure=self.average_pressure,
            gcmd_press=gcmd_press,
            scaled_parm=mu * radius,
            area=area,
            volume=volume,
            predicted_press=predicted_press,
        )


class OsmoticPressureReporter:
    HEADER = "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Time", "Pi(t)", "<Pi>", "Pi(gcmd)", "Parm", "Area", "Volume", "Press"
    )
    FMT = "{:12.1f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}"

    def __init__(
        self, context: mm.Context, topology: app.Topology, config: OsmoticConfig
    ) -> None:
        LOG.info("Initialising osmotic pressure calculation")
        self.config = config
        self.context = context
        self.topology = topology
        osmotic_force = self._generate_osmotic_force()
        num_particles = self.get_num_particles(osmotic_force)
        system = self.get_system(self.context)
        system.addForce(osmotic_force)
        self.context.reinitialize(preserveState=True)
        self.force_group = self.get_force_group(osmotic_force, system)
        self.file = File(self.config.file)
        timestep = (
            self.context.getIntegrator().getStepSize().value_in_unit(unit.picosecond)  # type: ignore
        )
        self.gcmd = GCMD(self.config, num_particles, timestep)
        self.file.write_line(OsmoticPressureReporter.HEADER)

    @staticmethod
    def get_system(context: mm.Context) -> mm.System:
        try:
            return context.getSystem()
        except Exception as e:
            raise RuntimeError("Could not extract System from supplied Context") from e

    @staticmethod
    def get_num_particles(force: mm.CustomExternalForce) -> int:
        try:
            num_particles = force.getNumParticles()
        except Exception as e:
            raise RuntimeError("Could not get num particles in osmotic force") from e
        if num_particles <= 0:
            raise ValueError("No particles for osmotic force")
        return num_particles

    def _generate_osmotic_force(self) -> mm.CustomExternalForce:
        try:
            geometry_interpreter = GeometryInterpreter(
                self.config.geometry, self.topology
            )
            return geometry_interpreter.process_geometry()
        except Exception as e:
            raise RuntimeError("Could not process input geometry.") from e

    def get_force_group(
        self, osmotic_force: mm.CustomExternalForce, system: mm.System
    ) -> Set[int]:
        # force_name = osmotic_force.getName()
        # force_group = set()
        # for force in system.getForces():
        #     if force.getName() == force_name:
        #         force_group.add(force.getForceGroup())
        #         break
        force_group = set([int(osmotic_force.getForceGroup())])

        if not force_group:
            raise ValueError("Found no force for osmotic calculation.")

        return force_group

    def describe_next_report(self, simulation: app.Simulation):
        steps = (
            self.config.compute_interval
            - simulation.currentStep % self.config.compute_interval
        )
        return (steps, False, False, True, False)

    def _report(self, simulation: app.Simulation, state: mm.State) -> None:
        stime = state.getTime().value_in_unit(unit.picosecond)  # type: ignore
        forces = simulation.context.getState(
            getForces=True, groups=self.force_group  # type: ignore
        ).getForces(asNumpy=True)
        forces = forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)  # type: ignore
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore
        radius = self.context.getParameter(GCMD.D0)
        output = self.gcmd.report(forces, box, radius)
        if simulation.currentStep % self.config.report_interval != 0:
            return
        self.file.write_line(
            OsmoticPressureReporter.FMT.format(stime, *astuple(output))
        )
        if not self.config.gcmd:
            return
        self.context.setParameter(GCMD.D0, output.scaled_parm)

    def report(self, *args):
        """
        Method for OpenMM
        """
        return self._report(*args)

    def describeNextReport(self, *args):
        """
        Method for OpenMM
        """
        return self.describe_next_report(*args)
