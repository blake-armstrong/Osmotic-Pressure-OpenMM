### Copyright (C) 2024  Blake I. Armstrong, Paolo Raiteri

import logging
from dataclasses import dataclass
from typing import Optional, Set

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parse

from .geometry import GeometryInterpreter, D0
from .gcmd import GCMD
from .io import File, OsmoticConfig

LOG = logging.getLogger(__name__)


class OsmoticPressureReporter:
    HEADER = "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Time", "Pi(t)", "<Pi>", "Pi(gcmd)", "Parm", "Area", "Volume", "Press"
    )
    FMT = "{:12.1f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}"

    def __init__(
        self, context: mm.Context, topology: app.Topology, config: OsmoticConfig
    ) -> None:
        LOG.debug("Initialising osmotic pressure calculation")
        self.config = config
        self.context = context
        self.topology = topology
        system = self.get_system(self.context)
        osmotic_force = self._generate_osmotic_force()
        num_particles = self.get_num_particles(osmotic_force)
        system.addForce(osmotic_force)
        self.context.reinitialize(preserveState=True)
        self.force_group = self.get_force_group(osmotic_force, system)
        self.file = File(self.config.file)
        timestep = (
            self.context.getIntegrator().getStepSize().value_in_unit(unit.picosecond)  # type: ignore
        )
        self.gcmd = GCMD(self.config, num_particles, timestep)
        if self.config.restart:
            self.restart()
        with self.file as f:
            f.write_line(OsmoticPressureReporter.HEADER)
        LOG.debug("Successfully initialised osmotic pressure report.")

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
        force_name = osmotic_force.getName()
        force_group = set()
        for force in system.getForces():
            if force.getName() == force_name:
                force_group.add(force.getForceGroup())
        # break
        # force_group = set([int(osmotic_force.getForceGroup())])

        if not force_group:
            raise ValueError("Found no force for osmotic calculation.")

        LOG.debug("Osmotic force group: %s", force_group)

        return force_group

    def restart(self) -> None:
        restart_file = File(self.config.restart, method="r")
        with restart_file as f:
            last_line = f.get_last_line()
        LOG.debug("Last line: '%s'", last_line)
        parsed = parse.parse(
            f"{OsmoticPressureReporter.FMT}\n", last_line, evaluate_result=True
        )
        if not isinstance(parsed, parse.Result):
            raise ValueError(
                f"Could not parse last line of restart file '{restart_file.filename}'"
            )
        parsed = parsed.fixed
        LOG.debug("Parsed: '%s'", parsed)
        try:
            output = GCMD.Output(*parsed[1:])
        except Exception as e:
            raise RuntimeError("Could not parse restart file.") from e

        self.context.setParameter(D0, output.scaled_parm)
        LOG.debug("Successfully restarted from file: '%s", restart_file.filename)

        # TODO: test!!!

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
        radius = self.context.getParameter(D0)
        output = self.gcmd.report(forces, box, radius)
        if simulation.currentStep % self.config.report_interval != 0:
            return
        with self.file as f:
            f.write_line(OsmoticPressureReporter.FMT.format(stime, *output.format()))
        if not self.config.gcmd:
            return
        self.context.setParameter(D0, output.scaled_parm)

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
