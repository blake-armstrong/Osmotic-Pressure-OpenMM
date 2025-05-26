from dataclasses import dataclass, astuple
import numpy as np

import openmm.unit as unit

from .io import OsmoticConfig
from .geometry import GeomData, Plane, set_scale_mu
import logging

LOG = logging.getLogger(__name__)

class GCMD:
    PRESS_CF0 = 16.605388  # <- convert 'kJ/mole/nm^3' to 'bar'
    PRESS_CF1 = 10  # <- convert 'kJ/L' to 'bar'
    CONC_CF = 1.66053904  # <- convert 'atm/nm^3' to 'mol/L'

    @dataclass
    class Output:
        surface_pressure: float
        average_pressure: float
        gcmd_press: float
        scaled_parm: float
        area: float
        volume: float
        predicted_press: float

        def format(self):
            vals = []
            for val in astuple(self):
                if isinstance(val, unit.Quantity):
                    val = val._value
                vals.append(val)
            return vals

    def __init__(
        self, config: OsmoticConfig, num_particles: int, timestep: float
    ) -> None:
        self.config = config
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
        self.scale_mu = set_scale_mu(self.geometry)

    @staticmethod
    def berendsen(
        timestep: float,
        tau: float,
        external_press: float,
        gcmd_press: float,
        compressibility: float = 0.01,
    ):
        return 1.0 - compressibility * timestep / tau * (external_press - gcmd_press)

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
        gcmd_press = float(
            np.mean(
                self.osmotic_pressure[: min(self.counter, self.config.sample_length)]
            )
        )
        conc = 0
        if not isinstance(self.config.geometry, Plane):
            conc = self.num_particles / volume * GCMD.CONC_CF
        predicted_press = conc * self.kt * GCMD.PRESS_CF1
        mu = 1.0
        if self.config.gcmd:
            mu = self.scale_mu(self.berendsen(
                self.timestep,
                self.config.tau,
                self.config.osmotic_pressure,
                gcmd_press,
                self.config.compressibility,
            ))
            if isinstance(self.config.geometry, Plane) and self.config.geometry.direction == "+":
               mu = 2.0 - mu 
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
