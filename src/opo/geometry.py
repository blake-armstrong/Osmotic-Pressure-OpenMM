from dataclasses import dataclass
from typing import List, Callable
import logging
from abc import abstractmethod

import numpy as np

import openmm as mm
import openmm.app as app

D0 = "d0"
LOG = logging.getLogger(__name__)

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
class GeomForceInfo:
    expression: str
    d0: float
    global_parms: dict

    def __post_init__(self):
        LOG.debug("Info used to generate osmotic force object: ")
        LOG.debug("  Expression: %s", self.expression)
        LOG.debug("  %s: %s", D0, self.d0)
        LOG.debug("  Global paramaters: %s", self.global_parms)


@dataclass
class Geometry:
    kappa: float
    species: str | List[str]
    # axis: str

    def __post_init__(self):
        if not isinstance(self.species, list):
            self.species = [self.species,]

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

    def __post_init__(self):
        super().__post_init__()
        self.axis_obj = Axis(self.axis)
        self.dirs = self.axis_obj.directions()


    def get_area(self, geom_data: GeomData) -> float:
        i, j, _ = self.dirs
        return float(geom_data.box[i][i] * geom_data.box[j][j])

    def get_volume(self, geom_data: GeomData) -> float:
        _ = geom_data
        return 0.0


@dataclass
class Slab(Geometry):
    width: float
    centre: float
    axis: str

    def __post_init__(self):
        super().__post_init__()
        self.axis_obj = Axis(self.axis)
        self.dirs = self.axis_obj.directions()

    def get_area(self, geom_data: GeomData) -> float:
        i, j, _ = self.dirs
        return float(geom_data.box[i][i] * geom_data.box[j][j] * 2)

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
        self.axis_obj = Axis(self.axis)
        self.dirs = self.axis_obj.directions()
        if len(self.centre) != 2:
            raise ValueError("centre arg for Cylinder should contain two elements.")

    def get_area(self, geom_data: GeomData) -> float:
        _, _, k = self.dirs
        return float(
            geom_data.box[k][k] * 2 * np.pi * (geom_data.radius + geom_data.factor1)
        )

    def get_volume(self, geom_data: GeomData) -> float:
        _, _, k = self.dirs
        return float(
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

    def __post_init__(self):
        super().__post_init__()
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


GeometryType = Plane | Slab | Cylinder | Sphere


class GeometryInterpreter:
    def __init__(self, geometry: GeometryType, topology: app.Topology) -> None:
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

    def _process_plane(self, plane: Plane) -> GeomForceInfo:
        expression = f"0.5*k*(max(0,{plane.axis_obj.axis}-{D0}))^2)"
        d0 = plane.pos
        global_parms = dict({})
        return GeomForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_slab(self, slab: Slab) -> GeomForceInfo:
        axis = slab.axis_obj.axis.lower()
        centre = f"{axis}0"
        expression = f"0.5*k*(max(0,d-{D0}))^2; " f"d=sqrt(({axis}-{centre})^2)"
        d0 = slab.width
        global_parms = dict({centre: slab.centre})
        return GeomForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_cylinder(self, cylinder: Cylinder) -> GeomForceInfo:
        try:
            a0, a1 = cylinder.axis_obj.orthog()
            a0 = a0.lower()
            a1 = a1.lower()
        except Exception as e:
            raise RuntimeError(
                f"Could not get plane orthogonal to axis '{cylinder.axis_obj.axis}'"
            ) from e
        g0, g1 = f"{a0}0", f"{a1}0"
        expression = f"0.5*k*(max(0,d-{D0}))^2; " f"d=sqrt(({a0}-{g0})^2+({a1}-{g1})^2)"
        centre = cylinder.centre
        global_parms = dict({g0: centre[0], g1: centre[1]})
        d0 = cylinder.radius
        return GeomForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_sphere(self, sphere: Sphere) -> GeomForceInfo:
        x0, y0, z0 = "x0", "y0", "z0"
        expression = (
            f"0.5*k*(max(0,d-{D0}))^2; " f"d=periodicdistance(x,y,z,{x0},{y0},{z0})"
        )
        centre = sphere.centre
        global_parms = dict({x0: centre[0], y0: centre[1], z0: centre[2]})
        d0 = sphere.radius
        return GeomForceInfo(expression=expression, d0=d0, global_parms=global_parms)

    def _process_geometry(self) -> mm.CustomExternalForce:
        osm_info = self.process(self.geometry)
        osm_force = mm.CustomExternalForce(osm_info.expression)
        osm_force.addGlobalParameter("k", self.geometry.kappa)
        osm_force.addGlobalParameter(f"{D0}", osm_info.d0)
        # osm_force.addPerParticleParameter(f"{D0}")
        for p_name, p_val in osm_info.global_parms.items():
            osm_force.addGlobalParameter(p_name, p_val)
        for atom_index in self.atoms:
            # osm_force.addParticle(atom_index, [osm_info.d0])
            osm_force.addParticle(atom_index, [])
        try:
            osm_force.setName(str(self.geometry))
            LOG.debug("Osmotic membrane created as: '%s'", str(self.geometry))
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

def set_scale_mu(geometry: GeometryType) -> Callable:
    scalings = {
        Plane: lambda mu: mu,
        Slab: lambda mu: mu,
        Cylinder: lambda mu: mu**(2/3),
        Sphere: lambda mu: mu**(1/3),
    }
    try:
        return scalings[type(geometry)]
    except Exception as e:
        raise ValueError(f"Could not determine mu scaling of geometry type: '{type(geometry)}'") from e
