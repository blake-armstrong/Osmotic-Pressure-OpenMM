import logging
from dataclasses import dataclass
from typing import Optional

from .geometry import GeometryType

LOG = logging.getLogger(__name__)


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
        if self.report_interval < self.compute_interval:
            raise ValueError(
                f"Report interval ({self.report_interval}) should be equal to"
                f" or larger than compute interval ({self.compute_interval})"
            )
        if self.report_interval % self.compute_interval != 0:
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


class File:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.method = "w"

    def __call__(self, method: str):
        self.method = method
        return self

    def __enter__(self):
        self.file = open(self.filename, self.method, encoding="utf-8")
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
