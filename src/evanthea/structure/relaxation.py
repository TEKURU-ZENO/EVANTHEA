"""Optional energy minimization and structural relaxation interface (OpenMM with graceful fallback)."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
from Bio.PDB import PDBParser, Structure


class StructureRelaxer:
    """Relieves minor steric clashes via localized energy minimization when OpenMM is present."""

    def __init__(self, max_iterations: int = 500, tolerance_kj_mol: float = 10.0) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance_kj_mol
        self._openmm_available = False

        try:
            import openmm
            import openmm.app as app
            self._openmm = openmm
            self._app = app
            self._openmm_available = True
        except ImportError:
            self._openmm = None
            self._app = None
            self._openmm_available = False

    @property
    def is_openmm_available(self) -> bool:
        """Returns True if OpenMM molecular dynamics toolkit is available."""
        return self._openmm_available

    def relax_pdb_text(
        self,
        pdb_text: str,
        structure_id: str = "RELAXED",
    ) -> Tuple[str, bool]:
        """Performs energy minimization on a PDB structure text.

        Returns:
            Tuple of (relaxed_pdb_text, was_relaxed_boolean).
        """
        if not self._openmm_available:
            # Transparent fallback: return original structure unrelaxed
            return pdb_text, False

        try:
            pdb_file = io.StringIO(pdb_text)
            pdb = self._app.PDBFile(pdb_file)
            forcefield = self._app.ForceField("amber14-all.xml", "implicit/obc2.xml")

            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=self._app.NoCutoff,
                constraints=self._app.HBonds,
            )
            integrator = self._openmm.LangevinMiddleIntegrator(
                300.0 * self._openmm.unit.kelvin,
                1.0 / self._openmm.unit.picosecond,
                0.002 * self._openmm.unit.picoseconds,
            )
            simulation = self._app.Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)

            # Energy minimize to relieve steric clashes
            simulation.minimizeEnergy(
                tolerance=self.tolerance * self._openmm.unit.kilojoule / self._openmm.unit.mole,
                maxIterations=self.max_iterations,
            )

            positions = simulation.context.getState(getPositions=True).getPositions()
            output_stream = io.StringIO()
            self._app.PDBFile.writeFile(pdb.topology, positions, output_stream)
            relaxed_text = output_stream.getvalue()
            return relaxed_text, True
        except Exception:
            # If relaxation encounters an issue, return original structure safely
            return pdb_text, False
