"""Persistent SQLite storage layer for evolutionary memory records, evaluations, and lineage DAG."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evanthea.data.schemas import Mutation
from evanthea.memory.schema import MemoryRecord


class PersistentMemoryStore:
    """Thread-safe, transaction-isolated SQLite backing store for evolutionary candidate history."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_database()

    def _init_database(self) -> None:
        """Initializes tables, indices, and performance pragmas."""
        cur = self.conn.cursor()

        if self.db_path != ":memory:":
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS variants (
                sequence_hash TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                sequence TEXT NOT NULL,
                mutations_json TEXT NOT NULL,
                generation INTEGER NOT NULL,
                is_wildtype INTEGER NOT NULL,
                is_feasible INTEGER NOT NULL,
                feasibility_violations_json TEXT NOT NULL,
                objectives_vector_json TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT NOT NULL,
                objective_name TEXT NOT NULL,
                raw_score REAL NOT NULL,
                evaluated_at TEXT NOT NULL,
                UNIQUE(sequence_hash, objective_name),
                FOREIGN KEY (sequence_hash) REFERENCES variants(sequence_hash) ON DELETE CASCADE
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS lineage_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_hash TEXT NOT NULL,
                parent_hash TEXT NOT NULL,
                operator TEXT,
                generation INTEGER,
                FOREIGN KEY (child_hash) REFERENCES variants(sequence_hash) ON DELETE CASCADE
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS structural_evaluations (
                sequence_hash TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                mean_plddt REAL NOT NULL,
                plddt_per_residue_json TEXT NOT NULL,
                disordered_count INTEGER NOT NULL,
                rmsd_to_reference REAL,
                clash_count INTEGER NOT NULL,
                clashes_json TEXT NOT NULL,
                contact_preservation_ratio REAL,
                sasa_total REAL,
                sasa_hydrophobic REAL,
                is_structurally_viable INTEGER NOT NULL,
                violations_json TEXT NOT NULL,
                pdb_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
        """)

        # Performance indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_variants_created ON variants(created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_seq ON evaluations(sequence_hash);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage_edges(child_hash);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage_edges(parent_hash);")

        self.conn.commit()

    def has_sequence(self, sequence_hash: str) -> bool:
        """Checks if a sequence hash has already been registered in the database."""
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM variants WHERE sequence_hash = ? LIMIT 1;", (sequence_hash,))
        return cur.fetchone() is not None

    def save_record(self, record: MemoryRecord) -> None:
        """Saves or updates a MemoryRecord and its associated evaluation and lineage entries."""
        cur = self.conn.cursor()

        mutations_dicts = [m.model_dump() for m in record.mutations]
        obj_vec_json = json.dumps(record.objectives_vector) if record.objectives_vector is not None else None

        cur.execute("""
            INSERT INTO variants (
                sequence_hash, variant_id, sequence, mutations_json, generation,
                is_wildtype, is_feasible, feasibility_violations_json, objectives_vector_json,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sequence_hash) DO UPDATE SET
                variant_id = excluded.variant_id,
                generation = excluded.generation,
                is_feasible = excluded.is_feasible,
                feasibility_violations_json = excluded.feasibility_violations_json,
                objectives_vector_json = COALESCE(excluded.objectives_vector_json, variants.objectives_vector_json),
                metadata_json = excluded.metadata_json;
        """, (
            record.sequence_hash,
            record.variant_id,
            record.sequence,
            json.dumps(mutations_dicts),
            record.generation,
            1 if len(record.mutations) == 0 else 0,
            1 if record.is_feasible else 0,
            json.dumps(record.feasibility_violations),
            obj_vec_json,
            record.created_at,
            json.dumps(record.metadata),
        ))

        # Save evaluations
        for obj_name, score in record.fitness_values.items():
            cur.execute("""
                INSERT INTO evaluations (sequence_hash, objective_name, raw_score, evaluated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sequence_hash, objective_name) DO UPDATE SET
                    raw_score = excluded.raw_score,
                    evaluated_at = excluded.evaluated_at;
            """, (record.sequence_hash, obj_name, float(score), record.created_at))

        # Save lineage edges
        for p_id in record.parent_ids:
            cur.execute("""
                INSERT INTO lineage_edges (child_hash, parent_hash, operator, generation)
                VALUES (?, ?, ?, ?);
            """, (record.sequence_hash, p_id, record.metadata.get("operator"), record.generation))

        self.conn.commit()

    def get_record(self, sequence_hash: str) -> Optional[MemoryRecord]:
        """Retrieves complete MemoryRecord including evaluated scores and parents."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM variants WHERE sequence_hash = ? LIMIT 1;", (sequence_hash,))
        v_row = cur.fetchone()
        if not v_row:
            return None

        # Fetch evaluations
        cur.execute("SELECT objective_name, raw_score FROM evaluations WHERE sequence_hash = ?;", (sequence_hash,))
        fit_vals = {row["objective_name"]: float(row["raw_score"]) for row in cur.fetchall()}

        # Fetch parents
        cur.execute("SELECT parent_hash FROM lineage_edges WHERE child_hash = ?;", (sequence_hash,))
        parent_ids = [row["parent_hash"] for row in cur.fetchall()]

        # Parse mutations
        mut_dicts = json.loads(v_row["mutations_json"])
        mutations = [Mutation(**d) for d in mut_dicts]

        obj_vec = json.loads(v_row["objectives_vector_json"]) if v_row["objectives_vector_json"] else None
        violations = json.loads(v_row["feasibility_violations_json"])
        metadata = json.loads(v_row["metadata_json"])

        return MemoryRecord(
            sequence_hash=v_row["sequence_hash"],
            variant_id=v_row["variant_id"],
            sequence=v_row["sequence"],
            mutations=mutations,
            fitness_values=fit_vals,
            objectives_vector=obj_vec,
            is_feasible=bool(v_row["is_feasible"]),
            feasibility_violations=violations,
            generation=v_row["generation"],
            parent_ids=parent_ids,
            created_at=v_row["created_at"],
            metadata=metadata,
        )

    def get_all_records(self) -> List[MemoryRecord]:
        """Retrieves all registered memory records."""
        cur = self.conn.cursor()
        cur.execute("SELECT sequence_hash FROM variants ORDER BY created_at ASC;")
        hashes = [row["sequence_hash"] for row in cur.fetchall()]
        records = []
        for h in hashes:
            rec = self.get_record(h)
            if rec:
                records.append(rec)
        return records

    def count_variants(self) -> int:
        """Returns the total number of distinct sequence variants recorded."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM variants;")
        return int(cur.fetchone()[0])

    def __len__(self) -> int:
        """Enables len(store) to return total distinct sequence variants recorded."""
        return self.count_variants()

    def get_parents(self, child_hash: str) -> List[str]:
        """Returns parent identifiers for a given candidate."""
        cur = self.conn.cursor()
        cur.execute("SELECT parent_hash FROM lineage_edges WHERE child_hash = ?;", (child_hash,))
        return [row["parent_hash"] for row in cur.fetchall()]

    def get_children(self, parent_hash: str) -> List[str]:
        """Returns direct children identifiers for a given parent."""
        cur = self.conn.cursor()
        cur.execute("SELECT child_hash FROM lineage_edges WHERE parent_hash = ?;", (parent_hash,))
        return [row["child_hash"] for row in cur.fetchall()]

    def save_structural_evaluation(self, assessment_dict: Dict[str, Any], pdb_text: str) -> None:
        """Stores a structural assessment and its full 3D PDB text."""
        cur = self.conn.cursor()
        created_at = datetime.now(timezone.utc).isoformat()
        cur.execute("""
            INSERT OR REPLACE INTO structural_evaluations (
                sequence_hash, variant_id, mean_plddt, plddt_per_residue_json,
                disordered_count, rmsd_to_reference, clash_count, clashes_json,
                contact_preservation_ratio, sasa_total, sasa_hydrophobic,
                is_structurally_viable, violations_json, pdb_text, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            assessment_dict["sequence_hash"],
            assessment_dict["variant_id"],
            float(assessment_dict["mean_plddt"]),
            json.dumps(assessment_dict.get("plddt_per_residue", [])),
            int(assessment_dict.get("disordered_residues_count", 0)),
            float(assessment_dict["rmsd_to_reference"]) if assessment_dict.get("rmsd_to_reference") is not None else None,
            int(assessment_dict.get("clash_count", 0)),
            json.dumps(assessment_dict.get("clashes", [])),
            float(assessment_dict["contact_preservation_ratio"]) if assessment_dict.get("contact_preservation_ratio") is not None else None,
            float(assessment_dict["sasa_total"]) if assessment_dict.get("sasa_total") is not None else None,
            float(assessment_dict["sasa_hydrophobic"]) if assessment_dict.get("sasa_hydrophobic") is not None else None,
            int(assessment_dict.get("is_structurally_viable", True)),
            json.dumps(assessment_dict.get("violations", [])),
            pdb_text,
            created_at,
            json.dumps(assessment_dict.get("metadata", {})),
        ))
        self.conn.commit()

    def get_structural_evaluation(self, sequence_hash: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Retrieves cached structural assessment dictionary and PDB text for a sequence hash."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM structural_evaluations WHERE sequence_hash = ?;", (sequence_hash,))
        row = cur.fetchone()
        if not row:
            return None

        assessment_dict = {
            "sequence_hash": row["sequence_hash"],
            "variant_id": row["variant_id"],
            "mean_plddt": row["mean_plddt"],
            "plddt_per_residue": json.loads(row["plddt_per_residue_json"]),
            "disordered_residues_count": row["disordered_count"],
            "rmsd_to_reference": row["rmsd_to_reference"],
            "clash_count": row["clash_count"],
            "clashes": json.loads(row["clashes_json"]),
            "contact_preservation_ratio": row["contact_preservation_ratio"],
            "sasa_total": row["sasa_total"],
            "sasa_hydrophobic": row["sasa_hydrophobic"],
            "is_structurally_viable": bool(row["is_structurally_viable"]),
            "violations": json.loads(row["violations_json"]),
            "metadata": json.loads(row["metadata_json"]),
        }
        return assessment_dict, row["pdb_text"]

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        self.conn.close()

    def __enter__(self) -> "PersistentMemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
