"""Data ingestion pipelines for ProteinGym deep mutational scanning assays and FireProtDB stability records."""

import csv
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evanthea.constants import DDG_SIGN_STABILIZING_NEGATIVE
from evanthea.data.schemas import Mutation, ProteinRecord, VariantRecord
from evanthea.data.validation import ValidationIssue, ValidationSummary, validate_variant_against_reference
from evanthea.protein.sequence import apply_mutations, parse_mutation_string, validate_amino_acid_string


def compute_robust_z_scores(scores: List[float]) -> List[float]:
    """Normalizes scores using robust median / interquartile range (IQR).

    formula: (score - median) / IQR
    If IQR == 0 (constant scores), returns 0.0 for all.
    """
    if not scores:
        return []

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def _median(arr: List[float]) -> float:
        m = len(arr)
        if m == 0:
            return 0.0
        if m % 2 == 1:
            return arr[m // 2]
        return (arr[m // 2 - 1] + arr[m // 2]) / 2.0

    med = _median(sorted_scores)
    q1 = _median(sorted_scores[: n // 2])
    q3 = _median(sorted_scores[(n + 1) // 2 :]) if n % 2 == 1 else _median(sorted_scores[n // 2 :])
    iqr = q3 - q1

    if abs(iqr) < 1e-9:
        return [0.0] * n

    return [(s - med) / iqr for s in scores]


class ProteinGymIngestor:
    """Ingests and validates ProteinGym CSV assay files against a canonical ProteinRecord."""

    def __init__(self, protein: ProteinRecord):
        self.protein = protein

    def ingest_csv(
        self,
        csv_path_or_buffer: Path | str | io.StringIO,
        mutant_col: str = "mutant",
        sequence_col: str = "mutated_sequence",
        score_col: str = "DMS_score",
        max_rows: Optional[int] = None,
    ) -> Tuple[List[VariantRecord], ValidationSummary]:
        """Parses a ProteinGym CSV file, validates variants, and computes robust normalized scores."""
        records: List[Dict[str, Any]] = []

        if isinstance(csv_path_or_buffer, (Path, str)):
            with open(csv_path_or_buffer, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if max_rows and idx >= max_rows:
                        break
                    records.append(row)
        else:
            reader = csv.DictReader(csv_path_or_buffer)
            for idx, row in enumerate(reader):
                if max_rows and idx >= max_rows:
                    break
                records.append(row)

        valid_variants: List[VariantRecord] = []
        issues: List[ValidationIssue] = []
        raw_scores: List[float] = []

        wt_count = 0
        single_count = 0
        multi_count = 0

        for idx, row in enumerate(records):
            mut_str = row.get(mutant_col, "").strip()
            seq_str = row.get(sequence_col, "").strip()
            score_raw = row.get(score_col)

            parsed_score: Optional[float] = None
            if score_raw is not None and score_raw != "":
                try:
                    val = float(score_raw)
                    if not math.isnan(val):
                        parsed_score = val
                except ValueError:
                    pass

            vid = f"{self.protein.id}_{idx}_{mut_str if mut_str else 'WT'}"

            # If mutated sequence is missing from CSV, reconstruct from wild-type + mutations
            if not seq_str and mut_str:
                try:
                    muts = parse_mutation_string(mut_str)
                    seq_str = apply_mutations(self.protein.wildtype_sequence, muts)
                except Exception as e:
                    issues.append(
                        ValidationIssue(
                            row_index=idx,
                            variant_id=vid,
                            error_type="ReconstructionFailed",
                            message=str(e),
                            mutation_string=mut_str,
                        )
                    )
                    continue

            var_record, issue = validate_variant_against_reference(
                row_idx=idx,
                variant_id=vid,
                mutated_seq_raw=seq_str,
                mutation_str_raw=mut_str,
                reference_protein=self.protein,
            )

            if issue:
                issues.append(issue)
            elif var_record:
                var_record.raw_score = parsed_score
                var_record.score_metric_name = score_col
                valid_variants.append(var_record)
                if parsed_score is not None:
                    raw_scores.append(parsed_score)

                if var_record.is_wildtype:
                    wt_count += 1
                elif var_record.num_mutations == 1:
                    single_count += 1
                else:
                    multi_count += 1

        # Compute robust normalized scores across valid records with non-null scores
        if len(raw_scores) == len(valid_variants) and raw_scores:
            z_scores = compute_robust_z_scores(raw_scores)
            for var, z in zip(valid_variants, z_scores):
                var.normalized_score = z

        summary = ValidationSummary(
            total_records=len(records),
            valid_records=len(valid_variants),
            invalid_records=len(issues),
            wildtype_records=wt_count,
            single_mutants=single_count,
            multi_mutants=multi_count,
            issues=issues,
        )

        return valid_variants, summary


class FireProtDBIngestor:
    """Ingests and validates experimental protein stability records from FireProtDB / ProTherm."""

    @staticmethod
    def parse_stability_record(
        row: Dict[str, Any],
        reference_protein: ProteinRecord,
        sign_convention: str = DDG_SIGN_STABILIZING_NEGATIVE,
    ) -> Tuple[Optional[VariantRecord], Optional[ValidationIssue]]:
        """Parses a single row from FireProtDB with explicit thermodynamic sign convention."""
        # Typically columns: mutation, dTm (C), ddG (kcal/mol), pH, temp, method
        mut_str = row.get("mutation", "").strip()
        variant_id = f"{reference_protein.id}_{mut_str}"

        try:
            muts = parse_mutation_string(mut_str)
            mutated_seq = apply_mutations(reference_protein.wildtype_sequence, muts)
        except Exception as e:
            return None, ValidationIssue(
                row_index=0,
                variant_id=variant_id,
                error_type="ParseError",
                message=str(e),
                mutation_string=mut_str,
            )

        var, issue = validate_variant_against_reference(
            row_idx=0,
            variant_id=variant_id,
            mutated_seq_raw=mutated_seq,
            mutation_str_raw=mut_str,
            reference_protein=reference_protein,
        )
        if issue:
            return None, issue

        # Extract stability measurements
        metadata: Dict[str, Any] = {
            "sign_convention": sign_convention,
            "experimental_method": row.get("method"),
            "pH": row.get("pH"),
            "temperature_C": row.get("temperature"),
        }

        # Check ddG or dTm
        ddg_raw = row.get("ddG")
        dtm_raw = row.get("dTm")

        if ddg_raw is not None and str(ddg_raw).strip() != "":
            try:
                metadata["ddG_kcal_mol"] = float(ddg_raw)
            except ValueError:
                pass

        if dtm_raw is not None and str(dtm_raw).strip() != "":
            try:
                metadata["dTm_celsius"] = float(dtm_raw)
            except ValueError:
                pass

        var.metadata = metadata
        return var, None
