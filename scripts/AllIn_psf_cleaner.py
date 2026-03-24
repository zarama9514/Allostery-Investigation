from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
from MDAnalysis.lib.formats.libdcd import DCDFile


EXCLUDED_SEGIDS = {"TIP3", "IONS", "MEMB"}
EXCLUDED_RESNAMES = {"TIP3", "SOD", "CLA", "POPC", "CHL1"}


def dcd_natoms(dcd_path: Path) -> int:
    with DCDFile(str(dcd_path)) as dcd:
        return int(dcd.header["natoms"])


def is_psf_compatible(psf_path: Path, dcd_path: Path) -> tuple[bool, int | str]:
    try:
        u = mda.Universe(str(psf_path), str(dcd_path))
        return True, int(u.atoms.n_atoms)
    except Exception as exc:
        return False, str(exc)


def contaminant_atoms(psf_path: Path, dcd_path: Path) -> int:
    u = mda.Universe(str(psf_path), str(dcd_path))
    selection = (
        f"segid {' '.join(sorted(EXCLUDED_SEGIDS))} or "
        f"resname {' '.join(sorted(EXCLUDED_RESNAMES))}"
    )
    return int(u.select_atoms(selection).n_atoms)


def candidate_psfs(input_psf: Path) -> Iterable[Path]:
    base = input_psf.parent
    stem = input_psf.stem
    yield base / f"{stem}_protein.psf"
    yield base / f"{stem}_nowat.psf"
    yield base / "step5_input_protein.psf"
    yield base / "step5_input_nowat.psf"


def pick_clean_candidate(input_psf: Path, dcd_path: Path) -> Path | None:
    target_natoms = int(dcd_natoms(dcd_path))
    for candidate in candidate_psfs(input_psf):
        if not candidate.exists():
            continue
        ok, atoms = is_psf_compatible(candidate, dcd_path)
        if not ok:
            continue
        if int(atoms) != target_natoms:
            continue
        if contaminant_atoms(candidate, dcd_path) != 0:
            continue
        return candidate
    return None


def try_clean_with_parmed(input_psf: Path, output_psf: Path) -> bool:
    try:
        import parmed as pmd
    except Exception:
        return False
    try:
        psf = pmd.load_file(str(input_psf))
        keep = []
        for i, atom in enumerate(psf.atoms):
            segid = str(getattr(atom.residue, "segid", "")).strip()
            resname = str(getattr(atom.residue, "name", "")).strip()
            if segid in EXCLUDED_SEGIDS:
                continue
            if resname in EXCLUDED_RESNAMES:
                continue
            keep.append(i)
        if not keep:
            return False
        filtered = psf[keep]
        filtered.save(str(output_psf), overwrite=True, format="psf")
        return True
    except Exception:
        return False


def backup_if_needed(input_psf: Path) -> None:
    backup_path = input_psf.with_suffix(".full_backup.psf")
    if not backup_path.exists():
        shutil.copy2(input_psf, backup_path)


def clean_psf(input_psf: Path, dcd_path: Path, output_psf: Path, backup: bool = True) -> Path:
    if output_psf.resolve() == input_psf.resolve() and backup:
        backup_if_needed(input_psf)
    if try_clean_with_parmed(input_psf, output_psf):
        ok, _ = is_psf_compatible(output_psf, dcd_path)
        if ok and contaminant_atoms(output_psf, dcd_path) == 0:
            return output_psf
    candidate = pick_clean_candidate(input_psf, dcd_path)
    if candidate is None:
        raise RuntimeError(
            "No clean compatible PSF found. Install parmed in the current environment or place a prebuilt *_protein.psf."
        )
    shutil.copy2(candidate, output_psf)
    ok, msg = is_psf_compatible(output_psf, dcd_path)
    if not ok:
        raise RuntimeError(f"Output PSF is not compatible with DCD: {msg}")
    return output_psf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-psf", required=True)
    parser.add_argument("--dcd", required=True)
    parser.add_argument("--output-psf", required=True)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_psf = Path(args.input_psf)
    dcd_path = Path(args.dcd)
    output_psf = Path(args.output_psf)
    output = clean_psf(input_psf, dcd_path, output_psf, backup=not args.no_backup)
    ok, atoms = is_psf_compatible(output, dcd_path)
    if not ok:
        raise RuntimeError(str(atoms))
    print(f"cleaned_psf={output}")
    print(f"atoms={int(atoms)}")
    print(f"contaminant_atoms={contaminant_atoms(output, dcd_path)}")
    print("compatible_with_dcd=true")


if __name__ == "__main__":
    main()
