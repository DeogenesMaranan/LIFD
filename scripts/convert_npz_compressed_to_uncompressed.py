import sys
import tarfile
from pathlib import Path
import numpy as np
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def convert_npz_file(npz_path: Path):
    """Convert a compressed .npz file to uncompressed in-place."""
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            arrays = {name: data[name] for name in data.files}
        np.savez(npz_path, **arrays)
        return True
    except Exception as e:
        print(f"Failed to convert {npz_path}: {e}")
        return False

def convert_tar_to_output(tar_path: Path, root_dir: Path, output_dir: Path, max_workers=4):
    """
    Extract a tar file, convert .npz files to uncompressed, and write a new tar
    in the output directory while keeping relative folder structure.
    """
    # Compute relative path to maintain folder structure
    rel_path = tar_path.relative_to(root_dir)
    output_tar_path = output_dir / rel_path
    output_tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Extract all files
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(tmpdir_path)

        # Find all .npz files
        npz_files = list(tmpdir_path.rglob("*.npz"))

        # Multithreaded conversion
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(convert_npz_file, npz): npz for npz in npz_files}
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Converting .npz in {tar_path.name}"):
                pass  # tqdm updates automatically

        # Repack tar with same compression
        mode = "w:gz" if tar_path.suffix in [".gz", ".tgz"] else "w"
        with tarfile.open(output_tar_path, mode) as tar_out:
            for file_path in tmpdir_path.rglob("*"):
                tar_out.add(file_path, arcname=file_path.relative_to(tmpdir_path))

    print(f"Converted tar written to: {output_tar_path}")

def convert_root_to_output(root: Path, output_dir: Path, max_workers=4):
    root = Path(root)
    tar_files = [p for p in root.rglob("*") if p.suffix in [".tar", ".tgz", ".gz"]]
    for tar_path in tqdm(tar_files, desc="Tar archives"):
        convert_tar_to_output(tar_path, root, output_dir, max_workers=max_workers)
    print(f"Done. Processed {len(tar_files)} tar archives.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_tar_fast_multithreaded.py /path/to/root /path/to/output_dir")
        raise SystemExit(1)
    root_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    convert_root_to_output(root_dir, output_dir, max_workers=8)  # adjust max_workers as needed
