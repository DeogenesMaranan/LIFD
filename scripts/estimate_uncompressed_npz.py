import sys
import tarfile
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def estimate_npz_size(npz_bytes: bytes):
    """Estimate uncompressed size of a .npz file from bytes."""
    import io
    with io.BytesIO(npz_bytes) as f:
        with np.load(f, allow_pickle=True) as data:
            return sum(arr.nbytes for arr in data.values())

def estimate_npz_in_tar_member(member, tar_fileobj):
    """Read a member and return uncompressed size."""
    f = tar_fileobj.extractfile(member)
    if f is None:
        return 0
    try:
        npz_bytes = f.read()
        return estimate_npz_size(npz_bytes)
    except Exception as e:
        print(f"Warning: Failed to read {member.name}: {e}")
        return 0

def estimate_tar_size(tar_path: Path, max_workers=4):
    """Estimate total uncompressed size of .npz files in a tar using multithreading."""
    total_size = 0
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".npz")]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(estimate_npz_in_tar_member, m, tar): m for m in members}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Estimating {tar_path.name}", leave=False):
                total_size += future.result()
    return total_size

def estimate_root_tar_size(root_dir: Path, max_workers=4):
    root_dir = Path(root_dir)
    tar_files = [p for p in root_dir.rglob("*") if p.suffix in [".tar", ".tgz", ".gz"]]
    total_bytes = 0

    # Process multiple tar files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(estimate_tar_size, tar_path, max_workers=max_workers): tar_path for tar_path in tar_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tar archives"):
            tar_path = futures[future]
            size = future.result()
            print(f"{tar_path}: {size / 1e9:.2f} GB uncompressed")
            total_bytes += size

    print(f"Estimated total uncompressed size: {total_bytes / 1e9:.2f} GB")
    return total_bytes

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python estimate_uncompressed_npz_fast.py /path/to/root")
        raise SystemExit(1)
    root_dir = Path(sys.argv[1])
    estimate_root_tar_size(root_dir, max_workers=8)  # adjust threads as needed
