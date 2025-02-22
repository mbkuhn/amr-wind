import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import amrex_particle
from amrex_particle import AmrexParticleFile

def main():
    """Compare particle files"""
    parser = argparse.ArgumentParser(description="A comparison tool for particles")
    parser.add_argument("f0", help="A particle directory", type=str)
    parser.add_argument("f1", help="A particle directory", type=str)
    parser.add_argument(
        "-a",
        "--abs_tol",
        help="Absolute tolerance (default is 0)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-r",
        "--rel_tol",
        help="Relative tolerance (default is 0)",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    assert Path(args.f0).is_dir()
    assert Path(args.f1).is_dir()

    p0f = AmrexParticleFile(Path(args.f0) / "particles")
    p1f = AmrexParticleFile(Path(args.f1) / "particles")
    p0df = p0f()
    p1df = p1f()
    assert np.abs(p0f.info["time"] - p1f.info["time"]) <= args.abs_tol
    assert p0df.shape == p1df.shape
    p0df.sort_values(by=["uid"], inplace=True, kind="stable", ignore_index=True)
    p1df.sort_values(by=["uid"], inplace=True, kind="stable", ignore_index=True)

    adiff = np.sqrt(np.square(p0df - p1df).sum(axis=0))
    rdiff = np.sqrt(np.square(p0df - p1df).sum(axis=0)) / np.sqrt(
        np.square(p0df).sum(axis=0)
    )
    adiff = adiff.to_frame(name="absolute_error")
    rdiff = rdiff.to_frame(name="relative_error")
    diff = pd.concat([adiff, rdiff], axis=1).fillna(value={"relative_error": 0.0})

    print(diff)
    assert (diff["absolute_error"] <= args.abs_tol).all()
    assert (diff["relative_error"] <= args.rel_tol).all()


if __name__ == "__main__":
    main()
