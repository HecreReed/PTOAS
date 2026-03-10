from pathlib import Path
import runpy


if __name__ == "__main__":
    samples_dir = Path(__file__).resolve().parents[1]
    runpy.run_path(str(samples_dir / "Addsc" / "addsc.py"), run_name="__main__")

