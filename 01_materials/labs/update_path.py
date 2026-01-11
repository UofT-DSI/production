from pathlib import Path
import sys

notebook_dir = Path.cwd()
src_path = (notebook_dir / "../../05_src").resolve()

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))  # insert(0) gives it priority