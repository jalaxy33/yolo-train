import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Self
import tomllib

sys.path.append(str(Path(__file__).parent))
from path_utils import get_project_root


@dataclass
class YoloSettings:
    datasets_dir: str
    weights_dir: str
    runs_dir: str

    def __post_init__(self):
        self.datasets_dir = self._check_dir_str(self.datasets_dir)
        self.weights_dir = self._check_dir_str(self.weights_dir)
        self.runs_dir = self._resolve_dir_str(self.runs_dir)

    @classmethod
    def from_toml(cls, toml_path: Path, verbose: bool = True) -> Self:
        """Load YOLO settings from a TOML configuration file."""

        assert toml_path.is_file() and toml_path.suffix == ".toml", (
            f"Invalid TOML file path: {toml_path}"
        )
        with toml_path.open("rb") as f:
            configs = tomllib.load(f)

        yolo_settings = cls(**configs["yolo-settings"])
        yolo_settings.apply()
        if verbose:
            yolo_settings.display()
        return yolo_settings

    # -- Public APIs

    def apply(self) -> None:
        from ultralytics import settings

        settings["datasets_dir"] = self.datasets_dir
        settings["weights_dir"] = self.weights_dir
        settings["runs_dir"] = self.runs_dir

    def display(self) -> None:
        from ultralytics import settings
        from pprint import pprint

        print("Current YOLO settings:")
        pprint(settings)
        print()

    # -- Internal methods

    def _check_dir_str(self, dir_str: str) -> str:
        dir = self._check_dir_path(Path(dir_str))
        return str(dir)

    def _check_dir_path(self, dir: Path) -> Path:
        dir = self._resolve_dir(dir)
        assert dir.is_dir(), f"Directory {dir} does not exist."
        return dir

    def _resolve_dir_str(self, dir_str: str) -> str:
        dir = str(self._resolve_dir(Path(dir_str)))
        return str(dir)

    def _resolve_dir(self, dir: Path) -> Path:
        if not dir.is_absolute():
            dir = get_project_root().joinpath(dir)
        return dir.resolve()


def load_yolo_settings(config_toml: Path, verbose: bool = True) -> YoloSettings:
    """Adjust YOLO settings from a TOML configuration file."""
    return YoloSettings.from_toml(config_toml, verbose=verbose)


if __name__ == "__main__":
    import os

    os.chdir(Path(__file__).parent)

    project_root = get_project_root()
    toml_path = project_root.joinpath("assets/configs/phantom-body.toml")

    _ = load_yolo_settings(toml_path, verbose=True)
