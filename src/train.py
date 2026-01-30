import os
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent))
from utils.yolo_settings import load_yolo_settings
from utils.path_utils import get_project_root


def main():
    os.chdir(Path(__file__).parent)

    project_root = get_project_root()

    # Load ultralytics settings
    yolo_config_toml = project_root.joinpath("assets/configs/phantom-body.toml")
    yolo_settings = load_yolo_settings(yolo_config_toml)

    # get checkpoint and data
    model_name = "yolo26n-seg"
    data_yamlfile = "phantom-body.train.yaml"

    checkpoint_path = Path(yolo_settings.weights_dir).joinpath(f"{model_name}.pt")
    data_yaml = Path(yolo_settings.datasets_dir).joinpath(data_yamlfile)

    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"
    assert data_yaml.is_file(), f"Data YAML not found: {data_yaml}"

    # load model checkpoint
    model = YOLO(checkpoint_path)

    # train
    epochs = 20
    model.train(
        data=data_yaml,
        epochs=epochs,
        project=model_name,
        name=f"epoch{epochs:02d}",
        exist_ok=True,
    )

    # export to ONNX
    model.export(format="onnx", half=True, dynamic=True, simplify=True)


if __name__ == "__main__":
    main()
