from pathlib import Path
from PIL import Image
import yaml

import json, logging, os
from kalos.config import YoloToKalosCOCOConfig
from kalos.utils.logging import setup_kalos_logging

logger = logging.getLogger(__name__)

EXT = ['jpg', 'jpeg', 'bmp', 'png']
ROOT_DIR = os.getcwd()
    
def _build_rater_dirs(rater_folders: list) -> dict:
    """Builds a mapping of rater IDs to their respective annotation folders."""
    return {
        f"labeler {i}": Path(folder)
        for i, folder in enumerate(rater_folders, start=1)
    }

def _collect_image_files(base_dir: Path) -> list[Path]:
    files = []
    for ext in EXT:
        files.extend(sorted(base_dir.rglob(f"*.{ext}")))
    return files

def _get_rater_list(img_path: Path, rater_dirs: dict) -> list[str]:
    """Returns a list of rater IDs that have annotations for the given image."""
    return [
        rater_id
        for rater_id, folder in rater_dirs.items()
        if (folder / img_path.with_suffix(".txt").name).exists()
    ]

def _parse_image(img_path: Path, img_id: int, rater_dirs: dict) -> tuple[dict, tuple]:
    """Parses image file to create COCO image entry and returns dimensions."""
    with Image.open(img_path) as img:
        W, H = img.size

    rater_list = _get_rater_list(img_path, rater_dirs)
    image_entry = {
        "id": img_id,
        "file_name": img_path.name,
        "height": H,
        "width": W,
        "rater_list": rater_list,
    }
    return image_entry, (W, H)

def _parse_annotations(
    txt_path: Path,
    rater_id: str,
    image_id_map: dict,
    ann_id_start: int,
) -> list[dict]:
    """Parses YOLO annotation txt file and converts to COCO format."""
    stem = txt_path.stem
    matched = next(
        (fname for fname in image_id_map if Path(fname).stem == stem), None
    )
    if matched is None:
        return []

    cur_img_id, W, H = image_id_map[matched]
    annotations = []

    with open(txt_path) as f:
        for ann_id, line in enumerate(f, start=ann_id_start):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            x_min = (xc - w / 2) * W
            y_min = (yc - h / 2) * H
            bw, bh = w * W, h * H
            annotations.append({
                "id": ann_id,
                "image_id": cur_img_id,
                "category_id": int(cls) + 1,
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "rater_id": rater_id,
            })
    return annotations

def _load_categories_from_yaml(rater_folders: list) -> list[dict]:
    """Assumes all raters share the same data.yaml structure."""
    for folder in rater_folders:
        yaml_files = list(Path(folder).rglob("data.yaml"))
        if not yaml_files:
            continue
        with open(yaml_files[0]) as f:
            data = yaml.safe_load(f)
            
        names = data.get("names", {})
        return [
            {"id": i + 1, "name": name}
            for i, name in (names.items() if isinstance(names, dict) else enumerate(names))
        ]
    raise FileNotFoundError("data.yaml")

def yolo_to_kalos_coco_pipeline(cfg: YoloToKalosCOCOConfig):
    """Converts YOLO annotations (only-bbox) to KaLOS-COCO format."""
    setup_kalos_logging(cfg.log_level)
    
    rater_dirs = _build_rater_dirs(cfg.rater_folders)
    base_rater = next(iter(rater_dirs.values()))
    
    if not len(rater_dirs) >= 2:
        logger.error("At least two rater folders are required.")
        raise ValueError("At least two rater folders are required.")

    if not base_rater.exists():
        logger.error(f"Base rater folder does not exist: {base_rater}")
        raise Exception(f"Base rater folder does not exist: {base_rater}")

    images, image_id_map = [], {}
    for img_id, img_path in enumerate(_collect_image_files(base_rater), start=1):
        image_entry, (W, H) = _parse_image(img_path, img_id, rater_dirs)
        images.append(image_entry)
        image_id_map[img_path.name] = (img_id, W, H)

    annotations = []
    for rater_id, folder in rater_dirs.items():
        for txt_path in sorted(folder.rglob("*.txt")):
            parsed = _parse_annotations(txt_path, rater_id, image_id_map, ann_id_start=len(annotations) + 1)
            annotations.extend(parsed)

    coco = {"images": images, "annotations": annotations, "categories": _load_categories_from_yaml(cfg.rater_folders)}
    if cfg.output_path == './' or cfg.output_path == '/':
        output_path = os.path.join(Path.cwd(), "kalos_coco_annotation.json")
    else:
        output_path = os.path.join(Path(cfg.output_path), "kalos_coco_annotation.json")

    if os.path.exists(output_path):
        logger.warning(f"Output file already exists and will be overwritten: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Success: images {len(images)}, annotations {len(annotations)}")
    logger.info(f"Save Path: {output_path}")