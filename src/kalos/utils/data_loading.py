"""
Centralized data loading and preprocessing for KaLOS.
Provides a unified interface for loading COCO-JSON, LIDC-IDRI, and YOLO 
annotation formats into the standardized KaLOS internal representation.
"""

import json
import logging
import os
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# --- 1. Core Annotation Formats (COCO & LIDC) ---
def load_annotations(file_path: Path) -> Dict[str, Any]:
    """
    Loads a COCO-style JSON annotation file.

    Args:
        file_path (Path): The path to the JSON annotation file.

    Returns:
        Dict[str, Any]: The loaded annotation data as a Python dictionary.
    """
    logger.info(f"Loading annotations from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.debug("Annotations loaded successfully.")
    return data

def _preprocess_coco(coco_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw COCO data to group annotations by image and then by rater.
    Supports both standard List rater_list and Dict session-based rater_list.

    This function restructures the data to make it easier to access all annotations
    for a specific image, subdivided by the annotator who created them.

    Args:
        coco_data (Dict[str, Any]): The raw data loaded from the COCO-style JSON file.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each key is an `image_id`.
        The value is another dictionary containing the image's 'file_name',
        'rater_list', and a dictionary of 'annotations_by_rater'.
    """
    processed_data = {}

    # First, create a base structure for each image
    image_id_map = {img['id']: img for img in coco_data['images']}
    for image_id, img_info in image_id_map.items():
        if "rater_list" not in img_info:
            raise ValueError(f"Image {image_id} missing mandatory attribute 'rater_list'.")
        
        raw_list = img_info['rater_list']
        
        # Branch 1: Standard List format
        if isinstance(raw_list, list):
            flattened_list = raw_list
            image_session_mode = False
        # Branch 2: Dictionary session format
        elif isinstance(raw_list, dict):
            # Flatten dict { "Rater": [1, 2] } into ["Rater (S1)", "Rater (S2)"]
            flattened_list = []
            for rater_id, sessions in raw_list.items():
                for s_id in sessions:
                    flattened_list.append(f"{rater_id} (S{s_id})")
            image_session_mode = True
        else:
            raise TypeError(f"Invalid rater_list type for image {image_id}. Expected list or dict.")

        processed_data[image_id] = {
            'file_name': img_info['file_name'],
            'rater_list': flattened_list,
            'is_session_mode': image_session_mode,
            'annotations_by_rater': defaultdict(list)
        }

    # Now, populate the structure with annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in processed_data:
            continue
            
        img_meta = image_id_map[image_id]
        width, height = img_meta['width'], img_meta['height']
        
        if "rater_id" not in ann:
            raise ValueError(f"Annotation {ann.get('id')} missing mandatory attribute 'rater_id'.")

        # Handle Identity Transformation
        rater_id = ann['rater_id']
        if processed_data[image_id]['is_session_mode']:
            s_id = ann.get('session_id')
            if s_id is None:
                raise ValueError(f"Session-mode detected for image {image_id}, but annotation {ann.get('id')} is missing 'session_id'.")
            # Map to the flattened identity
            internal_identity = f"{rater_id} (S{s_id})"
            # Update the annotation object to reflect its virtual session identity
            ann['rater_id'] = internal_identity
        else:
            internal_identity = rater_id

        # Relative coordinate conversion (bbox)
        if "bbox" in ann:
            bbox = ann['bbox']
            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width
            bbox[3] /= height
        # Relative coordinate conversion (segmentation)
        if "segmentation" in ann:
            ann['segmentation'] = [
                [
                    coord / width if i % 2 == 0 else coord / height
                    for i, coord in enumerate(polygon)
                ]
                for polygon in ann['segmentation']
            ]
        if "keypoints" in ann:
            # normalize coco keypoints
            keypoints = ann['keypoints']
            # keypoints are xyv, where v is visibility.
            # v=0: not labeled (x=y=0), v=1: labeled but not visible, v=2: labeled and visible
            for i in range(0, len(keypoints), 3):
                if keypoints[i+2] > 0: # only normalize labeled keypoints
                    keypoints[i] /= width
                    keypoints[i+1] /= height

        # Only add if the rater/session is in the assigned list for this image
        if internal_identity in processed_data[image_id]['rater_list']:
            processed_data[image_id]['annotations_by_rater'][internal_identity].append(ann)
            
    return processed_data

def _preprocess_lidc_idri_data(lidc_idri_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw LIDC-IDRI data into the internal standardized format.

    Args:
        lidc_idri_data (Dict[str, Any]): Raw LIDC-IDRI dictionary.

    Returns:
        Dict[int, Dict[str, Any]]: Preprocessed data structure compatible with KaLOS.
    """
    preprocess_data = {}

    for study_instance_uid, values in lidc_idri_data.items():
        # don't use case id, it is sometimes used multiple times
        preprocess_data[study_instance_uid] = {
            "file_name": values["file_paths"][0], # only first file extracted
            "rater_list": list(values["annotators"].keys()),
            "annotations_by_rater": defaultdict(list)
        }

    # second pass
    max_z = {}
    min_z = {}
    for study_instance_uid, values in lidc_idri_data.items():
        max_z[study_instance_uid] = float("-inf")
        min_z[study_instance_uid] = float("inf")
        for rater_id, annotations in values["annotators"].items():
            for annotation in annotations:
                for contour in annotation["contours"]:
                    max_z[study_instance_uid] = max(max_z[study_instance_uid], float(contour["z_position"]))
                    min_z[study_instance_uid] = min(min_z[study_instance_uid], float(contour["z_position"]))

    ann_id = 0
    # third pass populate the dictonary with annotations
    for study_instance_uid, values in lidc_idri_data.items():
        width, height, depth = values["width"], values["height"], values["depth"]
        z_range = max_z[study_instance_uid] - min_z[study_instance_uid]
        for rater_id, annotations in values["annotators"].items():
            # create empty dict
            preprocess_data[study_instance_uid]["annotations_by_rater"][rater_id] = []
            # fill with annotation data
            for annotation in annotations:
                for contour in annotation["contours"]:
                    if z_range > 0:
                        contour["z_position"] = (float(contour["z_position"]) - min_z[study_instance_uid]) / z_range
                    else:
                        contour["z_position"] = 0.0
                    contour["points"] = [[point[0] / width, point[1] / height] for point in contour["points"]]
                    assert 0 <= contour["z_position"] <= 1.0
                    assert all(0 <= x <= 1 for row in contour["points"] for x in row)
                ann = {"category_id": 1, "segmentation_3d": annotation["contours"], "id": ann_id, "rater_id": rater_id}
                ann_id += 1
                preprocess_data[study_instance_uid]["annotations_by_rater"][rater_id].append(
                    ann
                )

    return preprocess_data

def preprocess_data(annotation_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Pre-processes raw annotation data to group annotations by image and then by rater.

    This function inspects the data format and dispatches to the appropriate
    pre-processing function. Currently supports COCO and LIDC-IDRI formats. YOLO is loaded with a separate function.

    Args:
        annotation_data (Dict[str, Any]): The raw data loaded from the JSON file.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each key is an `image_id`.
        The value is another dictionary containing the image's 'file_name',
        'rater_list', and a dictionary of 'annotations_by_rater'.
    """
    logger.info("Preprocessing data...")

    # --- Data Format Dispatcher ---
    # Check for a key that is highly specific to the COCO format.
    if 'images' in annotation_data and 'annotations' in annotation_data:
        logger.debug("   - Detected COCO data format.")
        processed_data = _preprocess_coco(annotation_data)
    elif len(annotation_data) > 0 and "study_instance_uid" in next(iter(annotation_data.values())):
        processed_data = _preprocess_lidc_idri_data(annotation_data)
    else:
        raise NotImplementedError("Unsupported data format. Only COCO-style JSON is currently supported.")

    logger.debug(f"Preprocessing complete. Found data for {len(processed_data)} images.")
    return processed_data

# --- 2. YOLO Adapter ---

def load_yolo_dataset(dataset_root: Path) -> Dict[str, Any]:
    """
    Adapter for YOLO format following the multi-rater directory structure.
    Enforces [root]/[rater_id]/[files...] structure.

    Args:
        dataset_root (Path): Root directory containing rater subfolders.

    Returns:
        Dict[str, Any]: Standardized KaLOS package with processed_data and categories.
    """
    # 1. Load categories from data.yaml (root or first rater folder)
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        # Search recursively for the first data.yaml
        yaml_files = list(dataset_root.rglob("data.yaml"))
        if yaml_files:
            yaml_path = yaml_files[0]
        else:
            logger.warning(f"data.yaml not found in {dataset_root}. Using numeric category IDs.")
            yaml_path = None

    if yaml_path:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        names = yaml_data.get("names", {})
        categories = {i + 1: name for i, name in (names.items() if isinstance(names, dict) else enumerate(names))}
    else:
        categories = {}

    processed_data = {}
    all_images = set()
    
    # 2. Identify Raters (Subdirectories)
    rater_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    # Standardize Rater IDs and Sessions (Identity Virtualization)
    rater_id_map = {} 
    for r_dir in rater_dirs:
        folder_name = r_dir.name
        # Simple Identity Virtualization: Rater1_S1 -> Rater1 (S1)
        if "_S" in folder_name:
            parts = folder_name.split("_S")
            normalized_id = f"{parts[0]} (S{parts[1]})"
        else:
            normalized_id = folder_name
        rater_id_map[folder_name] = normalized_id
        
        # Collect all image names from this rater
        for txt_file in r_dir.glob("*.txt"):
            all_images.add(txt_file.stem)

    # 3. Build processed_data structure
    image_id_map = {name: i for i, name in enumerate(sorted(list(all_images)), start=1)}
    
    for img_name, img_id in image_id_map.items():
        assigned_raters = []
        annotations_by_rater = defaultdict(list)
        
        for folder_name, normalized_id in rater_id_map.items():
            txt_path = dataset_root / folder_name / f"{img_name}.txt"
            if txt_path.exists():
                assigned_raters.append(normalized_id)
                
                # Parse YOLO annotations
                with open(txt_path, 'r') as f:
                    for ann_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts)
                        
                        # Convert to COCO-style [x_min, y_min, bw, bh]
                        x_min = xc - w / 2
                        y_min = yc - h / 2
                        
                        annotations_by_rater[normalized_id].append({
                            "id": f"{img_id}_{normalized_id}_{ann_idx}",
                            "image_id": img_id,
                            "category_id": int(cls) + 1,
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "rater_id": normalized_id
                        })
        
        processed_data[img_id] = {
            'file_name': f"{img_name}",
            'rater_list': sorted(assigned_raters),
            'annotations_by_rater': annotations_by_rater
        }

    return {
        "processed_data": processed_data,
        "categories": categories,
        "all_raters": sorted(list(rater_id_map.values()))
    }

def load_and_preprocess_data(path: Path, annotation_type: str) -> Dict[str, Any]:
    """
    Unified entry point for all data loading in KaLOS.

    Args:
        path (Path): Path to the annotation file or directory.
        annotation_type (str): Format type ('coco-json', 'lidc-idri-json', 'yolo').

    Returns:
        Dict[str, Any]: Package containing 'processed_data', 'categories', and 'all_raters'.
    """
    if annotation_type == 'yolo':
        return load_yolo_dataset(path)

    ### Add new datatypes here and implement a function above.
    
    # Load Core Annotation Formats
    raw_data = load_annotations(path)
    if annotation_type == 'coco-json':
        categories = {cat["id"]: cat["name"] for cat in raw_data.get("categories", [])}
        processed_data = preprocess_data(raw_data)
    elif annotation_type == 'lidc-idri-json':
        categories = {}
        processed_data = preprocess_data(raw_data)
    else:
        raise ValueError(f"Unsupported annotation type: {annotation_type}")

    # Derive all_raters from the processed data
    all_raters_set = set()
    for img_data in processed_data.values():
        all_raters_set.update(img_data['rater_list'])
    
    return {
        "processed_data": processed_data,
        "categories": categories,
        "all_raters": sorted(list(all_raters_set))
    }
