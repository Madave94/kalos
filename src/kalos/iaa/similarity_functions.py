"""
Distance and similarity functions for comparing geometric annotations.
Provides implementations for IoU, GIoU, MPJPE, and 3D IoU across different task types.
"""

import numpy as np
import logging
from typing import Dict, Any, List

from PIL import Image, ImageDraw
from skimage.draw import polygon
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# --- Keypoint Distance Functions ---
def image_normalized_mpjpe_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any]) -> float:
    """
    Calculates an image-normalized mean per joint position similarity.

    This metric operates in relative image coordinates (0-1 range). It returns a
    similarity value between 0.0 (maximum disagreement) and 1.0 (perfect match).
    The similarity is calculated as `1.0 - mean_distance`.

    - For joints visible in BOTH annotations, it calculates the Euclidean distance,
      normalized by sqrt(2) (the diagonal of the unit square) to be in [0, 1].
    - For joints visible in ONLY ONE annotation, it assigns a maximum penalty of 1.0.
    - Joints not visible in either annotation are ignored.
    - The final distance is the average error over all joints that are visible
      in at least one of the annotations.

    Args:
        ann1 (Dict[str, Any]): The first annotation in COCO keypoint format,
                             with coordinates expected to be normalized to the [0, 1] range.
        ann2 (Dict[str, Any]): The second annotation, with normalized coordinates.

    Returns:
        float: The normalized MPJPE similarity, a value between 0.0 and 1.0.
               Returns 1.0 if no keypoints are visible in either annotation,
               as there is no disagreement.
    """
    if 'keypoints' not in ann1 or 'keypoints' not in ann2:
        raise ValueError("Annotations for MPJPE must contain a 'keypoints' key.")

    kpts1 = np.array(ann1['keypoints']).reshape(-1, 3)
    kpts2 = np.array(ann2['keypoints']).reshape(-1, 3)

    # Fail-fast: Check if coordinates are normalized
    # Using a threshold of 2.0 to allow for some points slightly outside the image,
    # but catch obvious pixel coordinates (e.g., 100, 200).
    if np.any(kpts1[:, :2] > 2.0) or np.any(kpts2[:, :2] > 2.0):
        raise ValueError("Keypoint coordinates appear to be in pixel values. They must be normalized to [0, 1].")

    # Visibility flags (v > 0 means visible/labeled)
    vis1 = kpts1[:, 2] > 0
    vis2 = kpts2[:, 2] > 0

    # The set of joints to consider is the union of all visible joints
    visible_in_either = vis1 | vis2
    num_joints_to_compare = np.sum(visible_in_either)

    if num_joints_to_compare == 0:
        # No visible keypoints in either annotation, so no basis for disagreement.
        return 1.0

    total_distance = 0.0

    # 1. Calculate positional error for joints visible in both
    visible_in_both = vis1 & vis2
    if np.any(visible_in_both):
        p1 = kpts1[visible_in_both, :2]
        p2 = kpts2[visible_in_both, :2]
        distances = np.linalg.norm(p1 - p2, axis=1)
        # Normalize by the max possible distance in a 1x1 square (sqrt(2))
        normalized_distances = distances / np.sqrt(2)
        total_distance += np.sum(normalized_distances)

    # 2. Calculate visibility error for joints visible in only one
    # This is the symmetric difference of the visibility sets.
    visible_in_one_only = np.logical_xor(vis1, vis2)
    num_mismatched_visibility = np.sum(visible_in_one_only)
    # Add a penalty of 1.0 for each mismatch
    total_distance += num_mismatched_visibility * 1.0

    # 3. The final distance is the mean over all considered joints.
    mean_distance = total_distance / num_joints_to_compare

    return 1.0 - mean_distance


# --- Segmentation and BBox ---

def centroid_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any]) -> float:
    """
    Calculates a similarity score from the normalized distance between centroids.

    The centroid is approximated by the center of the bounding box. The distance
    is normalized by the diagonal of the larger bounding box. The similarity
    is `max(0, 1 - normalized_distance)`, resulting in a score between 0.0
    (dissimilar) and 1.0 (coincident).

    Args:
        ann1 (Dict[str, Any]): The first annotation, must contain a 'bbox'.
        ann2 (Dict[str, Any]): The second annotation, must contain a 'bbox'.

    Returns:
        float: The centroid similarity. Returns 0.0 if a bounding box has
               zero area, indicating dissimilarity.
    """
    bbox1 = ann1.get('bbox')
    bbox2 = ann2.get('bbox')
    if not bbox1 or not bbox2:
        raise ValueError("Annotations must have a 'bbox' for centroid_similarity.")

    # Bbox format is [x, y, width, height]
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate centroids
    c1_x, c1_y = x1 + w1 / 2, y1 + h1 / 2
    c2_x, c2_y = x2 + w2 / 2, y2 + h2 / 2

    # L2 distance
    l2_dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

    # For symmetric normalization, use the diagonal of the larger bounding box
    if w1 * h1 > w2 * h2:
        norm_factor = np.sqrt(w1**2 + h1**2)
    else:
        norm_factor = np.sqrt(w2**2 + h2**2)

    if norm_factor == 0:
        # If the larger bbox has no area, the concept is ill-defined.
        # Treat as completely dissimilar.
        return 0.0

    normalized_distance = l2_dist / norm_factor

    # Convert distance to similarity, clamped to the [0, 1] range.
    similarity = max(0.0, 1.0 - normalized_distance)

    return similarity

def segm_iou_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any]) -> float:
    """
    Calculates Intersection over Union (IoU) similarity for segmentations.

    This function dynamically chooses between polygon or mask implementations
    based on the 'segmentation' field format. Both annotations must use the 
    same format (either lists of polygons or RLE dicts).

    Args:
        ann1 (Dict[str, Any]): The first annotation. Must contain a 'segmentation' key.
        ann2 (Dict[str, Any]): The second annotation. Must contain a 'segmentation' key.

    Returns:
        float: The calculated IoU similarity score, between 0.0 and 1.0.

    Raises:
        ValueError: If either annotation is missing the 'segmentation' key.
        TypeError: If the segmentation formats are mixed or unsupported.
    """
    seg1 = ann1.get('segmentation')
    seg2 = ann2.get('segmentation')

    if seg1 is None or seg2 is None:
        raise ValueError("Annotations for segm_iou_similarity must contain a 'segmentation' key.")

    # Determine segmentation format
    is_poly1 = isinstance(seg1, list)
    is_poly2 = isinstance(seg2, list)
    is_rle1 = isinstance(seg1, dict)
    is_rle2 = isinstance(seg2, dict)

    # Case 1: Both are polygons
    if is_poly1 and is_poly2:
        # The function expects a list of polygons for each annotation
        # No try/except block here for fail-fast behavior as requested
        return calc_iou_segm_poly(seg1, seg2)
    # Case 2: Both are RLE masks (common for LVIS)
    elif is_rle1 and is_rle2:
        # The function expects a list of RLE dicts
        return calc_iou_segm_mask([seg1], [seg2])
    # Case 3: Mixed or unsupported formats
    else:
        format1 = "polygon" if is_poly1 else "RLE mask" if is_rle1 else "unknown"
        format2 = "polygon" if is_poly2 else "RLE mask" if is_rle2 else "unknown"
        raise TypeError(
            f"Mismatch or unsupported segmentation format. Annotation 1 is '{format1}', "
            f"Annotation 2 is '{format2}'. Both must be of the same type."
        )

def segm_giou_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any]) -> float:
    """
    Calculates a normalized Generalized Intersection over Union (GIoU) similarity.

    This function returns a normalized distance in the range [0, 1], where 1 is a
    perfect match and 0 is maximally distant. It currently only supports polygon
    segmentations.

    Args:
        ann1 (Dict[str, Any]): The first annotation. Must contain a 'segmentation' key.
        ann2 (Dict[str, Any]): The second annotation. Must contain a 'segmentation' key.

    Returns:
        float: The normalized GIoU similarity score, between 0.0 and 1.0.

    Raises:
        ValueError: If either annotation is missing the 'segmentation' key.
        NotImplementedError: If the segmentations are provided as RLE masks.
        TypeError: If the segmentation formats are mixed or unsupported.
    """
    seg1 = ann1.get('segmentation')
    seg2 = ann2.get('segmentation')

    if seg1 is None or seg2 is None:
        raise ValueError("Annotations for segm_giou_similarity must contain a 'segmentation' key.")

    # Determine segmentation format
    is_poly1 = isinstance(seg1, list)
    is_poly2 = isinstance(seg2, list)
    is_rle1 = isinstance(seg1, dict)
    is_rle2 = isinstance(seg2, dict)

    # Case 1: Both are polygons
    if is_poly1 and is_poly2:
        giou = calc_giou_segm_poly(seg1, seg2)
        return (1 + giou) / 2.0
    # Case 2: RLE masks not supported for GIoU yet
    elif is_rle1 and is_rle2:
        raise NotImplementedError("segm_giou_similarity for RLE masks is not yet implemented.")
    # Case 3: Mixed or unsupported formats
    else:
        format1 = "polygon" if is_poly1 else "RLE mask" if is_rle1 else "unknown"
        format2 = "polygon" if is_poly2 else "RLE mask" if is_rle2 else "unknown"
        raise TypeError(
            f"Mismatch or unsupported segmentation format. Annotation 1 is '{format1}', "
            f"Annotation 2 is '{format2}'. Both must be of the same type."
        )

def bbox_iou_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any]) -> float:
    """
    Calculates the Intersection over Union (IoU) similarity for two bounding boxes.

    Args:
        ann1 (Dict[str, Any]): The first annotation. Must contain a 'bbox' key 
            formatted as [x, y, width, height].
        ann2 (Dict[str, Any]): The second annotation. Must contain a 'bbox' key 
            formatted as [x, y, width, height].

    Returns:
        float: The calculated IoU similarity score, between 0.0 and 1.0.

    Raises:
        ValueError: If either annotation is missing the 'bbox' key.
    """
    bbox1 = ann1.get('bbox')
    bbox2 = ann2.get('bbox')

    if bbox1 is None or bbox2 is None:
        raise ValueError("Annotations must have a 'bbox' for bbox_iou_similarity.")

    boxA = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    boxB = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA, 0)))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def calc_iou_segm_poly(segm1: List[List[float]], segm2: List[List[float]]) -> float:
    """
    Calculates the IoU of two segmentation masks using Shapely.

    Args:
        segm1 (List[List[float]]): List of polygons, each represented as a list of coordinates.
        segm2 (List[List[float]]): List of polygons, each represented as a list of coordinates.

    Returns:
        float: The calculated IoU similarity score.
    """
    if not segm1 or not segm2:
        return 0.0

    # Function to convert list of coordinates into a list of tuples
    def coords_to_tuples(coords):
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # Convert the sub-shapes into Shapely polygons
    def create_polygons(coords_list):
        return [Polygon(coords_to_tuples(coords)) for coords in coords_list]

    def validate_polygon(geom):
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)  # Attempt to fix small invalidities
            if not geom.is_valid:
                geom = make_valid(geom)  # Attempt to fix further issues
            if not geom.is_valid:
                geom = geom.simplify(0.001, preserve_topology=True)  # Simplify if still invalid
            return geom
        except Exception as e:
            logger.warning(f"Executed non-topolgy preserving simplication {e}.")
            return geom.simplify(0.001, preserve_topology=False)  # Non-topology-preserving simplification as last resort

    # Create and validate polygons for both segmentations
    polygon1_shapes = [validate_polygon(p) for p in create_polygons(segm1)]
    polygon2_shapes = [validate_polygon(p) for p in create_polygons(segm2)]

    # Combine sub-shapes into a single (Multi)Polygon if necessary
    polygon1 = unary_union(polygon1_shapes) if len(polygon1_shapes) > 1 else polygon1_shapes[0]
    polygon2 = unary_union(polygon2_shapes) if len(polygon2_shapes) > 1 else polygon2_shapes[0]

    try:
        intersection = polygon1.intersection(polygon2)
        if intersection.is_empty or intersection.area == 0.0:
            return 0.0  # Early exit if there is no intersection
        union = polygon1.union(polygon2)
        if union.area == 0.0:
            return 0.0
    except Exception as e:
        logger.error(f"ShapelyError during intersection/union: {e}")
        return 0.0

    iou = intersection.area / union.area
    return iou

def calc_giou_segm_poly(segm1: List[List[float]], segm2: List[List[float]]) -> float:
    """
    Calculates the GIoU of two segmentation masks using Shapely.

    Args:
        segm1 (List[List[float]]): List of polygons, each represented as a list of coordinates.
        segm2 (List[List[float]]): List of polygons, each represented as a list of coordinates.

    Returns:
        float: The calculated GIoU score.
    """
    if not segm1 or not segm2:
        return 0.0

    # Helper functions from calc_iou_segm_poly
    def coords_to_tuples(coords):
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    def create_polygons(coords_list):
        return [Polygon(coords_to_tuples(coords)) for coords in coords_list]

    def validate_polygon(geom):
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_valid:
                geom = make_valid(geom)
            if not geom.is_valid:
                geom = geom.simplify(0.001, preserve_topology=True)
            return geom
        except Exception as e:
            logger.warning(f"Executed non-topolgy preserving simplication {e}.")
            return geom.simplify(0.001, preserve_topology=False)

    polygon1_shapes = [validate_polygon(p) for p in create_polygons(segm1)]
    polygon2_shapes = [validate_polygon(p) for p in create_polygons(segm2)]

    polygon1 = unary_union(polygon1_shapes) if len(polygon1_shapes) > 1 else polygon1_shapes[0]
    polygon2 = unary_union(polygon2_shapes) if len(polygon2_shapes) > 1 else polygon2_shapes[0]

    try:
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area

        if union_area == 0.0:
            return 1.0  # Both shapes are empty, perfect match

        iou = intersection_area / union_area

        # C is the smallest convex hull that encloses both A and B
        enclosing_hull = polygon1.union(polygon2).convex_hull
        enclosing_area = enclosing_hull.area

        if enclosing_area == 0.0:
            # This can happen if the union is a line or point, which has no area.
            # In this case, the penalty term is undefined. Fallback to IoU.
            return iou

        # Penalty term: |C \ (A U B)| / |C|
        penalty = (enclosing_area - union_area) / enclosing_area
        
        giou = iou - penalty
        
        return giou

    except Exception as e:
        logger.error(f"ShapelyError during GIoU calculation: {e}")
        return 0.0

def calc_iou_segm_mask(entry1, entry2, image_size):
    """
    Calculates the IoU of two RLE segmentation masks.

    Args:
        entry1: The first annotation entry containing a mask and bbox.
        entry2: The second annotation entry containing a mask and bbox.
        image_size (Tuple[int, int]): The dimensions of the image.

    Returns:
        float: The calculated IoU score.
    """
    mask1 = entry1.segm
    mask2 = entry2.segm
    bbox1 = entry1.bbox
    bbox2 = entry2.bbox

    if mask1 is None or mask2 is None:
        return 0.0

    bbox_iou = bbox_iou_similarity(entry1, entry2)
    if bbox_iou == 0.0:
        return 0.0

    def adjust_bbox_and_region(mask, bbox, image_size):
        """Adjust bounding box to fit within image boundaries and prepare the mask."""
        x1_f, y1_f = bbox[0] * image_size[1], bbox[1] * image_size[0]
        x2_f, y2_f = (bbox[0] + bbox[2]) * image_size[1], (bbox[1] + bbox[3]) * image_size[0]

        x1, x2 = int(np.floor(x1_f)), int(np.ceil(x2_f))
        y1, y2 = int(np.floor(y1_f)), int(np.ceil(y2_f))

        # Adjust if there's a mismatch in broadcasting dimensions
        if x2 - x1 != mask.shape[1]:
            if abs(x1_f - x1) <= abs(x2 - x2_f):
                x1 = max(0, x1 - 1 if abs(x1_f - x1) > 1 else x1)
            else:
                x2 = min(image_size[1], x2 + 1 if abs(x2 - x2_f) > 1 else x2)

        if y2 - y1 != mask.shape[0]:
            if abs(y1_f - y1) <= abs(y2 - y2_f):
                y1 = max(0, y1 - 1 if abs(y1_f - y1) > 1 else y1)
            else:
                y2 = min(image_size[0], y2 + 1 if abs(y2 - y2_f) > 1 else y2)

        # Create the adjusted full-sized mask
        adjusted_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        adjusted_mask[:mask.shape[0], :mask.shape[1]] = mask

        return x1, y1, x2, y2, adjusted_mask

    # Adjust masks and bounding boxes
    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, adjusted_mask1 = adjust_bbox_and_region(mask1, bbox1, image_size)
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2, adjusted_mask2 = adjust_bbox_and_region(mask2, bbox2, image_size)

    # Place adjusted masks into full-sized image masks
    full_mask1 = np.zeros(image_size, dtype=bool)
    full_mask2 = np.zeros(image_size, dtype=bool)

    full_mask1[bbox1_y1:bbox1_y2, bbox1_x1:bbox1_x2] = adjusted_mask1
    full_mask2[bbox2_y1:bbox2_y2, bbox2_x1:bbox2_x2] = adjusted_mask2

    # Calculate intersection and union on full-sized masks
    intersection = np.logical_and(full_mask1, full_mask2).sum()
    union = np.logical_or(full_mask1, full_mask2).sum()

    if union == 0:  # Avoid division by zero
        return 0.0

    return intersection / union

def mask_to_array(seg, width, height):
    """
    Converts a polygon segmentation mask to a binary numpy array.

    Args:
        seg: The polygon segmentation data.
        width (int): The width of the target image.
        height (int): The height of the target image.

    Returns:
        np.ndarray: A binary array representing the mask.
    """
    arr_seg = Image.new('L', (width, height), 0)
    ImageDraw.Draw(arr_seg).polygon(seg, outline=1, fill=1)
    return np.array(arr_seg)

# 3D Instance Voxel Grid
def segm_3d_iou_similarity(ann1: Dict[str, Any], ann2: Dict[str, Any], grid_size=(128, 128, 128)) -> float:
    """
    Calculates the 3D Intersection over Union (IoU) similarity between two voxelized
    objects from relative coordinates.

    The function assumes input coordinates ('z_position' and 'points') are relative,
    i.e., in the range [0, 1]. It scales these coordinates to a discrete voxel grid of
    a specified size, rasterizes the polygon slices, and then computes the IoU.
    The returned value is `IoU`, (1 for identity, 0 for maximum dissimilarity).

    Args:
        ann1 (Dict[str, Any]): The first annotation. Must contain a 'segmentation_3d'
                             key, a list of dicts, each with 'z_position' (float)
                             and 'points' (list of [x, y] floats).
        ann2 (Dict[str, Any]): The second annotation, with the same structure as ann1.
        grid_size (tuple, optional): The dimensions of the voxel grid in (depth, height, width).
                                     Defaults to (128, 128, 128).

    Returns:
        float: The 3D IoU similarity, a value between 0.0 and 1.0.

    Raises:
        ValueError: If annotations are missing 'segmentation_3d', data is malformed,
                    or coordinates are not in the expected [0, 1] range.
    """
    obj1_data = ann1.get('segmentation_3d')
    obj2_data = ann2.get('segmentation_3d')

    if obj1_data is None or obj2_data is None:
        raise ValueError("Annotations for 3D IoU must contain a 'segmentation_3d' key.")

    if not obj1_data and not obj2_data:
        # Both are empty, which is a perfect match
        return 1.0
    if not obj1_data or not obj2_data:
        # One is empty, the other is not, maximum disagreement
        return 0.0

    depth, height, width = grid_size

    # --- Data Validation ---
    # This loop ensures that all coordinates are valid before proceeding.
    for obj in [obj1_data, obj2_data]:
        for slice_data in obj:
            if 'z_position' not in slice_data or 'points' not in slice_data:
                raise ValueError("Each 3D slice must have 'z_position' and 'points'.")

            # Validate z_position
            try:
                z_pos = float(slice_data['z_position'])
            except (ValueError, TypeError):
                raise ValueError("'z_position' must be a number.")
            if not (0.0 <= z_pos <= 1.0):
                raise ValueError(f"z_position {z_pos} is outside the [0, 1] range.")

            # Validate points
            points_data = slice_data['points']
            if not isinstance(points_data, list):
                raise ValueError("'points' must be a list of coordinate pairs.")

            if not points_data:  # Allow empty points list for a slice
                continue

            try:
                pts = np.array(points_data, dtype=np.float64)
            except ValueError:
                raise ValueError("'points' must be convertible to a numpy array of floats.")

            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError("Each 'points' array must have a shape of (N, 2).")
            if np.any(pts < 0.0) or np.any(pts > 1.0):
                raise ValueError("All x, y coordinates in 'points' must be in the [0, 1] range.")

    # --- Rasterization ---
    def rasterize_to_volume(obj_data):
        vol = np.zeros((depth, height, width), dtype=bool)

        for slice_data in obj_data:
            z_pos = float(slice_data['z_position'])
            pts = np.array(slice_data['points'])

            if pts.shape[0] < 3:  # A polygon needs at least 3 points
                continue

            # Scale relative coordinates to grid dimensions
            z_idx = int(round(z_pos * (depth - 1)))
            # Y maps to rows (height), X maps to columns (width)
            r_indices = np.round(pts[:, 1] * (height - 1)).astype(int)
            c_indices = np.round(pts[:, 0] * (width - 1)).astype(int)

            # Clip to ensure within bounds (though with [0,1] validation, this is a safeguard)
            z_idx = np.clip(z_idx, 0, depth - 1)
            r_indices = np.clip(r_indices, 0, height - 1)
            c_indices = np.clip(c_indices, 0, width - 1)

            # Rasterize the polygon for the current slice
            rr, cc = polygon(r_indices, c_indices, shape=(height, width))
            vol[z_idx, rr, cc] = True

        return vol

    # Create volumes for both annotations
    vol1 = rasterize_to_volume(obj1_data)
    vol2 = rasterize_to_volume(obj2_data)

    # --- IoU Calculation ---
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()

    if union == 0:
        # This occurs if both objects are empty after rasterization (e.g., no valid polygons)
        return 0.0

    iou = intersection / union
    return iou

SIMILARITY_FUNCTIONS = {
    # Object Detection
    "bbox_iou_similarity": bbox_iou_similarity,
    "centroid_similarity": centroid_similarity,

    # Instance Segmentation
    "segm_iou_similarity": segm_iou_similarity,
    "segm_giou_similarity": segm_giou_similarity,

    # Keypoints / Pose estimation
    "in-mpjpe_similarity": image_normalized_mpjpe_similarity,

    # 3D instance segmentation
    "3D_IoU_similarity": segm_3d_iou_similarity
}
