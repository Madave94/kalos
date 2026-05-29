
import pytest
from pathlib import Path
from kalos.utils.data_loading import load_and_preprocess_data

def test_yolo_native_loading():
    dataset_path = Path("tests/test_yolo_to_kalos_json_sample/")
    
    # Test unified loader
    data_package = load_and_preprocess_data(dataset_path, "yolo")
    
    processed_data = data_package["processed_data"]
    all_raters = data_package["all_raters"]
    categories = data_package["categories"]
    
    # 1. Check Raters
    # Folders are labeler1, labeler2, labeler3
    assert "labeler1" in all_raters
    assert "labeler2" in all_raters
    assert "labeler3" in all_raters
    assert len(all_raters) == 3
    
    # 2. Check Categories (from data.yaml)
    # Assuming data.yaml has some names
    assert len(categories) > 0
    
    # 3. Check Images
    # Sample files: 1275, 1276, 1278, 1279
    assert len(processed_data) == 4
    
    # 4. Check Annotations
    for img_id, img_data in processed_data.items():
        assert "rater_list" in img_data
        assert len(img_data["rater_list"]) > 0
        # Verify that annotations are correctly attached to raters
        for rater in img_data["rater_list"]:
            assert rater in img_data["annotations_by_rater"]
            anns = img_data["annotations_by_rater"][rater]
            for ann in anns:
                assert ann["rater_id"] == rater
                # Check coordinate format [x_min, y_min, w, h]
                assert len(ann["bbox"]) == 4

if __name__ == "__main__":
    pytest.main([__file__])
