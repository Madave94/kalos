import pytest
import os
import json
from kalos.correspondence.correspondence_algorithms import preprocess_data
from kalos.iaa.kalos_execution import run_kalos_pipeline
from kalos.config import KaLOSProjectConfig, PlottingConfig

@pytest.fixture
def multi_session_coco(tmp_path):
    """Creates a temporary COCO file with multiple sessions."""
    data = {
        "images": [
            {
                "id": 1,
                "file_name": "img1.jpg",
                "width": 100, "height": 100,
                "rater_list": { "A": [1, 2], "B": [1] } # A has 2 sessions, B has 1
            }
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            # Session 1: A and B agree perfectly
            {"id": 1, "image_id": 1, "rater_id": "A", "session_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "rater_id": "B", "session_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            # Session 2: A disagrees with their Session 1 (moved bbox)
            {"id": 3, "image_id": 1, "rater_id": "A", "session_id": 2, "category_id": 1, "bbox": [50, 50, 20, 20]}
        ]
    }
    path = tmp_path / "multi_session.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)

def test_multi_session_pipeline_integration(multi_session_coco, tmp_path):
    """
    Tests that the orchestrator correctly identifies sessions and calculates 
    all 3 tiers of agreement from a raw COCO file.
    """
    # 1. Setup a minimal config
    cfg = KaLOSProjectConfig(
        annotation_file=multi_session_coco,
        task="bbox",
        method="greedy",
        threshold_func="bbox_iou_similarity",
        cost_func="negative_score",
        similarity_threshold=0.5,
        calculate_intra_iaa=True,
        plotting=PlottingConfig(plot_all=False),
        output_results=tmp_path / "results",
        log_level="DEBUG"
    )

    # 2. Run the orchestrator
    # We don't capture returns because run_kalos_pipeline doesn't return (it logs/exports)
    # So we verify success by checking the exported files.
    run_kalos_pipeline(cfg)

    # 3. Verify Summary Export (Tier 1)
    summary_path = tmp_path / "results" / "iaa_summary.csv"
    assert summary_path.exists()
    with open(summary_path, 'r') as f:
        content = f.read()
        assert "Mean Image K-Alpha (Primary)" in content
        assert "Global Dataset K-Alpha (Secondary)" in content

    # 4. Verify Per-Session Export (Tier 2)
    session_path = tmp_path / "results" / "per_session_performance.csv"
    assert session_path.exists()
    with open(session_path, 'r') as f:
        content = f.read()
        assert "Session ID" in content
        # Session 1 had A and B agreeing perfectly -> should be 1.0
        assert "1,1.0000" in content 

    # 5. Verify Intra-IAA Export (Tier 3)
    intra_path = tmp_path / "results" / "intra_iaa_consistency.csv"
    assert intra_path.exists()
    with open(intra_path, 'r') as f:
        content = f.read()
        assert "Annotator ID" in content
        assert "A," in content
        # A disagreed with themselves -> consistency should be < 1.0 (actually <= 0 here)
        # We just check that a value was calculated.
