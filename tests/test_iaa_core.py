import pytest
import numpy as np
from kalos.iaa.core import calculate_iaa

@pytest.fixture
def base_categories():
    return {1: "cat", 2: "dog"}

def create_mock_ann(ann_id, rater_id, cat_id, bbox):
    """Helper to create a normalized COCO-style annotation."""
    return {
        'id': ann_id,
        'rater_id': rater_id,
        'category_id': cat_id,
        'bbox': bbox,
        'image_id': 1
    }

def test_iaa_identity(base_categories):
    """Test that identical annotations result in Alpha = 1.0."""
    processed_data = {
        1: {
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(1, 'A', 1, [0.1, 0.1, 0.2, 0.2])],
                'B': [create_mock_ann(2, 'B', 1, [0.1, 0.1, 0.2, 0.2])]
            }
        }
    }
    
    mean_alpha, global_alpha, _, _, _, _, _, _, _ = calculate_iaa(
        processed_data, base_categories, 
        method='greedy', threshold_func='bbox_iou_similarity', 
        cost_func='negative_score', similarity_threshold=0.5,
        calculate_vitality=False, calculate_difficulty=False, collaboration_clusters=False,
        all_raters=['A', 'B']
    )
    assert mean_alpha == 1.0
    assert global_alpha == 1.0

def test_iaa_category_disagreement(base_categories):
    """Test that same geometry but different classes reduces Alpha."""
    processed_data = {
        1: {
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(1, 'A', 1, [0.1, 0.1, 0.2, 0.2])], # Cat
                'B': [create_mock_ann(2, 'B', 2, [0.1, 0.1, 0.2, 0.2])]  # Dog
            }
        },
        2: { 
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(3, 'A', 1, [0.5, 0.5, 0.1, 0.1])],
                'B': [create_mock_ann(4, 'B', 1, [0.5, 0.5, 0.1, 0.1])]
            }
        }
    }
    
    mean_alpha, global_alpha, _, _, _, _, _, _, _ = calculate_iaa(
        processed_data, base_categories, 
        method='greedy', threshold_func='bbox_iou_similarity', 
        cost_func='negative_score', similarity_threshold=0.5,
        calculate_vitality=False, calculate_difficulty=False, collaboration_clusters=False,
        all_raters=['A', 'B']
    )
    assert mean_alpha < 1.0
    assert global_alpha < 1.0

def test_iaa_null_agreement(base_categories):
    """Test that zero overlap results in low Alpha."""
    processed_data = {
        1: {
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(1, 'A', 1, [0.0, 0.0, 0.1, 0.1])],
                'B': [create_mock_ann(2, 'B', 1, [0.8, 0.8, 0.1, 0.1])] 
            }
        }
    }
    
    mean_alpha, global_alpha, _, _, _, _, _, _, _ = calculate_iaa(
        processed_data, base_categories, 
        method='greedy', threshold_func='bbox_iou_similarity', 
        cost_func='negative_score', similarity_threshold=0.5,
        calculate_vitality=False, calculate_difficulty=False, collaboration_clusters=False,
        all_raters=['A', 'B']
    )
    assert mean_alpha <= 0.0
    assert global_alpha <= 0.0

def test_iaa_empty_images(base_categories):
    """Test that agreement on empty images results in Alpha = 1.0."""
    processed_data = {
        1: {
            'rater_list': ['A', 'B'],
            'annotations_by_rater': { 'A': [], 'B': [] }
        }
    }
    
    mean_alpha, global_alpha, _, _, _, _, _, _, _ = calculate_iaa(
        processed_data, base_categories, 
        method='greedy', threshold_func='bbox_iou_similarity', 
        cost_func='negative_score', similarity_threshold=0.5,
        calculate_vitality=False, calculate_difficulty=False, collaboration_clusters=False,
        all_raters=['A', 'B']
    )
    assert mean_alpha == 1.0
    assert global_alpha == 1.0

def test_iaa_permutation_invariance(base_categories):
    """Test that swapping raters doesn't change the Alpha score."""
    data_orig = {
        1: {
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(1, 'A', 1, [0.1, 0.1, 0.2, 0.2])],
                'B': [create_mock_ann(2, 'B', 1, [0.12, 0.12, 0.2, 0.2])]
            }
        }
    }
    
    data_swapped = {
        1: {
            'rater_list': ['B', 'A'],
            'annotations_by_rater': {
                'B': [create_mock_ann(2, 'B', 1, [0.12, 0.12, 0.2, 0.2])],
                'A': [create_mock_ann(1, 'A', 1, [0.1, 0.1, 0.2, 0.2])]
            }
        }
    }
    
    params = {
        'categories': base_categories, 'method': 'greedy', 
        'threshold_func': 'bbox_iou_similarity', 'cost_func': 'negative_score', 
        'similarity_threshold': 0.5, 'calculate_vitality': False, 
        'calculate_difficulty': False, 'collaboration_clusters': False,
        'all_raters': ['A', 'B']
    }
    
    mean_orig, global_orig, _, _, _, _, _, _, _ = calculate_iaa(data_orig, **params)
    mean_swapped, global_swapped, _, _, _, _, _, _, _ = calculate_iaa(data_swapped, **params)
    
    assert mean_orig == pytest.approx(mean_swapped)
    assert global_orig == pytest.approx(global_swapped)

def test_iaa_global_sparsity(base_categories):
    """Test global alpha with sparse rater assignments (Rater C only on image 2)."""
    processed_data = {
        1: { # Image 1: A and B agree
            'rater_list': ['A', 'B'],
            'annotations_by_rater': {
                'A': [create_mock_ann(1, 'A', 1, [0.1, 0.1, 0.2, 0.2])],
                'B': [create_mock_ann(2, 'B', 1, [0.1, 0.1, 0.2, 0.2])]
            }
        },
        2: { # Image 2: Only C is assigned, finds one object
            'rater_list': ['C'],
            'annotations_by_rater': {
                'C': [create_mock_ann(3, 'C', 1, [0.5, 0.5, 0.1, 0.1])]
            }
        }
    }
    
    mean_alpha, global_alpha, _, _, _, _, _, _, _ = calculate_iaa(
        processed_data, base_categories, 
        method='greedy', threshold_func='bbox_iou_similarity', 
        cost_func='negative_score', similarity_threshold=0.5,
        calculate_vitality=False, calculate_difficulty=False, collaboration_clusters=False,
        all_raters=['A', 'B', 'C']
    )
    
    assert mean_alpha == 1.0
    assert global_alpha == 1.0
