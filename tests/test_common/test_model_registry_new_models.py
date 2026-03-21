import yaml

import neuralforecast.auto as nf_auto
import neuralforecast.models as nf_models
from neuralforecast.core import MODEL_FILENAME_DICT
from residual.models import MODEL_CLASSES, supports_auto_mode


def test_new_model_support_matrix_is_explicit():
    search_space = yaml.safe_load(open('search_space.yaml', encoding='utf-8'))

    supported = {
        'DeformTime': {
            'pkg': hasattr(nf_models, 'DeformTime'),
            'auto': hasattr(nf_auto, 'AutoDeformTime'),
            'runtime': 'DeformTime' in MODEL_CLASSES,
            'supports_auto': supports_auto_mode('DeformTime'),
            'search': 'DeformTime' in search_space['models'],
            'filename_alias': 'deformtime' in MODEL_FILENAME_DICT,
        },
        'DeformableTST': {
            'pkg': hasattr(nf_models, 'DeformableTST'),
            'auto': hasattr(nf_auto, 'AutoDeformableTST'),
            'runtime': 'DeformableTST' in MODEL_CLASSES,
            'supports_auto': supports_auto_mode('DeformableTST'),
            'search': 'DeformableTST' in search_space['models'],
            'filename_alias': 'deformabletst' in MODEL_FILENAME_DICT,
        },
        'ModernTCN': {
            'pkg': hasattr(nf_models, 'ModernTCN'),
            'auto': hasattr(nf_auto, 'AutoModernTCN'),
            'runtime': 'ModernTCN' in MODEL_CLASSES,
            'supports_auto': supports_auto_mode('ModernTCN'),
            'search': 'ModernTCN' in search_space['models'],
            'filename_alias': 'moderntcn' in MODEL_FILENAME_DICT,
        },
    }

    for surface in supported.values():
        assert surface['pkg'] is True
        assert surface['runtime'] is True
        assert surface['supports_auto'] is True
        assert surface['search'] is True
        assert surface['filename_alias'] is True
        assert surface['auto'] is False


def test_excluded_or_covered_requested_models_are_explicit():
    search_space = yaml.safe_load(open('search_space.yaml', encoding='utf-8'))

    assert not hasattr(nf_models, 'LightGTS')
    assert 'LightGTS' not in MODEL_CLASSES
    assert 'LightGTS' not in search_space['models']
    assert 'lightgts' not in MODEL_FILENAME_DICT

    # TimeMixer++ is intentionally treated as the existing TimeMixer surface,
    # not as a separate model class in this repo.
    assert hasattr(nf_models, 'TimeMixer')
    assert 'TimeMixer' in MODEL_CLASSES
    assert 'TimeMixer' in search_space['models']
    assert 'timemixer' in MODEL_FILENAME_DICT

    for alias in ('TimeMixerPP', 'TimeMixerPlusPlus'):
        assert not hasattr(nf_models, alias)
        assert alias not in MODEL_CLASSES
        assert alias not in search_space['models']
        assert alias.lower() not in MODEL_FILENAME_DICT
