from src.attacks.data_source import DataSource

from src.attacks.features_extraction.extractor import FeatureExtractor
from src.attacks.features_extraction.carlini import CarliniLossThresholdExtractor
from src.attacks.features_extraction.secmi import SecMIExtractor
from src.attacks.features_extraction.combination_attack import (
    CombinationAttackExtractor,
)

from src.attacks.features_extraction.pia import PIAExtractor
from src.attacks.features_extraction.gradient_masking import GradientMaskingExtractor
from src.attacks.features_extraction.multiple_loss import MultipleLossExtractor
from src.attacks.features_extraction.noise_optim import NoiseOptimExtractor
from src.attacks.features_extraction.pandora import PandoraExtractor
from src.attacks.features_extraction.clid import CLiDExtractor
from src.attacks.features_extraction.image_paths import ImagePathsExtractor

from src.attacks.scores_computation.computer import ScoreComputer
from src.attacks.scores_computation.carlini import CarliniLossThresholdComputer
from src.attacks.scores_computation.secmi import SecMIStat
from src.attacks.scores_computation.combination_attack import CombinationAttackComputer

from src.attacks.scores_computation.pia import PIAComputer, PIANComputer
from src.attacks.scores_computation.gradient_masking import GradientMaskingComputer
from src.attacks.scores_computation.multiple_loss import MultipleLossComputer
from src.attacks.scores_computation.noise_optim import NoiseOptimComputer
from src.attacks.scores_computation.pandora import PandoraComputer
from src.attacks.scores_computation.clid import CLiDComputer


from typing import Dict


feature_extractors: Dict[str, FeatureExtractor] = {
    "carlini_lt": CarliniLossThresholdExtractor,
    "secmi_nn": SecMIExtractor,
    "secmi_stat": SecMIExtractor,
    "combination_attack": CombinationAttackExtractor,
    "combination_attack_no_gm": CombinationAttackExtractor,
    "combination_attack_lite": CombinationAttackExtractor,
    "pia": PIAExtractor,
    "pian": PIAExtractor,
    "gradient_masking": GradientMaskingExtractor,
    "multiple_loss": MultipleLossExtractor,
    "noise_optim": NoiseOptimExtractor,
    "pandora": PandoraExtractor,
    "clid": CLiDExtractor,
    "image_paths": ImagePathsExtractor,
}

score_computers: Dict[str, ScoreComputer] = {
    "carlini_lt": CarliniLossThresholdComputer,
    "secmi_stat": SecMIStat,
    "combination_attack": CombinationAttackComputer,
    "pia": PIAComputer,
    "pian": PIANComputer,
    "gradient_masking": GradientMaskingComputer,
    "multiple_loss": MultipleLossComputer,
    "noise_optim": NoiseOptimComputer,
    "pandora": PandoraComputer,
    "clid": CLiDComputer,
}

from src.attacks.utils import (
    load_data,
    get_datasets_clf,
    load_members_nonmembers_scores,
)
