"""
人口预测与分析工具包
"""
__version__ = "0.2.0"

from . import data_loader
from . import parameters
from . import model
from . import analysis
from . import utils

from .model import CohortComponentModel
from .analysis import (
    calculate_age_group_stats,
    calculate_dependency_ratio_custom,
    plot_population_pyramid_enhanced,
    print_detailed_stats,
    print_year_summary
)