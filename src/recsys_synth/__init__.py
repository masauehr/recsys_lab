"""推薦システム用の合成データ生成器（本当の好み＋偏った観測）。"""
from .generator import (  # noqa: F401
    ORACLE_COLUMNS, RELEVANT, SPLIT_TIMES, World, Dataset, make_dataset, make_world, simulate_interactions,
)
