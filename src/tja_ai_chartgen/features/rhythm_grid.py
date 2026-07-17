from tja_ai_chartgen.tja.model import BarFeature, ChartBar

AI_ARBITRARY_GRID_MIN_HITS = 16
AI_ARBITRARY_GRID_MIN_COUNT = 4
AI_ARBITRARY_GRID_MAX_RATE = 0.10


def is_stable_rhythmic_position(grid: int) -> bool:
    """判断 canonical 位置是否属于稳定直拍或三连音子网格。"""
    return grid >= 0 and (grid % 2 == 0 or grid % 3 == 0)


def is_stable_rhythmic_grid(grid: int, canonical_grids_per_bar: int) -> bool:
    """判断格点是否属于项目支持的稳定直拍或三连音子网格。"""
    if grid < 0 or grid >= canonical_grids_per_bar:
        return False
    if canonical_grids_per_bar % 12:
        return True
    return is_stable_rhythmic_position(grid)


def chart_arbitrary_grid_positions(
    chart_bars: list[ChartBar],
    feature_bars: list[BarFeature],
) -> tuple[int, list[tuple[int, int, int]]]:
    """返回普通击打总数及不属于稳定子网格的位置。"""
    evaluated = 0
    arbitrary: list[tuple[int, int, int]] = []
    for position, (chart_bar, feature_bar) in enumerate(
        zip(chart_bars, feature_bars, strict=False)
    ):
        if not chart_bar.notes or feature_bar.grids_per_bar <= 0:
            continue
        for grid, note in enumerate(chart_bar.notes):
            if note not in "1234":
                continue
            evaluated += 1
            canonical_tick = min(
                feature_bar.grids_per_bar - 1,
                round(grid / len(chart_bar.notes) * feature_bar.grids_per_bar),
            )
            if not is_stable_rhythmic_grid(
                canonical_tick,
                feature_bar.grids_per_bar,
            ):
                arbitrary.append((position, grid, canonical_tick))
    return evaluated, arbitrary
