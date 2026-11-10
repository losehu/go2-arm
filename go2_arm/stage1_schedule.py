"""Separate speed and internal attitude expansion; final requirements stay intact."""
STAGE1_LEVELS = ((.5, 3.), (.75, 3.), (.75, 8.), (1., 8.), (1., 11.5), (1., 15.))
STAGE1_BUDGET = 40000
MIN_DWELL = 500


def stage1_ready(level, range_scale, guidance_cap_deg, success_ema):
    return (level == len(STAGE1_LEVELS) - 1 and range_scale == 1.
            and guidance_cap_deg == 15. and success_ema >= .7)


def next_stage1_level(level, success_ema, iteration, last_advance):
    if (level < len(STAGE1_LEVELS) - 1 and success_ema >= .7
            and iteration - last_advance >= MIN_DWELL):
        return level + 1
    return level
