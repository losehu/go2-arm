"""Fixed full command limits and continuous attitude expansion after full-speed competence."""
STAGE1_BUDGET = 40000
# Only kept for rejecting incompatible legacy resume files; no discrete speed schedule is used.
STAGE1_LEVELS = ((1., 3.),)
ATTITUDE_CURRICULUM = {
    'revision': 'full_speed_gate_then_continuous_attitude_v8',
    'initial_cap_deg': 3., 'final_cap_deg': 15.,
    'minimum_omni_fraction': .6, 'minimum_success_ema': .7,
    'expansion_updates': 4000,
}


def stage1_ready(range_scale, guidance_cap_deg, success_ema, full_speed_learned):
    return (range_scale == 1. and guidance_cap_deg >= 15.
            and success_ema >= .7 and full_speed_learned)


def next_attitude_cap(cap, success_ema, full_speed_learned, omni_fraction):
    cfg = ATTITUDE_CURRICULUM
    if full_speed_learned and omni_fraction >= cfg['minimum_omni_fraction'] and success_ema >= .7:
        return min(cfg['final_cap_deg'], cap + (cfg['final_cap_deg'] - cfg['initial_cap_deg']) / cfg['expansion_updates'])
    return cap
