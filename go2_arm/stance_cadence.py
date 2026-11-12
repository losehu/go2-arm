"""Heading-frame stance width and measured full-stride cadence; no phase oscillator."""
import torch

STANCE_CADENCE = {
    "revision": "support_width_measured_cadence_bounded_positive_v10b",
    "reward_application": "bounded_positive_shaping_discount_before_failure_event",
    "style_cost_discount_scale": .25,
    "minimum_positive_shaping_factor": .25,
    "minimum_straight_foot_half_width_m": .12,
    "lateral_half_width_allowance_m": .015,
    "width_shortfall_scale_m": .06,
    "width_weight_per_second": 2.,
    "cadence_upper_hz": [2., 2.5],
    "cadence_lower_hz": [1.2, 1.6],
    "yaw_equivalent_radius_m": .25,
    "cadence_period_filter_alpha": .3,
    "cadence_excess_scale_hz": .8,
    "fast_cadence_weight_per_second": 1.5,
    "slow_cadence_weight_per_second": .5,
}


def style_reward_factor(cost):
    # Preserve a positive tracking gradient even when the style is initially poor.
    # Physical failure events are charged separately, after this discount.
    return (1. / (1. + STANCE_CADENCE['style_cost_discount_scale'] * cost.clamp_min(0.))).clamp_min(
        STANCE_CADENCE['minimum_positive_shaping_factor'])


def stance_width_cost(heading_foot_y, contact, commands):
    cfg = STANCE_CADENCE
    signed_y = heading_foot_y * heading_foot_y.new_tensor([1., -1., 1., -1.])
    lateral = (commands[:, 1].abs() / .6).clamp(0., 1.)
    minimum = cfg['minimum_straight_foot_half_width_m'] - cfg['lateral_half_width_allowance_m'] * lateral
    deficit = ((minimum[:, None] - signed_y) / cfg['width_shortfall_scale_m']).clamp(0., 2.)
    cost = cfg['width_weight_per_second'] * (deficit.square() * contact).mean(-1)
    return cost, signed_y, deficit.gt(0.) & contact


class MeasuredCadence:
    def __init__(self, n, device):
        self.since = torch.zeros(n, 4, device=device)
        self.period = torch.zeros_like(self.since)
        self.seen = torch.zeros_like(self.since, dtype=torch.bool)
        self.ready = torch.zeros_like(self.seen)

    def reset(self, ids):
        self.since[ids] = 0; self.period[ids] = 0
        self.seen[ids] = False; self.ready[ids] = False

    def step(self, qualified_touchdown, commands, moving, dt):
        cfg = STANCE_CADENCE
        active = moving[:, None]
        elapsed = (self.since + dt).clamp_max(2.)
        event = qualified_touchdown & active
        cycle = event & self.seen
        measured = elapsed.clamp_min(.04)
        estimate = torch.where(self.ready,
            self.period.lerp(measured, cfg['cadence_period_filter_alpha']), measured)
        self.period.copy_(torch.where(active, torch.where(cycle, estimate, self.period), 0.))
        self.ready.copy_(active & (self.ready | cycle))
        self.seen.copy_(active & (self.seen | event))
        self.since.copy_(torch.where(active & ~event, elapsed, 0.))
        # Elapsed time also detects a foot that stops contributing after learning a rate.
        effective_period = torch.maximum(self.period, self.since).clamp_min(.04)
        hz = torch.where(self.ready, effective_period.reciprocal(), 0.)
        demand = torch.cat((commands[:, :2], cfg['yaw_equivalent_radius_m'] * commands[:, 2:3]), -1).norm(dim=-1).clamp(0., 1.)
        upper = cfg['cadence_upper_hz'][0] + (cfg['cadence_upper_hz'][1] - cfg['cadence_upper_hz'][0]) * demand
        lower = cfg['cadence_lower_hz'][0] + (cfg['cadence_lower_hz'][1] - cfg['cadence_lower_hz'][0]) * demand
        high = ((hz - upper[:, None]) / cfg['cadence_excess_scale_hz']).clamp(0., 2.)
        low = ((lower[:, None] - hz) / cfg['cadence_excess_scale_hz']).clamp(0., 2.)
        cost = ((cfg['fast_cadence_weight_per_second'] * high.square() +
                 cfg['slow_cadence_weight_per_second'] * low.square()) * self.ready).mean(-1)
        return cost, hz, self.ready, high.gt(0.) & self.ready
