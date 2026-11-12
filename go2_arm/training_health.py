"""Bounded stop for sustained reset loops that starve AMP of its history window."""

TRAINING_HEALTH = {
    "revision": "amp_short_episode_stop_v9",
    "warmup_updates": 100,
    "minimum_resets_per_environment_per_update": .5,
    "maximum_mean_episode_steps": 10.,
    "maximum_amp_valid_fraction": .005,
    "required_consecutive_updates": 20,
}


class TrainingHealth:
    def __init__(self):
        self.consecutive = 0

    def check(self, iteration, num_envs, resets, mean_steps, amp_valid_fraction):
        cfg = TRAINING_HEALTH
        bad = (resets >= cfg["minimum_resets_per_environment_per_update"] * num_envs and
               mean_steps <= cfg["maximum_mean_episode_steps"] and
               amp_valid_fraction <= cfg["maximum_amp_valid_fraction"])
        self.consecutive = self.consecutive + 1 if bad else 0
        return (iteration >= cfg["warmup_updates"] and
                self.consecutive >= cfg["required_consecutive_updates"])
