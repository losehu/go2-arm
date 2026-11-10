"""Estimate command-target coverage of the installed reference bank."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1] / "go2_arm")
parser.add_argument("--samples", type=int, default=200_000)
args = parser.parse_args()
sys.path.insert(0, str(args.package.resolve()))
from base_commands import apply_velocity_deadzone
from directional_amp import DirectionalAMP, command_signature, guide_mixed_targets, matched_references


def targets(n, cap, rng):
    low = .5*cap
    low[1] = .12
    axis = torch.randint(0, 3, (n,), generator=rng)
    command = torch.zeros(n, 3)
    sign = 2*(torch.rand(n, generator=rng) > .5).float()-1
    command[torch.arange(n), axis] = sign*(low[axis]+(cap[axis]-low[axis])*torch.rand(n, generator=rng))
    mixed = torch.rand(n, generator=rng) < .25
    command[mixed] += (2*torch.rand((int(mixed.sum()), 3), generator=rng)-1)*cap*.25
    command = command.clamp(-cap, cap)
    command[torch.rand(n, generator=rng) < .10] = 0.
    return apply_velocity_deadzone(command)


def main():
    if args.samples < 1:
        parser.error("--samples must be positive")
    torch.set_num_threads(2)
    amp = DirectionalAMP(args.package / "amp_reference", SimpleNamespace(device="cpu", num_envs=1, step_dt=.02))
    rng = torch.Generator().manual_seed(20261110)
    report = {"clips": len(amp.metadata["clips"]), "signatures": len(amp.clip_signatures.unique(dim=0)),
              "speed_distance": amp.metadata["max_normalized_speed_distance"], "courses": []}
    for scale in (.50, .75, 1.):
        command = targets(args.samples, scale*amp.command_scale, rng) / amp.command_scale
        grouped = []
        for batch in command.split(4096):
            direction = command_signature(batch)
            support = (direction[:, None] == amp.clip_signatures).all(dim=-1).any(dim=-1)
            matches, _ = matched_references(batch, amp.clip_commands, amp.clip_signatures)
            guided = guide_mixed_targets(batch, batch.new_full((3,), scale), amp.clip_commands, amp.clip_signatures, rng)
            guided_matches, _ = matched_references(guided, amp.clip_commands, amp.clip_signatures)
            grouped.append(torch.stack((direction.count_nonzero(dim=1)>1, support, matches.any(dim=-1),
                                        guided_matches.any(dim=-1)), dim=-1))
        mixed, sign_match, speed_match, guided = torch.cat(grouped).unbind(-1)
        report["courses"].append({"scale": scale, "samples": args.samples, "mixed_samples": int(mixed.sum()),
            "direction_coverage": float(sign_match.float().mean()),
            "mixed_direction_coverage": float(sign_match[mixed].float().mean()),
            "speed_and_direction_coverage": float(speed_match.float().mean()),
            "mixed_speed_and_direction_coverage": float(speed_match[mixed].float().mean()),
            "guided_speed_and_direction_coverage": float(guided.float().mean()),
            "guided_mixed_speed_and_direction_coverage": float(guided[mixed].float().mean())})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
