#!/usr/bin/env python3
"""Serve a G1 + Dex3 pi0.5 checkpoint over openpi's websocket protocol.

Run it with external/openpi on the path (it provides g1_dex3_policy and the
pi05_g1_* train configs):

    uv run --project external/openpi python scripts/pi05/serve_pi05.py \
        --checkpoint-dir checkpoints/pi05_g1_fruit_v2/10000

The checkpoints are trained with DeltaActions(make_bool_mask(7, -7, 7, -7)):

    arms  (dims 0-6, 14-20)  relative: target - current joint angle
    hands (dims 7-13, 21-27) absolute joint angle

--arm-actions delta (default) serves the arm deltas as the model outputs them;
the client adds the state once (run_pi05_policy.py --arm-action-mode delta, its
default). --arm-actions absolute keeps openpi's AbsoluteActions, and the client
must then use --arm-action-mode absolute. Never both: that adds the state twice
and freezes the arms.
"""

import dataclasses
import logging
import pathlib
from typing import Literal

import tyro

from openpi import transforms
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    config: str = "pi05_g1_fruit_v2"
    port: int = 8017
    # "delta": serve the model's own output (relative arms). "absolute": add the state on the server.
    arm_actions: Literal["delta", "absolute"] = "delta"


def main(args: Args) -> None:
    ckpt = args.checkpoint_dir.resolve()
    for need in ("params", "assets"):
        if not (ckpt / need).is_dir():
            raise FileNotFoundError(f"{ckpt / need} missing -- not an openpi checkpoint dir")

    cfg = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, ckpt)

    # Fail loudly if this config was not trained relative-arms / absolute-hands.
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    deltas = [t for t in dc.data_transforms.inputs if isinstance(t, transforms.DeltaActions)]
    want = tuple(transforms.make_bool_mask(7, -7, 7, -7))
    if len(deltas) != 1 or tuple(deltas[0].mask) != want:
        raise SystemExit(f"{args.config} was not trained with DeltaActions{want}: {deltas}")

    if args.arm_actions == "delta":
        kept = [t for t in policy._output_transform.transforms if not isinstance(t, transforms.AbsoluteActions)]
        assert len(kept) == len(policy._output_transform.transforms) - 1, "AbsoluteActions not found"
        policy._output_transform = transforms.compose(kept)
        client_flag = "--arm-action-mode delta --hand-action-mode absolute"
    else:
        client_flag = "--arm-action-mode absolute --hand-action-mode absolute"
    logging.info("ARMS: %s on the wire, HANDS: absolute. Client must use: %s",
                 "RELATIVE (delta)" if args.arm_actions == "delta" else "ABSOLUTE", client_flag)
    logging.info("Client keys: --state-key observation/state --image-key observation/ego_view "
                 "--left-wrist-image-key observation/ego_left --right-wrist-image-key observation/ego_right")
    logging.info("serving %s (%s) on 0.0.0.0:%d", ckpt, args.config, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
