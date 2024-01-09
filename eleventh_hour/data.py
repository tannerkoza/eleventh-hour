import pickle
import numpy as np
import navsim as ns
import pandas as pd
import eleventh_hour.navigators.vt as vt

from pathlib import Path
from dataclasses import dataclass
from navtools.conversions import ecef2lla, ecef2enu, uvw2enu


@dataclass
class States:
    truth_lla: np.ndarray
    truth_enu_pos: np.ndarray
    truth_enu_vel: np.ndarray

    lla: np.ndarray
    enu_pos: np.ndarray
    enu_vel: np.ndarray


@dataclass
class Errors:
    pos: np.ndarray
    vel: np.ndarray
    clock_bias: np.ndarray
    clock_drift: np.ndarray


def process_truth_states(truth: ns.ReceiverTruthStates):
    lla = np.array(
        ecef2lla(
            x=truth.pos[:, 0],
            y=truth.pos[:, 1],
            z=truth.pos[:, 2],
        )
    )
    enu_pos = np.array(
        ecef2enu(
            x=truth.pos[:, 0],
            y=truth.pos[:, 1],
            z=truth.pos[:, 2],
            lat0=lla[0, 0],
            lon0=lla[1, 0],
            alt0=lla[2, 0],
        )
    )
    enu_vel = np.array(
        uvw2enu(
            u=truth.vel[:, 0],
            v=truth.vel[:, 1],
            w=truth.vel[:, 2],
            lat0=lla[0, 0],
            lon0=lla[1, 0],
        )
    )

    return lla, enu_pos, enu_vel


def process_vdfll_states(
    truth: ns.ReceiverTruthStates, rx_states: vt.VDFLLReceiverStates
):
    truth_lla, truth_enu_pos, truth_enu_vel = process_truth_states(truth=truth)

    pos = rx_states.pos
    vel = rx_states.vel
    lla = np.array(
        ecef2lla(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
        )
    )
    enu_pos = np.array(
        ecef2enu(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            lat0=truth_lla[0, 0],
            lon0=truth_lla[1, 0],
            alt0=truth_lla[2, 0],
        )
    )
    enu_vel = np.array(
        uvw2enu(
            u=vel[:, 0],
            v=vel[:, 1],
            w=vel[:, 2],
            lat0=truth_lla[0, 0],
            lon0=truth_lla[1, 0],
        )
    )

    clock_bias = rx_states.clock_bias
    clock_drift = rx_states.clock_drift

    pos_error = truth_enu_pos - enu_pos
    vel_error = truth_enu_vel - enu_vel
    clock_bias_error = truth.clock_bias - clock_bias
    clock_drift_error = truth.clock_drift - clock_drift

    states = States(
        truth_lla=truth_lla,
        truth_enu_pos=truth_enu_pos,
        truth_enu_vel=truth_enu_vel,
        lla=lla,
        enu_pos=enu_pos,
        enu_vel=enu_vel,
    )
    errors = Errors(
        pos=pos_error,
        vel=vel_error,
        clock_bias=clock_bias_error,
        clock_drift=clock_drift_error,
    )

    return states, errors


def pickle_objects(data: dict, output_path: Path):
    for name, obj in data.items():
        file_name = f"{name}.pkl"
        output_file = output_path / file_name

        with open(output_file, "+bw") as file:
            pickle.dump(obj=obj, file=file)
