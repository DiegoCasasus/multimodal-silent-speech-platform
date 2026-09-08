from __future__ import annotations

import argparse
import math
import random
import signal
import threading
import time
from typing import List

from pylsl import StreamInfo, StreamOutlet


STOP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--eeg-name", default="eeg_32_speech_eeg")
    parser.add_argument("--eeg-type", default="EEG")
    parser.add_argument("--eeg-channels", type=int, default=32)
    parser.add_argument("--eeg-srate", type=float, default=256.0)

    parser.add_argument("--emg-name", default="emg_5_speech_emg")
    parser.add_argument("--emg-type", default="EMG")
    parser.add_argument("--emg-channels", type=int, default=5)
    parser.add_argument("--emg-srate", type=float, default=256.0)

    parser.add_argument("--imu-name-prefix", default="emg-imu_daux1")
    parser.add_argument("--imu-type", default="IMU")
    parser.add_argument("--imu-streams", type=int, default=4)
    parser.add_argument("--imu-channels-per-stream", type=int, default=9)
    parser.add_argument("--imu-srate", type=float, default=32.0)

    return parser.parse_args()


def handle_stop(signum, frame):
    global STOP
    STOP = True


def make_eeg_sample(t: float, n_channels: int) -> List[float]:
    sample = []
    base_freqs = [8.0, 10.0, 12.0, 15.0]

    for ch in range(n_channels):
        f = base_freqs[ch % len(base_freqs)] + 0.05 * ch
        v = (
            20.0 * math.sin(2.0 * math.pi * f * t)
            + 8.0 * math.sin(2.0 * math.pi * 0.5 * t + 0.1 * ch)
            + random.gauss(0.0, 2.0)
        )
        sample.append(float(v))

    return sample


def make_emg_sample(t: float, n_channels: int) -> List[float]:
    sample = []

    for ch in range(n_channels):
        # Fake EMG-like high-frequency activity with slow envelope modulation.
        envelope = 1.0 + 0.5 * math.sin(2.0 * math.pi * 0.8 * t + 0.4 * ch)
        carrier = (
            25.0 * math.sin(2.0 * math.pi * (35.0 + 3.0 * ch) * t)
            + 10.0 * math.sin(2.0 * math.pi * (70.0 + 2.0 * ch) * t)
        )
        v = envelope * carrier + random.gauss(0.0, 5.0)
        sample.append(float(v))

    return sample


def make_imu_sample(t: float, n_channels: int, stream_idx: int) -> List[float]:
    sample = []

    for ch in range(n_channels):
        phase = 0.7 * stream_idx + 0.2 * ch
        v = (
            0.4 * math.sin(2.0 * math.pi * 1.2 * t + phase)
            + 0.2 * math.sin(2.0 * math.pi * 0.3 * t + 0.3 * ch)
            + random.gauss(0.0, 0.03)
        )
        sample.append(float(v))

    return sample


class PeriodicOutletThread(threading.Thread):
    def __init__(self, outlet: StreamOutlet, fs: float, make_sample_fn, name: str):
        super().__init__(daemon=True, name=name)
        self.outlet = outlet
        self.fs = fs
        self.period = 1.0 / fs
        self.make_sample_fn = make_sample_fn

    def run(self):
        next_t = time.perf_counter()
        t0 = time.perf_counter()

        while not STOP:
            now = time.perf_counter()

            if now >= next_t:
                rel_t = now - t0
                self.outlet.push_sample(self.make_sample_fn(rel_t))
                next_t += self.period
            else:
                time.sleep(min(0.001, next_t - now))


def main() -> int:
    args = parse_args()

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    eeg_info = StreamInfo(
        name=args.eeg_name,
        type=args.eeg_type,
        channel_count=args.eeg_channels,
        nominal_srate=args.eeg_srate,
        channel_format="float32",
        source_id="fake_eeg_source",
    )
    eeg_outlet = StreamOutlet(eeg_info)

    emg_info = StreamInfo(
        name=args.emg_name,
        type=args.emg_type,
        channel_count=args.emg_channels,
        nominal_srate=args.emg_srate,
        channel_format="float32",
        source_id="fake_emg_source",
    )
    emg_outlet = StreamOutlet(emg_info)

    imu_outlets = []
    for i in range(args.imu_streams):
        info = StreamInfo(
            name=f"{args.imu_name_prefix}_{i + 1}",
            type=args.imu_type,
            channel_count=args.imu_channels_per_stream,
            nominal_srate=args.imu_srate,
            channel_format="float32",
            source_id=f"fake_imu_source_{i + 1}",
        )
        imu_outlets.append(StreamOutlet(info))

    threads = []

    threads.append(
        PeriodicOutletThread(
            outlet=eeg_outlet,
            fs=args.eeg_srate,
            make_sample_fn=lambda t: make_eeg_sample(t, args.eeg_channels),
            name="fake_eeg_thread",
        )
    )

    threads.append(
        PeriodicOutletThread(
            outlet=emg_outlet,
            fs=args.emg_srate,
            make_sample_fn=lambda t: make_emg_sample(t, args.emg_channels),
            name="fake_emg_thread",
        )
    )

    for i, outlet in enumerate(imu_outlets):
        threads.append(
            PeriodicOutletThread(
                outlet=outlet,
                fs=args.imu_srate,
                make_sample_fn=lambda t, idx=i: make_imu_sample(
                    t, args.imu_channels_per_stream, idx
                ),
                name=f"fake_imu_thread_{i + 1}",
            )
        )

    print(
        f"Started fake EEG stream: {args.eeg_name} "
        f"({args.eeg_channels} ch @ {args.eeg_srate} Hz, type={args.eeg_type})"
    )
    print(
        f"Started fake EMG stream: {args.emg_name} "
        f"({args.emg_channels} ch @ {args.emg_srate} Hz, type={args.emg_type})"
    )
    print(
        f"Started {args.imu_streams} fake IMU stream(s): "
        f"{args.imu_name_prefix}_1..{args.imu_name_prefix}_{args.imu_streams} "
        f"({args.imu_channels_per_stream} ch each @ {args.imu_srate} Hz, type={args.imu_type})"
    )
    print("Press Ctrl+C to stop.")

    for th in threads:
        th.start()

    try:
        while not STOP:
            time.sleep(0.5)
    finally:
        print("Stopping fake LSL streams...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())