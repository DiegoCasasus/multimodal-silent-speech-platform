#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyxdf


CAL_PATTERN = re.compile(r"^CAL_STIM_ON:(.+):(T|NT)$")


def find_marker_stream(streams, preferred_name=None):
    if preferred_name is not None:
        for s in streams:
            if s["info"]["name"][0] == preferred_name:
                return s
    for s in streams:
        if s["info"]["type"][0].lower() == "markers":
            return s
    raise RuntimeError("No marker stream found.")


def find_signal_stream(streams, preferred_name=None, preferred_type=None):
    if preferred_name is not None:
        for s in streams:
            if s["info"]["name"][0] == preferred_name:
                return s

    if preferred_type is not None:
        for s in streams:
            if s["info"]["type"][0].upper() == preferred_type.upper():
                return s

    # Fallback preference order
    for wanted_type in ["EEG", "EXG"]:
        for s in streams:
            if s["info"]["type"][0].upper() == wanted_type:
                return s

    raise RuntimeError("No suitable EEG/EXG stream found.")


def extract_calibration_events(marker_stream):
    raw_markers = marker_stream["time_series"]
    marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float)

    events = []
    for m, t in zip(raw_markers, marker_ts):
        if isinstance(m, (list, np.ndarray)):
            label = str(m[0])
        else:
            label = str(m)

        match = CAL_PATTERN.match(label)
        if match:
            stim = match.group(1)
            tag = match.group(2)
            events.append(
                {
                    "time": t,
                    "stim": stim,
                    "tag": tag,
                    "label": label,
                }
            )
    return events


def extract_epochs(data, ts, events, fs, tmin=-0.1, tmax=0.6):
    data = np.asarray(data, dtype=float)
    ts = np.asarray(ts, dtype=float)

    n_times = int(round((tmax - tmin) * fs)) + 1
    rel_times = np.arange(n_times) / fs + tmin
    baseline_mask = rel_times < 0

    epochs = []
    kept_events = []

    for ev in events:
        start_time = ev["time"] + tmin
        start_idx = np.searchsorted(ts, start_time, side="left")
        end_idx = start_idx + n_times

        if start_idx < 0 or end_idx > len(ts):
            continue

        epoch = data[start_idx:end_idx, :].T  # shape: channels x time

        if epoch.shape[1] != n_times:
            continue

        # Baseline correction
        if np.any(baseline_mask):
            baseline = epoch[:, baseline_mask].mean(axis=1, keepdims=True)
            epoch = epoch - baseline

        epochs.append(epoch)
        kept_events.append(ev)

    if len(epochs) == 0:
        raise RuntimeError("No valid calibration epochs could be extracted.")

    return np.stack(epochs, axis=0), kept_events, rel_times


def build_template(target_epochs, nontarget_epochs):
    target_avg = target_epochs.mean(axis=0)       # channels x time
    nontarget_avg = nontarget_epochs.mean(axis=0) # channels x time

    template_vector = np.concatenate(
        [target_avg.ravel(), nontarget_avg.ravel()]
    )

    return template_vector, target_avg, nontarget_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdf", required=True, help="Path to .xdf file")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--signal-stream-name", default=None, help="Exact EEG/EXG stream name")
    parser.add_argument("--signal-stream-type", default=None, help="Preferred stream type, e.g. EEG or EXG")
    parser.add_argument("--marker-stream-name", default=None, help="Exact marker stream name")
    parser.add_argument("--tmin", type=float, default=-0.1, help="Epoch start in seconds")
    parser.add_argument("--tmax", type=float, default=0.6, help="Epoch end in seconds")
    args = parser.parse_args()

    xdf_path = Path(args.xdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    streams, header = pyxdf.load_xdf(str(xdf_path))

    marker_stream = find_marker_stream(streams, preferred_name=args.marker_stream_name)
    signal_stream = find_signal_stream(
        streams,
        preferred_name=args.signal_stream_name,
        preferred_type=args.signal_stream_type,
    )

    signal_name = signal_stream["info"]["name"][0]
    signal_type = signal_stream["info"]["type"][0]
    fs = float(signal_stream["info"]["nominal_srate"][0])

    data = np.asarray(signal_stream["time_series"], dtype=float)
    ts = np.asarray(signal_stream["time_stamps"], dtype=float)

    events = extract_calibration_events(marker_stream)

    if len(events) == 0:
        raise RuntimeError("No CAL_STIM_ON markers found in marker stream.")

    epochs, kept_events, rel_times = extract_epochs(
        data=data,
        ts=ts,
        events=events,
        fs=fs,
        tmin=args.tmin,
        tmax=args.tmax,
    )

    tags = np.array([ev["tag"] for ev in kept_events])
    target_mask = tags == "T"
    nontarget_mask = tags == "NT"

    if target_mask.sum() == 0 or nontarget_mask.sum() == 0:
        raise RuntimeError("Need at least one target and one non-target epoch.")

    target_epochs = epochs[target_mask]
    nontarget_epochs = epochs[nontarget_mask]

    template_vector, target_avg, nontarget_avg = build_template(
        target_epochs, nontarget_epochs
    )

    np.savez(
        out_path,
        template_vector=template_vector,
        target_avg=target_avg,
        nontarget_avg=nontarget_avg,
        rel_times=rel_times,
        fs=fs,
        signal_name=signal_name,
        signal_type=signal_type,
        n_target_epochs=target_epochs.shape[0],
        n_nontarget_epochs=nontarget_epochs.shape[0],
        tmin=args.tmin,
        tmax=args.tmax,
    )

    meta = {
        "xdf": str(xdf_path),
        "output": str(out_path),
        "signal_name": signal_name,
        "signal_type": signal_type,
        "fs": fs,
        "n_target_epochs": int(target_epochs.shape[0]),
        "n_nontarget_epochs": int(nontarget_epochs.shape[0]),
        "tmin": args.tmin,
        "tmax": args.tmax,
    }

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Brainprint template saved to:", out_path)
    print("Metadata saved to:", meta_path)
    print("Signal stream:", signal_name, "|", signal_type, "| fs =", fs)
    print("Target epochs:", target_epochs.shape[0])
    print("Non-target epochs:", nontarget_epochs.shape[0])


if __name__ == "__main__":
    main()

