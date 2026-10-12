import argparse
import pickle
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.masking import apply_mask

rootdir = Path(__file__).resolve().parents[1]
smoothing = 4  # smoothing kernel
subjects = ["01", "02", "03"]


def run_key(path):
    match = re.search(r"ses-things(\d+)_run-(\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Missing session/run number in {path}")
    return tuple(map(int, match.groups()))


def get_files(betasdir, sub):
    """Pair beta images and conditions, sorted numerically by session and run."""
    subject_dir = Path(betasdir) / f"sub-{sub}"
    betas = sorted(subject_dir.rglob(f"sub-{sub}_*_betas.nii*"), key=run_key)
    events = sorted(subject_dir.rglob(f"sub-{sub}_*_conditions.tsv"), key=run_key)
    beta_keys = [run_key(path) for path in betas]
    event_keys = [run_key(path) for path in events]
    if not betas or beta_keys != event_keys or len(set(beta_keys)) != len(beta_keys):
        raise ValueError(f"Missing, duplicate, or unmatched beta/conditions files for sub-{sub}")
    return betas, events


def correct_file_names(events, wrong, correct):
    """Correct known filename prefixes in place, preserving image suffixes."""
    if len(wrong) != len(correct):
        raise ValueError("Each wrong filename must have a corresponding correction")
    for old, new in zip(wrong, correct):
        events["image_filename"] = events["image_filename"].str.replace(old, new, regex=False)
    return events


def load_events(event_files):
    # Load all TSV files in the same order as the beta NIfTI paths.
    frames = []
    for file in event_files:
        frame = pd.read_csv(file, sep="\t")
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")].copy()
        frame["session"], frame["run"] = run_key(file)
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    correct_file_names(
        events,
        wrong=['coat/coat_rack', 'sim/sim_card', 'hula/hula_hoop',
               'ferris/ferris_wheel', 'mosquito/mosquito_net_13s.jpg'],
        correct=['coat_rack/coat_rack', 'sim_card/sim_card', 'hula_hoop/hula_hoop',
                 'ferris_wheel/ferris_wheel', 'mosquito_net/mosquito_net_13s.jpg'],
    )
    # Add concept word to the events and run number.
    events["uniqueID"] = events["image_filename"].str.split("/").str[0]
    return events


def extract_beta_mats(beta_files, mask, smoothing=4):
    """Concatenate runs into an array with trial rows and voxel columns."""
    all_betas = []
    reference = nib.load(beta_files[0])
    resamp_mask = image.resample_to_img(mask, reference, interpolation="nearest")
    for file in beta_files:
        img = nib.load(file)
        # A common grid is required so voxel columns mean the same across runs.
        if img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine):
            raise ValueError(f"Beta image grid differs between runs: {file}")
        img_smoothed = image.smooth_img(img, fwhm=smoothing)
        all_betas.append(np.atleast_2d(apply_mask(img_smoothed, resamp_mask)))
    return np.concatenate(all_betas, axis=0)


def experimental_trials(events, trial_info):
    # The filename order must match events and therefore the beta matrix rows.
    trial_info = trial_info.sort_values(["session", "run"], kind="stable").reset_index(drop=True)
    names = trial_info.rename(columns={"filename": "image_filename"}).copy()
    correct_file_names(names,
                       ['coat/coat_rack', 'sim/sim_card', 'hula/hula_hoop', 'ferris/ferris_wheel', 'mosquito/mosquito_net_13s.jpg'],
                       ['coat_rack/coat_rack', 'sim_card/sim_card', 'hula_hoop/hula_hoop', 'ferris_wheel/ferris_wheel', 'mosquito_net/mosquito_net_13s.jpg'])
    for column in ["session", "run", "image_filename"]:
        if not np.array_equal(events[column].to_numpy(), names[column].to_numpy()):
            raise ValueError(f"Trial-info {column} order does not match conditions")
    # Drop all rows in events and matrices with trial_type != exp.
    return trial_info["trial_type"].eq("exp").to_numpy()


def aggregate_over_concepts(events, matrices):
    # Average all rows across the same uniqueID and sort alphabetically.
    groups = events.groupby("uniqueID", sort=True).indices
    means = np.stack([matrices[indices].mean(axis=0) for indices in groups.values()])
    mean_events = pd.DataFrame({"uniqueID": list(groups),
                                "n_trials": [len(indices) for indices in groups.values()]})
    return means, mean_events


def main(validate_only=False):
    # Create output directory. Generated outputs belong in results/.
    outdir = rootdir / "data" / f"whole_brain_FWHM{smoothing}"
    betasdir = rootdir / "data" / "scalematched"
    for sub in subjects:
        beta_files, event_files = get_files(betasdir, sub)
        events_all = load_events(event_files)
        for beta_file, event_file in zip(beta_files, event_files):
            shape = nib.load(beta_file).shape
            n_trials = shape[3] if len(shape) == 4 else 1
            if n_trials != len(pd.read_csv(event_file, sep="\t")):
                raise ValueError(f"Events and betas have different trial counts: {beta_file}")
        trial_info = pd.read_csv(rootdir / "data" / f"sub-{sub}_trialinfo.csv")
        keep = experimental_trials(events_all, trial_info)
        if validate_only:
            print(f"sub-{sub}: {len(beta_files)} runs, {len(events_all)} trials, {keep.sum()} experimental trials; alignment valid")
            continue
        mask = rootdir / "data" / "masks" / f"sub-{sub}_space-T1w_brainmask.nii.gz"
        matrices = extract_beta_mats(beta_files, mask, smoothing=smoothing)
        assert matrices.shape[0] == events_all.shape[0], 'events and betas have different n trials'
        betas_clean = matrices[keep]
        events_clean = events_all.loc[keep].reset_index(drop=True)
        assert betas_clean.shape[0] == events_clean.shape[0], "different n rows dropped in betas and events"
        outdir.mkdir(parents=True, exist_ok=True)
        # Save data.
        with open(outdir / f"sub{sub}_beta_matrix_whole_brain_no_catch_no_dublicates.pkl", 'wb') as f:
            pickle.dump(betas_clean, f)
        events_clean.to_csv(outdir / f"sub{sub}_events_no_catch_no_dublicates.csv", index=False)
        mean, mean_events = aggregate_over_concepts(events_clean, betas_clean)
        with open(outdir / f"sub{sub}_beta_matrix_whole_brain_means.pkl", 'wb') as f:
            pickle.dump(mean, f)
        mean_events.to_csv(outdir / f"sub{sub}_events_means.csv", index=False)
        print(f"sub-{sub}: saved {len(events_clean)} trials and {len(mean_events)} concept means")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare smoothed whole-brain beta matrices.")
    parser.add_argument("--validate-only", action="store_true", help="Check trial alignment without smoothing or writing outputs")
    main(validate_only=parser.parse_args().validate_only)
