# -----------------------------------------------------------------------------
# Fig 6: Step 2: Load the bootstrapped regression slopes of all regressors that 
# were computed in Step 1. Quantifify how "smooth" vs. "patchy" each property (regressor) 
# map is on. the cortical surface, both in an unweighted way and in a way that
# softly down-weights unreliable regions using a noise-ceiling (nc) map.
# -----------------------------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
from nilearn.surface import vol_to_surf
from nilearn.image import load_img
from nilearn import masking
import heapq
import pickle
import cortex 

rootdir =  "/YOUR_PATH/principlesOTC"

shuffled = False  # Run again with shuffled = True to compute the shuffled baselines in Fig 6

resdir = pjoin(rootdir, "results/Fig5_smoothness_bootstrap")
ncdir = pjoin(rootdir, "data/noise_ceilings")
topodir = pjoin(rootdir,  "results/Fig6_bootstrap_topographies")
os.makedirs(resdir, exist_ok=True)

subjects = ["1", "2", "3"]

regressors = [
    "animacy", "curvature", "size", "natural", "moves", "precious", "be_moved",
    "heavy", "manmade", "pleasant", "arousal", "agency", "humanness",
    "cuteness", "hardness", "manipulability"
]

fwhm_nc_mm = 6.0 
sigmas_mm = [3.0]
radius_factor = 3.0 
max_neighbors_per_source = 100

rel_scheme = "mean"
rel_gamma = 1.0
normalize_nc = True
nc_floor = 0.0
boundary_taper = False
tau_nc = 0.10

data_type = ""
model_type = "real"
if shuffled:
    model_type = "shuffled"
    data_type = "_shuffled"


# ------------------------------ Mesh utils -----------------------------------
def get_mesh_coords_faces(sub, hemi):
    coords, faces = cortex.db.get_surf(f"S{sub}", type="fiducial", hemisphere=hemi)
    return coords.astype(np.float64), faces.astype(np.int32)


def faces_to_edges_unique(faces):
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    edges = np.vstack([
        np.stack([i, j], axis=1),
        np.stack([j, k], axis=1),
        np.stack([k, i], axis=1),
    ])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    row_u = edges[:, 0].astype(np.int32)
    col_u = edges[:, 1].astype(np.int32)
    return row_u, col_u


def build_adj_lists(n_vertices, row_u, col_u, coords):
    edge_len = np.linalg.norm(coords[row_u] - coords[col_u], axis=1)
    adj = [[] for _ in range(n_vertices)]
    for u, v, d in zip(row_u, col_u, edge_len):
        d_float = float(d)
        adj[u].append((v, d_float))
        adj[v].append((u, d_float))
    return adj, edge_len


# --------------------------- Geodesic neighborhoods ---------------------------
def dijkstra_ball(src, adj, r_max, max_neighbors=None):
    dist = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > r_max:
            continue
        if u in dist:
            continue
        dist[u] = d
        if max_neighbors is not None and len(dist) > max_neighbors:
            break
        for v, wlen in adj[u]:
            nd = d + wlen
            if nd <= r_max and nd < dist.get(v, float("inf")):
                heapq.heappush(heap, (nd, v))
    dist.pop(src, None)
    return dist


def precompute_geodesic_balls(adj, r_max, max_neighbors_per_source):
    n = len(adj)
    balls = []
    for i in range(n):
        balls.append(dijkstra_ball(i, adj, r_max, max_neighbors=max_neighbors_per_source))
    return balls


def balls_to_arrays(geodesic_balls):
    ball_indices = []
    ball_distances = []
    for ball in geodesic_balls:
        if ball:
            neighbors = np.fromiter(ball.keys(), dtype=np.int32, count=len(ball))
            distances = np.fromiter(ball.values(), dtype=np.float64, count=len(ball))
        else:
            neighbors = np.empty(0, dtype=np.int32)
            distances = np.empty(0, dtype=np.float64)
        ball_indices.append(neighbors)
        ball_distances.append(distances)
    return ball_indices, ball_distances


def precompute_spatial_weights(ball_distances, sigmas_mm, radius_factor):
    spa_weights_list = []
    for sigma in sigmas_mm:
        r_cut = radius_factor * sigma
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
        spa_weights_for_sigma = []
        for distances in ball_distances:
            if distances.size == 0:
                spa_weights_for_sigma.append(np.empty(0, dtype=np.float64))
                continue
            weights = np.zeros_like(distances, dtype=np.float64)
            mask = distances <= r_cut
            if np.any(mask):
                d_sel = distances[mask]
                weights[mask] = np.exp(-(d_sel * d_sel) * inv_two_sigma2)
            spa_weights_for_sigma.append(weights)
        spa_weights_list.append(spa_weights_for_sigma)
    return spa_weights_list


def precompute_rel_weights_for_balls(nc, ball_indices, scheme, gamma, floor, use_taper, tau):
    rel_weights_ball = []
    for i, neighbors in enumerate(ball_indices):
        if neighbors.size == 0:
            rel_weights_ball.append(np.empty(0, dtype=np.float64))
            continue
        a_idx = np.full(neighbors.shape, i, dtype=np.int32)
        b_idx = neighbors
        weights = pairwise_rel_weight(
            nc,
            a_idx,
            b_idx,
            scheme=scheme,
            gamma=gamma,
            floor=floor,
            use_taper=use_taper,
            tau=tau,
        )
        rel_weights_ball.append(weights.astype(np.float64))
    return rel_weights_ball


# ------------------------ Project volumes to surface --------------------------
def vol_to_surface_vector(nifti_path, coords_lh, faces_lh, coords_rh, faces_rh):
    img = load_img(nifti_path)
    data_lh = vol_to_surf(img, (coords_lh, faces_lh))
    data_rh = vol_to_surf(img, (coords_rh, faces_rh))
    return np.concatenate([data_lh, data_rh], axis=0).astype(np.float64)


def vols_to_surface_matrix(img_4d, coords_lh, faces_lh, coords_rh, faces_rh):
    # img_4d is already a nifti image (output of masking.unmask)
    data_lh = vol_to_surf(img_4d, (coords_lh, faces_lh))
    data_rh = vol_to_surf(img_4d, (coords_rh, faces_rh))
    return np.concatenate([data_lh, data_rh], axis=0).astype(np.float64)


# ----------------------- Surface gaussian smoothing --------------------------
def fwhm_to_sigma_mm(fwhm_mm):
    return float(fwhm_mm / np.sqrt(8.0 * np.log(2.0)))


def gaussian_surface_smooth(f_vec, geodesic_balls, sigma_mm,
                            nan_safe=True, fill=0.0, radius_multiple=3.0):
    f_vec = np.asarray(f_vec, dtype=np.float64)
    n = f_vec.shape[0]
    out = np.empty_like(f_vec, dtype=np.float64)

    inv_two_sigma2 = 1.0 / (2.0 * sigma_mm * sigma_mm)
    r_cut = radius_multiple * sigma_mm

    if nan_safe:
        is_valid = np.isfinite(f_vec)

    for i in range(n):
        acc = 0.0
        wsum = 0.0

        if not nan_safe or is_valid[i]:
            acc += f_vec[i]
            wsum += 1.0

        ball = geodesic_balls[i]
        for j, d in ball.items():
            if d > r_cut:
                continue
            if nan_safe and not is_valid[j]:
                continue
            w = np.exp(-(d * d) * inv_two_sigma2)
            acc += w * f_vec[j]
            wsum += w

        out[i] = (acc / wsum) if wsum > 1e-12 else float(fill)

    return out


# -------------------------- Reliability weights ------------------------------
def normalize_01(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return (x - xmin) / (max(xmax - xmin, 1e-12))


def pairwise_rel_weight(nc, a, b, scheme="min", gamma=1.0, floor=0.0, use_taper=False, tau=0.10):
    if scheme == "min":
        w = np.minimum(nc[a], nc[b])
    elif scheme == "mean":
        w = 0.5 * (nc[a] + nc[b])
    elif scheme == "product":
        w = nc[a] * nc[b]
    else:
        raise ValueError("rel_scheme must be 'min', 'mean', or 'product'")

    w = np.clip(w, 0.0, 1.0) ** float(gamma)

    if use_taper:
        dn = np.abs(nc[a] - nc[b])
        w *= np.exp(- (dn * dn) / (2.0 * tau * tau))

    return np.maximum(w, floor)


# --------------------------------- Metrics -----------------------------------
def weighted_total_variation(f, row_u, col_u, w_dist, w_dist_sum):
    diffs = np.abs(f[row_u] - f[col_u])
    return float(np.sum(w_dist * diffs) / max(w_dist_sum, 1e-12))


def weighted_total_variation_rel(f, row_u, col_u, w_edges, w_edges_sum):
    diffs = np.abs(f[row_u] - f[col_u])
    num = np.sum(w_edges * diffs)
    return float(num / max(w_edges_sum, 1e-12))


def multiscale_weighted_total_variation(f, ball_indices, spa_weights_list):
    results = {}
    for sigma_idx, sigma in enumerate(sigmas_mm):
        num = 0.0
        den = 0.0
        spa_weights_for_sigma = spa_weights_list[sigma_idx]
        for i, neighbors in enumerate(ball_indices):
            weights = spa_weights_for_sigma[i]
            if neighbors.size == 0 or weights.size == 0:
                continue
            fi = f[i]
            diffs = np.abs(fi - f[neighbors])
            num += np.sum(weights * diffs)
            den += np.sum(weights)
        results[f"wtv_sigma_{int(sigma)}mm"] = float(num / max(den, 1e-12))
    return results


def multiscale_weighted_total_variation_rel(f, ball_indices, spa_weights_list, rel_weights_ball):
    results = {}
    for sigma_idx, sigma in enumerate(sigmas_mm):
        num = 0.0
        den = 0.0
        spa_weights_for_sigma = spa_weights_list[sigma_idx]
        for i, neighbors in enumerate(ball_indices):
            weights_spa = spa_weights_for_sigma[i]
            weights_rel = rel_weights_ball[i]
            if neighbors.size == 0 or weights_spa.size == 0:
                continue
            weights = weights_spa * weights_rel
            fi = f[i]
            diffs = np.abs(fi - f[neighbors])
            num += np.sum(weights * diffs)
            den += np.sum(weights)
        results[f"wtv_sigma_{int(sigma)}mm_rel"] = float(num / max(den, 1e-12))
    return results


# --------------------------------- Runner ------------------------------------
for sub in subjects:
    # 1) meshes
    coords_lh, faces_lh = get_mesh_coords_faces(sub, "lh")
    coords_rh, faces_rh = get_mesh_coords_faces(sub, "rh")

    coords = np.vstack([coords_lh, coords_rh])
    faces_rh_shift = faces_rh + coords_lh.shape[0]
    faces = np.vstack([faces_lh, faces_rh_shift])

    # 2) edges, adjacency, and edge weights
    row_u, col_u = faces_to_edges_unique(faces)
    adj, edge_len = build_adj_lists(coords.shape[0], row_u, col_u, coords)
    edge_len = np.asarray(edge_len, dtype=np.float64)
    w_dist = 1.0 / np.clip(edge_len, 1e-6, None)
    w_dist_sum = float(w_dist.sum())

    # 3) geodesic balls
    r_max_max = max(
        radius_factor * max(sigmas_mm),
        3.0 * fwhm_to_sigma_mm(fwhm_nc_mm),
    )
    geodesic_balls = precompute_geodesic_balls(adj, r_max_max, max_neighbors_per_source)
    ball_indices, ball_distances = balls_to_arrays(geodesic_balls)

    # 4) noise ceiling: project, smooth, normalize
    nc_path = pjoin(ncdir, f"sub-0{sub}_nc_n-12.nii.gz")
    nc_raw = vol_to_surface_vector(nc_path, coords_lh, faces_lh, coords_rh, faces_rh)

    nan_count = np.isnan(nc_raw).sum()
    if nan_count:
        print(
            f"subject {sub}: nc_raw has {nan_count} NaNs → "
            f"these will be handled during surface smoothing"
        )

    sigma_nc = fwhm_to_sigma_mm(fwhm_nc_mm)
    nc = gaussian_surface_smooth(nc_raw, geodesic_balls, sigma_nc)
    if normalize_nc:
        nc = normalize_01(nc)

    # 5) reliability weights (edges + geodesic balls), and spatial weights per sigma
    w_rel_edges = pairwise_rel_weight(
        nc,
        row_u,
        col_u,
        scheme=rel_scheme,
        gamma=rel_gamma,
        floor=nc_floor,
        use_taper=boundary_taper,
        tau=tau_nc,
    )
    w_edges = w_dist * w_rel_edges
    w_edges_sum = float(w_edges.sum())

    rel_weights_ball = precompute_rel_weights_for_balls(
        nc,
        ball_indices,
        scheme=rel_scheme,
        gamma=rel_gamma,
        floor=nc_floor,
        use_taper=boundary_taper,
        tau=tau_nc,
    )
    spa_weights_list = precompute_spatial_weights(
        ball_distances, sigmas_mm, radius_factor
    )

    # 6) regression maps / bootstraps
    sub_rows = []

    maskdir = pjoin(rootdir, "data/masks")
    mask_brain = pjoin(maskdir, f"sub-{int(sub):02d}_space-T1w_brainmask.nii.gz")

    filename = pjoin(topodir,  f"sub0{sub}_all_regressors_bootstrap.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    for reg in regressors:
        all_coefs = np.asarray(data[model_type][reg])  # (n_bootstrap, n_voxels)

        # single unmask call → 4d nifti (mask_shape + (n_bootstrap,))
        img_4d = masking.unmask(all_coefs, mask_brain, order="F")

        # project all bootstraps to surface at once
        f_all = vols_to_surface_matrix(
            img_4d, coords_lh, faces_lh, coords_rh, faces_rh
        )  # (n_vertices, n_bootstrap)

        n_bootstrap = f_all.shape[1]
        for i_boot in range(n_bootstrap):
            f = f_all[:, i_boot]
            f = (f - f.mean()) / (f.std(ddof=1) + 1e-12)

            row = {
                "subject": sub,
                "regressor": reg,
                "bootstrap": i_boot,
                "weighted_total_variation": weighted_total_variation(
                    f, row_u, col_u, w_dist, w_dist_sum
                ),
            }

            row.update(
                multiscale_weighted_total_variation(
                    f, ball_indices, spa_weights_list
                )
            )

            rel_tag = f"{rel_scheme}_g{rel_gamma:.1f}{'_bt' if boundary_taper else ''}"

            row[f"weighted_total_variation_rel_{rel_tag}"] = weighted_total_variation_rel(
                f, row_u, col_u, w_edges, w_edges_sum
            )

            row.update(
                {
                    k.replace("wtv", f"wtv_rel_{rel_tag}"): v
                    for k, v in multiscale_weighted_total_variation_rel(
                        f, ball_indices, spa_weights_list, rel_weights_ball
                    ).items()
                }
            )

            sub_rows.append(row)

    df_sub = pd.DataFrame(sub_rows)
    df_sub.to_csv(
        pjoin(
            resdir,
            f"sub0{sub}_property_topographies_smoothness_weightedVar"
            f"{data_type}_Dijkstra-r3x3_gamma{rel_gamma}_rel-{rel_scheme}.csv",
        ),
        index=False,
    )
