from pycortex_setup import cortex
import cortex.polyutils
import os
from os.path import join as pjoin
import numpy as np
import pickle


rootdir = "/YOUR_PATH/principlesOTC"

surftype = "inflated"
xfmname = "align_auto"
radius = 10 # in mm
n_anchors = [7, 5, 7, 5, 7] # define how many spheres to add between the rois

resdir = pjoin(rootdir, "data/vector-of-roi")
os.makedirs(resdir, exist_ok=True)

rois = ["MPA", "PPA", "FFA", "EBA", "OPA", "MPA"]


def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


def find_closest_coordinates(array, input_coordinate):
    closest_coordinate = None
    closest_index = None
    min_distance = float("inf")  # Set initial distance to infinity

    for i, coordinate in enumerate(array):
        distance = euclidean_distance(coordinate, input_coordinate)
        if distance < min_distance:
            min_distance = distance
            closest_coordinate = coordinate
            closest_index = i

    return closest_coordinate, closest_index


def extract_integers(a1, x):
    first_value = a1[:0]

    if x > 1:
        step_size = (len(a1) - 1) / (x - 1)
        indices = [int(round(i * step_size)) for i in range(x)]
        values = [a1[i] for i in indices]
    else:
        values = []

    anchors = np.concatenate((first_value, values))
    return [int(a) for a in anchors]


def get_centroids_vertex(subject, roi):
    """ improved version that defines the centroid based on the vertex coordinaztes in mm space"""

    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(f"S{subject}", surftype)] # both surfaces
    all_roi = cortex.utils.get_roi_verts(f"S{subject}", roi)[roi] # vertices indices of lh and rh roi together

    verts_mm_l = surfs[0].pts # left vertices coordinates  (all, not only surfce, so N > numl)
    verts_mm_r = surfs[1].pts # right vertices

    numl = surfs[0].pts.shape[0] # number of vertices in lh
    roi_verts = [all_roi[all_roi < numl], all_roi[all_roi >= numl] - numl] # 0 is vertices of the lh roi, 1 of the rh roi

    # use roi_verts as indices to geht the mm coordinates of the verts_l and verts_r
    roi_verts_mm_l = verts_mm_l[roi_verts[0]] # the vertices coordinates of the ROI in lh
    roi_verts_mm_r = verts_mm_r[roi_verts[1]] # same for rh

    # mean of the vertex mm coordinates
    roi_mean_mm_l = np.mean(roi_verts_mm_l, axis=0)
    roi_mean_mm_r = np.mean(roi_verts_mm_r, axis=0)

    # find closest vertex coordinate in the roi that match the mean
    _, il = find_closest_coordinates(roi_verts_mm_l, roi_mean_mm_l)
    _, ir = find_closest_coordinates(roi_verts_mm_r, roi_mean_mm_r)

    # now get the vertex index that corresponds to the mean coordiante
    mean_l = roi_verts[0][il]
    mean_r = roi_verts[1][ir]

    return mean_l, mean_r, roi_verts[0], roi_verts[1]

for subject in [1, 2, 3]:

    # load vertex file ilvuding the single-trial betas, averaged over all trials and categories (to find peak object-activation); length of betas = vertices-l + vertices-r
    betas = np.load(pjoin(rootdir, f"data/whole_brain_FWHM4/average_activation/sub-0{subject}_avg_obj_activation_vertex.pkl"), mmap_mode=None, allow_pickle=True)

    # save data as vertices (1D) and also as 3D arrays so we can compute the results in both vertex and volume space
    vertices_lh_rh = []
    arrays_lh_rh = []

    # version defining the centroid in vertex mm space
    anl = []
    anr = []
    for roi in rois:
        vert_l, vert_r, roi_idx_l, roi_idx_r = get_centroids_vertex(subject, roi)

        if roi == "EBA": # FIND PEAK ACTIVATION VERTEX IN ROI THAT FALLS ON PATH FROM ROI-centroid to V1-centroid: ---------------
            v1_l, v1_r, *_ = get_centroids_vertex(subject, "V1") # 0) get centroid of V1

            surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(f"S{subject}", surftype)]
            assert surfs[0].pts.shape[0] + surfs[1].pts.shape[0] == betas.shape[0], "betas-file containing avg-obj-activations has different shape as surface"

            # 1) paths connecting Roi-centroid and V1-centroid, that are still within the ROI
            paths = [s.geodesic_path(verts[0], verts[1]) for s, verts in zip(surfs, [[vert_l, v1_l], [vert_r, v1_r]])] # path Roi-centroid to V1
            paths_in_roi = [[i for i in paths[0] if i in roi_idx_l], [i for i in paths[1] if i in roi_idx_r]]

            # 2) extract the mean-object-activations for each vertex that falls onto the path AND is int he ROI
            betas_vals_l = betas[paths_in_roi[0]]
            betas_vals_r = betas[np.asarray(paths_in_roi[1], dtype=int) + surfs[0].pts.shape[0]]

            assert len(betas_vals_l) == len(paths_in_roi[0])

            # 3) get the vertex-ID that has the max activation (= new anchor)
            peak_vertex_anchor_l = paths_in_roi[0][np.argmax(betas_vals_l)]
            peak_vertex_anchor_r = paths_in_roi[1][np.argmax(betas_vals_r)]

            vert_l = peak_vertex_anchor_l
            vert_r = peak_vertex_anchor_r

        anl.append(vert_l)
        anr.append(vert_r)

    # get path and extra anchors --------------------------------------------
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(f"S{subject}", surftype)]
    numl = surfs[0].pts.shape[0] # number of vertices in lh
    numr = surfs[1].pts.shape[0] # number of vertices in lh
    nhemis = [numl, numr]

    roi_verts = [anl, anr]

    for i, n_anchor in enumerate(n_anchors):
        paths = [s.geodesic_path(verts[i], verts[i + 1]) for s, verts in zip(surfs, roi_verts)] # path connecting e.g. FFA and PPA

        anchors_l = extract_integers(paths[0], n_anchor) # get additonal anchors on the path between e.g. FFA and PPA
        anchors_r = extract_integers(paths[1], n_anchor)

        # iterate trough all anchors of each path between e.g. FFA and PPA to create a sphere
        for al, ar in zip(anchors_l, anchors_r):
            cur_anchors = [al, ar]
            dists = [s.geodesic_distance(anchor) for anchor, s in zip(cur_anchors, surfs)]

            # Threshold the distances of the left or right hemispher to define surface-circular ROIs
            roi_indices_lh = np.where(dists[0] <= radius)[0]
            roi_indices_rh = np.where(dists[1] <= radius)[0]

            # add a 1 to each vertex belonging to the spheres. I do both hemispheres together, but it is possible to separate them here
            empty_vertex_l = np.zeros(nhemis[0])
            empty_vertex_r = np.zeros(nhemis[1])

            for v in roi_indices_lh:
                empty_vertex_l[v] = 1 # filll vertex mask

            for w in roi_indices_rh:
                empty_vertex_r[w] = 1

            both_sphere_roi = np.hstack((empty_vertex_l, empty_vertex_r))
            both_sphere_roi = cortex.Vertex(both_sphere_roi, f"S{subject}", cmap="plasma")

            # save the vertex map of each sphere as an array, so I can easily multiply it with the data later. Shape = (n_spheres, n_vertices_of_brain)
            vertices_lh_rh.append(both_sphere_roi.data)

            # make volume out of vertex
            sphere_vol = both_sphere_roi.volume(xfmname, "nearest") #To Do: which transformation
            sphere_vol_arr = np.array(sphere_vol.data)

            arrays_lh_rh.append(sphere_vol_arr)

    # save output ------------------------------------------------
    ## all spheres in vertex format
    final_spheres = np.vstack(vertices_lh_rh)
    filename = pjoin(resdir, f"sub0{subject}_surf_spheres.pkl")
    with open(filename, "wb") as f:
        pickle.dump(final_spheres, f)
    print(final_spheres.shape)

    ## all 3D sphere-arrays as a 4D nifti UPDATE
    filename = pjoin(resdir, f"sub0{subject}_surf_spheres.npz")
    np.savez(filename, *arrays_lh_rh)
