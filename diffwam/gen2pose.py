"""gen2pose: metric camera trajectory of a video that starts at a frame with known metric depth (OpenCV only).

DiffWAM's teacher reconstructs the completed rollout with Pi3 and fixes the scale with MoGe2 depth (Sec. 3.4,
Eq. 16-21). Here the first frame is a simulator render, so its depth is exact. The trajectory is tracked directly
in metres:

  1. corners in frame 0 are back-projected with the true depth -> metric map points (camera-0 frame);
  2. KLT optical flow (forward-backward checked) tracks them frame to frame;
  3. each frame's pose comes from PnP + RANSAC on the tracked map points, refined with Levenberg-Marquardt;
  4. when the map runs low (turns), new corners are tracked as candidates and triangulated once their baseline
     between two posed frames is large enough;
  5. if PnP fails, the pose is propagated at constant velocity (flagged in `ok`).

Poses are returned in the drone's start FLU frame (+x forward, +y left, +z up), as (T, 4) x, y, z, yaw plus full
4x4 camera-to-start matrices. OpenCV camera axes: x right, y down, z forward.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

CV_TO_FLU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # rows: FLU axes in camera coordinates


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)


def _track(prev, cur, pts):
    """KLT with forward-backward check -> (new points, mask)."""
    if len(pts) == 0:
        return pts, np.zeros(0, bool)
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
    fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
    h, w = cur.shape
    p1 = p1.reshape(-1, 2)
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    return p1, (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0) & inside


def _corners(gray, existing, max_corners, mask=None):
    m = np.full(gray.shape, 255, np.uint8) if mask is None else mask.copy()
    for x, y in existing.astype(int):
        cv2.circle(m, (int(x), int(y)), 8, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 8, mask=m)
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


def gen2pose(frames: np.ndarray, depth0: np.ndarray, K: np.ndarray, min_map: int = 250,
             min_parallax_deg: float = 2.0) -> dict:
    """frames (T, H, W, 3) uint8 with frames[0] matching depth0 (H, W) metres -> trajectory dict."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    T = len(gray)
    Kinv = np.linalg.inv(K)
    # 1. metric map from frame 0
    valid = ((depth0 > 0.3) & (depth0 < 30)).astype(np.uint8) * 255
    pts = _corners(gray[0], np.zeros((0, 2)), 1500, valid)
    z = depth0[pts[:, 1].astype(int), pts[:, 0].astype(int)].astype(float)
    xyz = (Kinv @ np.c_[pts, np.ones(len(pts))].T).T * z[:, None]
    map_uv, map_xyz = pts, xyz  # current 2-D positions and 3-D points (camera-0 frame)
    cand_uv = np.zeros((0, 2), np.float32)  # candidates: current uv, first uv, first frame index
    cand_first_uv = np.zeros((0, 2), np.float32)
    cand_first_k = np.zeros(0, int)

    R_wc = [np.eye(3)]  # world (camera-0) -> camera k
    t_wc = [np.zeros(3)]
    ok = [True]
    n_map = [len(map_xyz)]
    for k in range(1, T):
        map_uv, m = _track(gray[k - 1], gray[k], map_uv)
        map_uv, map_xyz = map_uv[m], map_xyz[m]
        cand_uv, mc = _track(gray[k - 1], gray[k], cand_uv)
        cand_uv, cand_first_uv, cand_first_k = cand_uv[mc], cand_first_uv[mc], cand_first_k[mc]

        R, t, good = R_wc[-1], t_wc[-1], False
        if len(map_xyz) >= 12:
            rvec0, _ = cv2.Rodrigues(R)
            # With useExtrinsicGuess OpenCV writes the result into the guess arrays: pass copies, never views of the
            # stored previous pose.
            found, rvec, tvec, inl = cv2.solvePnPRansac(map_xyz, map_uv.astype(float), K, None, rvec0.copy(),
                                                        t.reshape(3, 1).copy(),
                                                        useExtrinsicGuess=True, iterationsCount=200,
                                                        reprojectionError=2.5, confidence=0.999,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if found and inl is not None and len(inl) >= 12:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(map_xyz[inl], map_uv[inl].astype(float), K, None, rvec, tvec)
                R, t, good = cv2.Rodrigues(rvec)[0], tvec.ravel(), True
                keep = np.zeros(len(map_xyz), bool)
                keep[inl] = True
                map_uv, map_xyz = map_uv[keep], map_xyz[keep]
        if not good and len(R_wc) >= 2:  # constant-velocity fallback
            dR = R_wc[-1] @ R_wc[-2].T
            R = dR @ R_wc[-1]
            t = dR @ t_wc[-1] + (t_wc[-1] - dR @ t_wc[-2])
        R_wc.append(R)
        t_wc.append(t)
        ok.append(good)

        # 4. promote candidates with enough parallax to map points by triangulation
        if len(cand_uv) and good:
            P_k = K @ np.c_[R, t]
            new_xyz, promote = [], np.zeros(len(cand_uv), bool)
            for j in np.unique(cand_first_k):
                sel = np.where(cand_first_k == j)[0]
                P_j = K @ np.c_[R_wc[j], t_wc[j]]
                X = cv2.triangulatePoints(P_j, P_k, cand_first_uv[sel].T.astype(float), cand_uv[sel].T.astype(float))
                X = (X[:3] / X[3]).T
                c_j, c_k = -R_wc[j].T @ t_wc[j], -R.T @ t
                a, b = X - c_j, X - c_k
                cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                parallax = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                depth_k = (X @ R.T + t)[:, 2]
                depth_j = (X @ R_wc[j].T + t_wc[j])[:, 2]
                proj = (P_k @ np.c_[X, np.ones(len(X))].T).T
                err = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - cand_uv[sel], axis=1)
                good_pt = (parallax > min_parallax_deg) & (depth_k > 0.2) & (depth_j > 0.2) & (depth_k < 40) & (err < 2.5)
                promote[sel[good_pt]] = True
                new_xyz.append((sel[good_pt], X[good_pt]))
            for idx, X in new_xyz:
                map_uv = np.concatenate([map_uv, cand_uv[idx]])
                map_xyz = np.concatenate([map_xyz, X])
            cand_uv, cand_first_uv, cand_first_k = cand_uv[~promote], cand_first_uv[~promote], cand_first_k[~promote]
        if len(map_xyz) + len(cand_uv) < 4 * min_map and good:  # seed new candidates
            new = _corners(gray[k], np.concatenate([map_uv, cand_uv]), 800)
            cand_uv = np.concatenate([cand_uv, new])
            cand_first_uv = np.concatenate([cand_first_uv, new])
            cand_first_k = np.concatenate([cand_first_k, np.full(len(new), k)])
        n_map.append(len(map_xyz))

    # camera-to-start (camera-0) poses -> drone start FLU frame
    c2s = np.zeros((T, 4, 4))
    out = np.zeros((T, 4))
    for k in range(T):
        Rc = R_wc[k].T  # camera k -> camera 0
        c = -Rc @ t_wc[k]  # camera k centre in camera-0 coordinates
        c2s[k, :3, :3] = CV_TO_FLU @ Rc @ CV_TO_FLU.T
        c2s[k, :3, 3] = CV_TO_FLU @ c
        c2s[k, 3, 3] = 1
        fwd = c2s[k, :3, 0]  # body +x (camera forward) in the start frame
        out[k] = [*c2s[k, :3, 3], math.atan2(fwd[1], fwd[0])]
    return {"pose": out, "c2s": c2s, "ok": np.array(ok), "n_map": np.array(n_map)}
