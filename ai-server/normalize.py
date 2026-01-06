import numpy as np

POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_HIP = 23
POSE_R_HIP = 24

def _safe(arr, idx):
    if arr is None or idx < 0 or idx >= arr.shape[0]:
        return None
    return arr[idx]

def normalize_frame(pose, lhand, rhand, face):
    """
    pose:(25,3), lhand:(21,3), rhand:(21,3), face:(70,3) or None
    return: flat (411,)
    전역 정규화:
      center = mid-hip (fallback mean)
      scale  = shoulder distance (fallback bbox)
    """
    def to_fixed(a, n):
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float32)
        out = np.zeros((n, 3), dtype=np.float32)
        m = min(n, a.shape[0])
        out[:m] = a[:m]
        return out

    pose  = to_fixed(pose, 25)
    lhand = to_fixed(lhand, 21)
    rhand = to_fixed(rhand, 21)
    face  = to_fixed(face, 70)

    # center/scale from pose
    if pose is not None:
        lhip = _safe(pose, POSE_L_HIP)
        rhip = _safe(pose, POSE_R_HIP)
        lsh  = _safe(pose, POSE_L_SHOULDER)
        rsh  = _safe(pose, POSE_R_SHOULDER)

        if lhip is not None and rhip is not None:
            center = (lhip[:2] + rhip[:2]) / 2.0
        else:
            center = pose[:, :2].mean(axis=0)

        if lsh is not None and rsh is not None:
            scale = float(np.linalg.norm(lsh[:2] - rsh[:2]))
        else:
            xy = pose[:, :2]
            scale = float(max(xy[:,0].max()-xy[:,0].min(), xy[:,1].max()-xy[:,1].min()))
    else:
        # pose 없으면 전체 bbox로
        pts = []
        for a in (lhand, rhand, face):
            if a is not None:
                pts.append(a[:, :2])
        if not pts:
            return np.zeros((411,), dtype=np.float32)
        allxy = np.concatenate(pts, axis=0)
        center = allxy.mean(axis=0)
        scale = float(max(allxy[:,0].max()-allxy[:,0].min(), allxy[:,1].max()-allxy[:,1].min()))

    if scale < 1e-6:
        scale = 1.0

    def norm(a, n):
        if a is None:
            return np.zeros((n,3), dtype=np.float32)
        out = a.copy()
        out[:,0] = (out[:,0] - center[0]) / scale
        out[:,1] = (out[:,1] - center[1]) / scale
        # out[:,2] confidence 유지
        return out

    pose_n  = norm(pose, 25)
    lhand_n = norm(lhand, 21)
    rhand_n = norm(rhand, 21)
    face_n  = norm(face, 70)

    feat = np.concatenate([
        pose_n.reshape(-1),
        lhand_n.reshape(-1),
        rhand_n.reshape(-1),
        face_n.reshape(-1),
    ]).astype(np.float32)

    if feat.shape[0] != 411:
        raise ValueError(f"Expected 411 dims, got {feat.shape[0]}")
    return feat
