"""Referência local 2D do piso; não equivale a calibrar um plano 3D."""

import av
import cv2
import numpy as np

from footskate import CONTACT_POINTS, MIN_CONF


def stabilize(video, poses):
    """Compensa o movimento dominante do fundo antes de comparar alturas."""
    transforms, quality, timestamps = [], [], []
    accumulated = np.eye(3)
    previous = None
    previous_pose = None
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            image = frame.to_ndarray(format="bgr24")
            scale = min(1., 800 / image.shape[1])
            gray = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            ok = i == 0
            if previous is not None:
                mask = np.zeros_like(previous)
                border = max(5, int(min(mask.shape) * .04))
                mask[border:-border, border:-border] = 255
                visible = previous_pose[:, 2] >= MIN_CONF
                if visible.any():
                    points = previous_pose[visible, :2] * scale
                    low = np.maximum(points.min(axis=0) - 12, 0).astype(int)
                    high = np.minimum(points.max(axis=0) + 12, [gray.shape[1], gray.shape[0]]).astype(int)
                    mask[low[1]:high[1], low[0]:high[0]] = 0
                corners = cv2.goodFeaturesToTrack(previous, 250, .015, 7, mask=mask)
                if corners is not None and len(corners) >= 20:
                    moved, valid, error = cv2.calcOpticalFlowPyrLK(previous, gray, corners, None)
                    valid = (valid.ravel() > 0) & (error.ravel() < 20)
                    if valid.sum() >= 20:
                        a, b = corners[valid, 0], moved[valid, 0]
                        matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
                        if matrix is not None and inliers.mean() >= .65:
                            adjust = np.diag([scale, scale, 1.])
                            matrix = np.linalg.inv(adjust) @ matrix @ adjust
                            if np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > .01:
                                accumulated = accumulated @ np.linalg.inv(matrix)
                                accumulated /= accumulated[2, 2]
                                ok = True
            transforms.append(accumulated.copy())
            quality.append(ok)
            timestamps.append(float(frame.pts * frame.time_base))
            if ok:
                previous, previous_pose = gray, poses[i]
    if len(transforms) != len(poses):
        raise ValueError("Pose e vídeo têm quantidades diferentes de frames.")
    transforms = np.asarray(transforms)
    points = np.concatenate([poses[:, :, :2], np.ones((*poses.shape[:2], 1))], axis=-1)
    stable = np.einsum("nij,nkj->nki", transforms, points)
    denominator = stable[:, :, 2:]
    stable = stable[:, :, :2] / np.where(abs(denominator) > 1e-8, denominator, np.nan)
    return stable, transforms, np.asarray(quality), np.asarray(timestamps)


def body_scale(stable, poses):
    lengths = []
    for hip, knee, ankle in [(9, 10, 11), (12, 13, 14)]:
        valid = (poses[:, [hip, knee, ankle], 2] >= MIN_CONF).all(axis=1)
        length = 2 * (np.linalg.norm(stable[:, hip] - stable[:, knee], axis=1) +
                      np.linalg.norm(stable[:, knee] - stable[:, ankle], axis=1))
        length[~valid] = np.nan
        lengths.append(length)
    # Evita assumir uma altura constante quando o atleta se afasta da câmera.
    combined = np.stack(lengths, axis=1)
    count = np.isfinite(combined).sum(axis=1)
    size = np.nansum(combined, axis=1) / np.maximum(count, 1)
    good = np.isfinite(size) & (size > 10)
    if not good.any():
        return None
    return np.interp(np.arange(len(size)), np.flatnonzero(good), size[good])


def contact_path(stable, poses, probabilities, camera_ok, floor, size, times, threshold):
    """Referência local entre apoios estáveis; os pés no ar não movem o piso."""
    feet = stable[:, CONTACT_POINTS]
    offsets = feet[:, :, 1] - floor["slope"] * feet[:, :, 0]
    visible = (poses[:, CONTACT_POINTS, 2] >= MIN_CONF) & np.isfinite(offsets)
    visible &= camera_ok[:, None] & (poses[:, 8, 2] >= MIN_CONF)[:, None]
    strong = visible & (probabilities >= max(.8, threshold))
    hip = stable[:, 8, 1] - floor["slope"] * stable[:, 8, 0]
    torso = np.linalg.norm(stable[:, 1] - stable[:, 8], axis=1)
    use_torso = strong.any(axis=1) & (poses[:, [1, 8], 2] >= MIN_CONF).all(axis=1)
    use_torso &= np.isfinite(torso) & (torso > 3)
    scale_samples = torso[use_torso]
    depth_changes = (len(scale_samples) >= 6 and
                     np.ptp(np.quantile(scale_samples, [.1, .9])) > .2 * np.median(scale_samples))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    anchors, values = [], []
    previous = floor["offset"]
    max_shift = .2 * float(np.median(size))
    for i, time in enumerate(times):
        start, end = np.searchsorted(times, [time - .075, time + .075], side="left")
        candidates = []
        for j in np.flatnonzero(strong[i]):
            use = np.flatnonzero(strong[start:end, j]) + start
            if len(use) < 3 or len(use) < .6 * (end - start):
                continue
            if times[use[-1]] - times[use[0]] + dt < .12 - 1e-9:
                continue
            scale = float(np.median(size[use]))
            span = np.ptp(np.quantile(offsets[use, j], [.1, .9]))
            if span > (.035 + floor["error_body_fraction"]) * scale:
                continue
            value = float(np.median(offsets[use, j]))
            near = abs(value - previous) <= (.025 + floor["error_body_fraction"]) * scale
            if not depth_changes:
                if near:
                    value = previous  # Ruído e flexão não deslocam o chão.
                else:
                    before = np.arange(np.searchsorted(times, time - .3), i)
                    before = before[visible[before, j]]
                    arriving = (len(before) >= 3 and value - np.quantile(offsets[before, j], .1) > .07 * scale
                                and hip[i] - np.quantile(hip[before], .1) > .05 * scale)
                    if not arriving:
                        continue
            if abs(value - previous) <= max_shift:
                candidates.append(value)
        if candidates:
            previous = float(np.median(candidates))
            anchors.append(i)
            values.append(previous)
    if len(anchors) < 6:
        return np.full(len(poses), floor["offset"]), []
    return np.interp(times, times[anchors], values), anchors


def estimate_ground(stable, poses, probabilities, camera_ok, threshold=.5, timestamps=None):
    """Reta local ajustada a contatos previstos, rejeitando apoios suspensos."""
    size = body_scale(stable, poses)
    if size is None or camera_ok.mean() < .85:
        return None, {"available": False, "reason": "Câmera ou escala corporal sem referência confiável."}
    valid = (poses[:, CONTACT_POINTS, 2] >= MIN_CONF) & (probabilities >= max(.8, threshold)) & camera_ok[:, None]
    indices, regions = np.where(valid)
    if len(indices) < 12 or len(np.unique(indices)) < 6:
        return None, {"available": False, "reason": "Poucos contatos para estimar o piso."}
    samples = stable[indices, np.asarray(CONTACT_POINTS)[regions]]
    heights = size[indices]
    finite = np.isfinite(samples).all(axis=1)
    samples, heights, indices = samples[finite], heights[finite], indices[finite]
    if len(samples) < 12 or len(np.unique(indices)) < 6:
        return None, {"available": False, "reason": "Poucos contatos válidos para estimar o piso."}
    x, y = samples.T
    # Uma referência válida não pode deixar muitos pés abaixo do piso estimado.
    rng = np.random.default_rng(17)
    pairs = rng.integers(0, len(samples), size=(800, 2))
    candidates = [(0., float(np.quantile(y, .9)))]
    for first, second in pairs:
        if abs(x[first] - x[second]) < .1 * min(heights[first], heights[second]):
            continue
        slope = float((y[second] - y[first]) / (x[second] - x[first]))
        if abs(slope) <= 2:
            candidates.append((slope, float(y[first] - slope * x[first])))
    best = None
    for slope, offset in candidates:
        residual = (y - slope * x - offset) / heights
        inliers = abs(residual) <= .045
        below = residual > .07
        if below.mean() > .1 or inliers.sum() < 10:
            continue
        score = int(inliers.sum()) - 4 * int(below.sum())
        if best is None or score > best[0]:
            best = (score, slope, offset, inliers)
    if best is None:
        return None, {"available": False, "reason": "Contatos não sustentam uma referência local de piso."}
    _, slope, offset, inliers = best
    if inliers.mean() < .2 or len(np.unique(indices[inliers])) < 6:
        return None, {"available": False, "reason": "Referência do piso com pouco suporte."}
    residual = (y[inliers] - slope * x[inliers] - offset) / heights[inliers]
    error = max(.015, float(np.quantile(abs(residual), .8)))
    reference = {"slope": slope, "offset": offset, "error_body_fraction": error,
                 "x_min": float(x[inliers].min()), "x_max": float(x[inliers].max())}
    times = np.arange(len(poses)) / 30 if timestamps is None else np.asarray(timestamps)
    offsets, anchors = contact_path(stable, poses, probabilities, camera_ok, reference, size, times, threshold)
    info = {"available": True, "type": "local_2d_contacts", **reference,
            "support_frames": len(np.unique(indices[inliers])), "camera_valid_fraction": float(camera_ok.mean()),
            "contact_anchor_frames": len(anchors), "offset_min": float(offsets.min()), "offset_max": float(offsets.max())}
    return (reference, size, offsets), info


def ground_states(stable, poses, probabilities, camera_ok, reference, timestamps, threshold=.5):
    """-1 fora da evidência; 0 no ar; 1 próximo do piso com apoio previsto."""
    floor, size, offsets = reference
    feet = stable[:, CONTACT_POINTS]
    distance_px = floor["slope"] * feet[:, :, 0] + offsets[:, None] - feet[:, :, 1]
    clearance = distance_px / size[:, None]
    valid = (poses[:, CONTACT_POINTS, 2] >= MIN_CONF) & np.isfinite(clearance)
    valid &= (poses[:, 8, 2] >= MIN_CONF)[:, None]
    valid &= camera_ok[:, None]
    states = np.full(len(poses), -1, dtype=np.int8)
    error = floor["error_body_fraction"]
    padded = np.pad(distance_px, ((1, 1), (0, 0)), mode="edge")
    smoothed = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    speed = (np.gradient(smoothed, timestamps, axis=0) / size[:, None]
             if len(poses) > 1 else np.zeros_like(clearance))
    regions = np.full((len(poses), 4), -1, dtype=np.int8)
    regions[valid & (clearance > .07 + error)] = 0
    rising = (clearance > error).all(axis=1) & (speed > .15).all(axis=1)
    regions[valid.all(axis=1) & rising] = 0
    near = valid & (abs(clearance) <= .025 + error)
    # Apenas proximidade na imagem não basta para confirmar contato físico.
    regions[near & (probabilities >= threshold) & (abs(speed) <= .4)] = 1
    states[(regions == 0).all(axis=1)] = 0
    states[(regions == 1).any(axis=1)] = 1
    return states, regions


def refine_with_ground(video, poses, probabilities, threshold=.5):
    """Estados de contato conferidos com o piso. Sem referência utilizável, states é None."""
    stable, transforms, camera_ok, timestamps = stabilize(video, poses)
    reference, info = estimate_ground(stable, poses, probabilities, camera_ok, threshold, timestamps)
    if reference is None:
        return None, None, info, None
    states, regions = ground_states(stable, poses, probabilities, camera_ok, reference, timestamps, threshold)
    line = reference[0]
    lines = []
    for i, pose in enumerate(poses):
        if pose[8, 2] < MIN_CONF or not camera_ok[i]:
            lines.append(np.full((2, 2), np.nan))
            continue
        x = stable[i, 8, 0]
        xs = np.array([x - reference[1][i] * .5, x + reference[1][i] * .5])
        points = np.column_stack((xs, line["slope"] * xs + reference[2][i], np.ones(2)))
        projected = (np.linalg.inv(transforms[i]) @ points.T).T
        lines.append(projected[:, :2] / projected[:, 2:])
    return states, np.asarray(lines), info, regions
