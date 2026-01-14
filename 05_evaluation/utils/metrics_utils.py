import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

try:
    from skimage.feature import peak_local_max as sk_peak_local_max
except ImportError:
    sk_peak_local_max = None

DEFAULT_IGNORE_TRAIN_IDS = {255, -1}
DEFAULT_IGNORE_NAMES = {"background"}

# Per-class instance method. Adjust here as needed.
# Supported methods: "cc", "watershed", "peaks".
INSTANCE_METHODS = {
    "background": "cc",
    "road": "cc",
    "sidewalk": "cc",
    "parking": "cc",
    "rail track": "cc",
    "building": "cc",
    "guard rail": "cc",
    "pole": "cc",
    "traffic light": "cc",
    "traffic sign": "cc",
    "vegetation": "cc",
    "terrain": "cc",
    "sky": "cc",
    "person": "watershed",
    "rider": "watershed",
    "car": "watershed",
    "truck": "watershed",
    "bus": "watershed",
    "train": "watershed",
    "motorcycle": "peaks",
    "bicycle": "peaks",
}
DEFAULT_INSTANCE_METHOD = "cc"

# Sample manual metrics (replace with real values).
MANUAL_METRICS = {
    "bild6.png": {
        "instances": {
            "background": 0,
            "road": 0,
            "sidewalk": 0,
            "parking": 0,
            "rail track": 0,
            "building": 0,
            "guard rail": 0,
            "pole": 0,
            "traffic light": 0,
            "traffic sign": 0,
            "vegetation": 0,
            "terrain": 0,
            "sky": 0,
            "person": 3,
            "rider": 0,
            "car": 7,
            "truck": 1,
            "bus": 0,
            "train": 0,
            "motorcycle": 0,
            "bicycle": 1,
        },
        "notes": "sample values",
    }
}

# Watershed parameters
WS_DIST_REL = 0.20
WS_MIN_AREA = 6
WS_ERODE_ITERS = 1
WS_USE_PEAKS = True
WS_PEAK_EPS = 1e-6
WS_SEED_OPEN_KERNEL = 1

# Peak parameters
PEAK_DIST_REL = 0.20
PEAK_MIN_DISTANCE = 6
PEAK_ERODE_ITERS = 1


def find_cvat_annotations(start_path=None):
    if start_path is None:
        start_path = Path.cwd().resolve()
    else:
        start_path = Path(start_path).resolve()

    candidate_names = [
        ("05_evaluation", "CVATAnnotations"),
        ("05_evaluation", "CVAT Annotations"),
    ]
    for cand in [start_path, *start_path.parents]:
        for p1, p2 in candidate_names:
            candidate_dir = cand / p1 / p2
            if candidate_dir.exists():
                return candidate_dir

    raise AssertionError(
        f"Cannot find CVATAnnotations. Start notebook in repo. CWD={start_path}"
    )


def _load_rgb_image(path: Path, label):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"{label} not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_demo_context(image_name="bild6r.jpg", label_name="bild6.png", start_path=None):
    cvat_path = find_cvat_annotations(start_path=start_path)
    image_path = cvat_path / image_name
    label_path = cvat_path / label_name

    image = _load_rgb_image(image_path, "Image")
    label = _load_rgb_image(label_path, "Label")

    labelmap_path = cvat_path / "labelmap.txt"
    classes, color_to_class = load_labelmap(labelmap_path)

    return {
        "cvat_path": cvat_path,
        "image_path": image_path,
        "label_path": label_path,
        "image": image,
        "label": label,
        "labelmap_path": labelmap_path,
        "classes": classes,
        "color_to_class": color_to_class,
        "image_key": label_path.name,
    }


def _find_repo_root(start_path: Path):
    for p in [start_path, *start_path.parents]:
        if (p / "labels.py").exists():
            return p
    return None


def _try_load_name2label(repo_root):
    if repo_root is None:
        return {}
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.append(repo_str)
    try:
        from labels import name2label

        return name2label
    except Exception:
        return {}


def load_labelmap(labelmap_path: Path):
    classes = []
    color_to_class = {}
    name2label = _try_load_name2label(_find_repo_root(labelmap_path.resolve()))

    for line in labelmap_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, rest = line.split(":", 1)
            color_str = rest.split(":", 1)[0]
        except ValueError:
            continue

        rgb = tuple(int(c) for c in color_str.split(","))
        rec = {
            "class_id": len(classes),
            "name": name,
            "color": rgb,
            "trainId": name2label[name].trainId if name in name2label else None,
            "id": name2label[name].id if name in name2label else None,
        }
        classes.append(rec)
        color_to_class[rgb] = rec

    return classes, color_to_class


def _ensure_rgb(mask_rgb):
    if isinstance(mask_rgb, np.ndarray):
        arr = mask_rgb
    else:
        try:
            arr = np.asarray(mask_rgb)
        except Exception as exc:
            raise ValueError("mask_rgb must be array-like") from exc

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected mask shape: {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def compute_class_counts(mask_rgb, classes, color_to_class):
    mask = _ensure_rgb(mask_rgb)
    class_counts = {c["name"]: 0 for c in classes}
    flat = mask.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    for color, count in zip(colors, counts):
        rec = color_to_class.get(tuple(color.tolist()))
        if rec is None:
            continue
        class_counts[rec["name"]] += int(count)
    return class_counts


def compute_entropy(class_counts, classes, ignore_train_ids=None, ignore_names=None):
    if ignore_train_ids is None:
        ignore_train_ids = DEFAULT_IGNORE_TRAIN_IDS
    if ignore_names is None:
        ignore_names = DEFAULT_IGNORE_NAMES

    name_to_rec = {c["name"]: c for c in classes}

    def is_ignored(name: str) -> bool:
        rec = name_to_rec.get(name)
        if rec is None:
            return False
        return (rec["trainId"] in ignore_train_ids) or (name in ignore_names)

    valid_pixel_sum = sum(
        count for name, count in class_counts.items() if not is_ignored(name)
    )
    entropy = 0.0
    if valid_pixel_sum > 0:
        for name, count in class_counts.items():
            if is_ignored(name) or count == 0:
                continue
            p = count / valid_pixel_sum
            entropy -= p * math.log(p)
    return entropy


def compute_cc_instances(mask_rgb, classes):
    mask = _ensure_rgb(mask_rgb)
    counts = {c["name"]: 0 for c in classes}
    for rec in classes:
        color = np.array(rec["color"], dtype=np.uint8)
        binary = np.all(mask == color, axis=2).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_components, _ = cv2.connectedComponents(binary)
        counts[rec["name"]] = max(num_components - 1, 0)
    return counts


def _peak_local_max_fallback(dist, min_distance=6, threshold_rel=0.2, labels=None):
    if labels is not None:
        dist = dist.copy()
        dist[labels == 0] = 0
    max_d = float(dist.max())
    if max_d <= 0:
        return np.empty((0, 2), dtype=int)

    thresh = threshold_rel * max_d
    k = max(1, int(min_distance))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    dist32 = dist.astype(np.float32)
    dilated = cv2.dilate(dist32, kernel)
    peaks = (dist32 >= thresh) & (dist32 == dilated)

    if labels is not None:
        peaks = peaks & (labels > 0)

    peaks_u8 = peaks.astype(np.uint8)
    num, cc = cv2.connectedComponents(peaks_u8)
    if num <= 1:
        return np.empty((0, 2), dtype=int)

    coords = []
    for lbl in range(1, num):
        ys, xs = np.where(cc == lbl)
        if len(ys) == 0:
            continue
        idx = np.argmax(dist32[ys, xs])
        coords.append([ys[idx], xs[idx]])
    return np.array(coords, dtype=int)


def compute_watershed_instances(
    mask_rgb,
    classes,
    dist_rel=WS_DIST_REL,
    min_area=WS_MIN_AREA,
    erode_iters=WS_ERODE_ITERS,
    use_peaks=WS_USE_PEAKS,
    peak_eps=WS_PEAK_EPS,
    seed_open_kernel=WS_SEED_OPEN_KERNEL,
    instance_classes=None,
):
    mask = _ensure_rgb(mask_rgb)
    counts = {c["name"]: 0 for c in classes}
    kernel = np.ones((3, 3), np.uint8)
    instance_classes = (
        set(instance_classes)
        if instance_classes is not None
        else {c["name"] for c in classes}
    )

    for rec in classes:
        name = rec["name"]
        if name not in instance_classes:
            continue
        color = np.array(rec["color"], dtype=np.uint8)
        binary = (np.all(mask == color, axis=2)).astype(np.uint8)
        if binary.sum() == 0:
            continue

        work = binary.copy()
        if erode_iters > 0:
            work = cv2.erode(work, kernel, iterations=erode_iters)
            if work.sum() == 0:
                work = binary

        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        max_d = dist.max()
        if max_d <= 0:
            continue

        dist_norm = dist / max_d

        if use_peaks:
            dilated = cv2.dilate(dist, kernel)
            peaks = (dist >= (dilated - peak_eps)) & (dist_norm > dist_rel)
            seed_mask = peaks.astype(np.uint8)
        else:
            seed_mask = (dist_norm > dist_rel).astype(np.uint8)

        if seed_open_kernel and seed_open_kernel > 1:
            sk = np.ones((seed_open_kernel, seed_open_kernel), np.uint8)
            seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, sk)

        num_markers, markers = cv2.connectedComponents(seed_mask)
        if num_markers <= 1 and use_peaks:
            seed_mask = (dist_norm > dist_rel).astype(np.uint8)
            if seed_open_kernel and seed_open_kernel > 1:
                sk = np.ones((seed_open_kernel, seed_open_kernel), np.uint8)
                seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, sk)
            num_markers, markers = cv2.connectedComponents(seed_mask)

        if num_markers <= 1:
            continue

        markers = markers + 1
        markers[binary == 0] = 0

        img_for_ws = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_for_ws, markers)

        unique, areas = np.unique(markers, return_counts=True)
        ws_count = 0
        for lbl, area in zip(unique, areas):
            if lbl > 1 and area >= min_area:
                ws_count += 1
        counts[name] = ws_count

    return counts


def compute_peak_instances(
    mask_rgb,
    classes,
    dist_rel=PEAK_DIST_REL,
    min_distance=PEAK_MIN_DISTANCE,
    erode_iters=PEAK_ERODE_ITERS,
    instance_classes=None,
):
    mask = _ensure_rgb(mask_rgb)
    counts = {c["name"]: 0 for c in classes}
    kernel = np.ones((3, 3), np.uint8)
    instance_classes = (
        set(instance_classes)
        if instance_classes is not None
        else {c["name"] for c in classes}
    )

    for rec in classes:
        name = rec["name"]
        if name not in instance_classes:
            continue
        color = np.array(rec["color"], dtype=np.uint8)
        binary = (np.all(mask == color, axis=2)).astype(np.uint8)
        if binary.sum() == 0:
            continue

        work = binary.copy()
        if erode_iters > 0:
            work = cv2.erode(work, kernel, iterations=erode_iters)
            if work.sum() == 0:
                work = binary

        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        max_d = dist.max()
        if max_d <= 0:
            continue

        if sk_peak_local_max is None:
            peaks = _peak_local_max_fallback(
                dist,
                min_distance=min_distance,
                threshold_rel=dist_rel,
                labels=work,
            )
        else:
            peaks = sk_peak_local_max(
                dist,
                min_distance=min_distance,
                threshold_rel=dist_rel,
                labels=work,
                exclude_border=False,
            )

        counts[name] = len(peaks)

    return counts


def compute_metrics_from_rgb(
    mask_rgb,
    classes,
    color_to_class,
    instance_methods=None,
    ignore_train_ids=None,
    ignore_names=None,
):
    mask = _ensure_rgb(mask_rgb)
    h, w, _ = mask.shape

    class_counts = compute_class_counts(mask, classes, color_to_class)
    class_shares = {
        name: (count / (h * w) if h * w else 0.0)
        for name, count in class_counts.items()
    }
    entropy = compute_entropy(
        class_counts,
        classes,
        ignore_train_ids=ignore_train_ids,
        ignore_names=ignore_names,
    )

    method_map = {}
    for rec in classes:
        name = rec["name"]
        method = (instance_methods or INSTANCE_METHODS).get(name, DEFAULT_INSTANCE_METHOD)
        method_map[name] = method

    ws_classes = {n for n, m in method_map.items() if m == "watershed"}
    peak_classes = {n for n, m in method_map.items() if m == "peaks"}
    cc_counts = compute_cc_instances(mask, classes)
    ws_counts = (
        compute_watershed_instances(mask, classes, instance_classes=ws_classes)
        if ws_classes
        else {c["name"]: 0 for c in classes}
    )
    peak_counts = (
        compute_peak_instances(mask, classes, instance_classes=peak_classes)
        if peak_classes
        else {c["name"]: 0 for c in classes}
    )

    combined_counts = {}
    for rec in classes:
        name = rec["name"]
        method = method_map[name]
        if method == "watershed":
            combined_counts[name] = ws_counts.get(name, 0)
        elif method == "peaks":
            combined_counts[name] = peak_counts.get(name, 0)
        else:
            combined_counts[name] = cc_counts.get(name, 0)

    present_classes = [name for name, count in class_counts.items() if count > 0]

    return {
        "width": w,
        "height": h,
        "class_counts": class_counts,
        "class_shares": class_shares,
        "entropy": entropy,
        "num_present_classes": len(present_classes),
        "present_classes": present_classes,
        "instance_method": method_map,
        "instances": combined_counts,
        "instances_cc": cc_counts,
        "instances_ws": ws_counts,
        "instances_peaks": peak_counts,
    }


def build_instance_table(metrics, classes):
    rows = []
    for rec in classes:
        name = rec["name"]
        rows.append(
            {
                "class": name,
                "method": metrics["instance_method"].get(name, DEFAULT_INSTANCE_METHOD),
                "pixel_share": metrics["class_shares"].get(name, 0.0),
                "pixel_count": metrics["class_counts"].get(name, 0),
                "computed_instances": metrics["instances"].get(name, 0),
            }
        )
    df = pd.DataFrame(rows).set_index("class")
    return df


def build_manual_instance_table(image_key, manual_metrics=None):
    manual_metrics = manual_metrics or MANUAL_METRICS
    manual = manual_metrics.get(image_key, {}).get("instances", {})
    df = pd.DataFrame({"manual_instances": manual}).rename_axis("class")
    return df


def build_instance_comparison_table(computed_df, manual_df):
    df = computed_df[["method", "computed_instances"]].join(manual_df, how="outer")
    df["computed_instances"] = df["computed_instances"].fillna(0).astype(int)
    df["manual_instances"] = df["manual_instances"].fillna(0).astype(int)
    df["diff"] = df["computed_instances"] - df["manual_instances"]
    df["abs_diff"] = df["diff"].abs()
    return df


def build_metrics_report(image_key, metrics, classes, manual_metrics=None):
    summary_df = pd.DataFrame(
        [
            {
                "image": image_key,
                "width": metrics["width"],
                "height": metrics["height"],
                "entropy": metrics["entropy"],
                "num_present_classes": metrics["num_present_classes"],
            }
        ]
    ).set_index("image")

    computed_df = build_instance_table(metrics, classes)
    manual_df = build_manual_instance_table(image_key, manual_metrics=manual_metrics)
    compare_df = build_instance_comparison_table(computed_df, manual_df)

    return {
        "summary": summary_df,
        "computed_instances": computed_df,
        "manual_instances": manual_df,
        "instance_comparison": compare_df,
    }


def make_instance_report(
    rgb_mask,
    image_key,
    classes,
    color_to_class,
    manual_metrics=None,
    instance_methods=None,
):
    metrics = compute_metrics_from_rgb(
        rgb_mask,
        classes,
        color_to_class,
        instance_methods=instance_methods or INSTANCE_METHODS,
    )
    return build_metrics_report(
        image_key,
        metrics,
        classes,
        manual_metrics=manual_metrics or MANUAL_METRICS,
    )
