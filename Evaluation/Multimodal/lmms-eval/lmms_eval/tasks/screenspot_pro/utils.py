import re
from collections import defaultdict


def screenspot_pro_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def screenspot_pro_doc_to_text(doc: dict) -> str:
    return (
        "Bounding box coordinates are specified in the format "
        "(top-left x, top-left y, bottom-right x, bottom-right y). "
        "All values are floating point numbers bounded between 0 and 1 "
        "with two decimal places of precision (e.g., 0.15). "
        "Please provide the bounding box coordinates of the region that "
        "corresponds to the command: " + doc["instruction"]
    )


def _parse_bbox(pred: str) -> list[float]:
    """Extract [x1, y1, x2, y2] from model output string."""
    pattern = (
        r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),"
        r"\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    match = re.search(pattern, pred)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]


def _normalize_bbox(bbox_px: list, img_size: list) -> list[float]:
    """Convert pixel bbox to normalized [0, 1] coordinates."""
    w, h = float(img_size[0]), float(img_size[1])
    return [bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h]


def _center_in_box(gt: list[float], pred: list[float]) -> bool:
    """Return True if the center of pred bbox lies within gt bbox."""
    cx = (pred[0] + pred[2]) / 2
    cy = (pred[1] + pred[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def screenspot_pro_process_result(doc: dict, result: list) -> dict:
    pred = result[0] if result else ""
    pred_bbox = _parse_bbox(pred)
    gt_bbox = _normalize_bbox(list(doc["bbox"]), list(doc["img_size"]))
    return {
        "center_acc": {
            "score": _center_in_box(gt_bbox, pred_bbox),
            "platform": doc["platform"],
            "ui_type": doc["ui_type"],
            "group": doc["group"],
            "application": doc["application"],
        }
    }


def screenspot_pro_center_acc(results: list) -> float:
    """Aggregate center accuracy with per-platform/ui_type/group breakdown."""
    all_scores: list[float] = []
    by_platform: dict[str, list] = defaultdict(list)
    by_ui_type: dict[str, list] = defaultdict(list)
    by_group: dict[str, list] = defaultdict(list)

    for r in results:
        s = float(r["score"])
        all_scores.append(s)
        by_platform[r["platform"]].append(s)
        by_ui_type[r["ui_type"]].append(s)
        by_group[r["group"]].append(s)

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    print("ScreenSpot-Pro Center Accuracy Breakdown:")
    for k, v in sorted(by_platform.items()):
        print(f"  Platform [{k}]: {avg(v):.4f} ({len(v)} samples)")
    for k, v in sorted(by_ui_type.items()):
        print(f"  UI type  [{k}]: {avg(v):.4f} ({len(v)} samples)")
    for k, v in sorted(by_group.items()):
        print(f"  Group    [{k}]: {avg(v):.4f} ({len(v)} samples)")

    return avg(all_scores)
