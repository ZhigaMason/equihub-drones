"""The numbers a benchmark run is scored by, as the papers that define them compute them.

FAST-EQA (arXiv 2602.15813, Table 1) scores HM-EQA and MT-HM3D by success rate and normalized
steps, EXPRESS-Bench by LLM-Score and E_path, and OpenEQA's A-EQA by LLM-Match. It defines none
of these itself: the step budget is Explore-EQA's (`int(sqrt(scene area) * 3)`, its
`max_step_room_size_ratio`), C* and E_path are EXPRESS-Bench's (arXiv 2503.11117), LLM-Match
is OpenEQA's. IndoorUAV (arXiv 2512.19024) scores its long flights, the VLN set, by SR within
2 m, NE, OSR and nDTW with d_th = 10.

Where this differs from them, and why, is in drones.sim.scoring.DEVIATIONS: distances here are
straight lines in the scan, not geodesics on a navmesh, and the floor area is a vertex box.
"""
import math
import re

import numpy as np

EQA_STEP_RATIO = 3          # Explore-EQA's max_step_room_size_ratio
FLOOR_BAND = (0.1, 2.0)     # m above the floor: the vertices whose box stands in for the navmesh
DEFAULT_REACH = 0.4         # m one decision flies at most: 16 discrete forwards at 2.5 cm
SUCCESS_RADIUS = 2.0        # m: IndoorUAV's VLN success
NDTW_THRESHOLD = 10.0       # IndoorUAV's d_th for VLN

# A letter alone, or opening the answer: B, (B), B), B., B) blue. Not the B of "Blue".
_LETTER = re.compile(r'^\(?([A-H])\)?(?:[).:,]|\s|$)')


def floor_area(vertices, floor_z, band=FLOOR_BAND):
    """m² of the x-y box around the scan's `vertices` between band[0] and band[1] above
    `floor_z`. Explore-EQA takes this from habitat's navmesh bounds on the floor; there is no
    navmesh here, and vertices a little above the floor are the walls and furniture that bound
    the same space."""
    v = np.asarray(vertices, float).reshape(-1, 3)
    near = v[(v[:, 2] > floor_z + band[0]) & (v[:, 2] < floor_z + band[1])]
    if not len(near):
        return 0.0
    return float(np.ptp(near[:, 0]) * np.ptp(near[:, 1]))


def eqa_budget(area):
    """Explore-EQA's step budget for a floor of `area` m², at least 1."""
    return max(1, int(math.sqrt(area) * EQA_STEP_RATIO))


def path_budget(reference_length, reach=DEFAULT_REACH):
    """Decisions to fly twice a reference path of `reference_length` m, `reach` m each."""
    return max(1, math.ceil(2 * reference_length / reach))


def path_length(points):
    """m along `points` (N, 3), straight from each to the next."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def choice_letter(answer, choices):
    """The letter of the option `answer` picks among `choices` ('A) red', ...), or None. A
    letter alone may be lower case; otherwise it must open the answer in upper case, so that
    "Blue sofa" is not read as B. An option's text alone also picks it."""
    if answer is None:
        return None
    text = str(answer).strip()
    letters = [c.split(')', 1)[0].strip() for c in choices]
    if len(text) == 1:
        text = text.upper()
    match = _LETTER.match(text)
    if match and match.group(1) in letters:
        return match.group(1)
    options = {c.split(')', 1)[1].strip().lower(): c.split(')', 1)[0].strip() for c in choices}
    return options.get(text.lower().rstrip('.'))


def parse_mark(text):
    """The judge's mark, 1 to 5, read as OpenEQA's parse_score reads it: the whole reply, or
    the number after "Your mark:"."""
    text = text.strip()
    found = text if text.isdigit() else None
    if found is None:
        match = re.search(r'Your mark:\s*(\d+)', text)
        found = match.group(1) if match else None
    if found is None or not 1 <= int(found) <= 5:
        raise ValueError(f'the judge gave no mark from 1 to 5: {text[:200]!r}')
    return int(found)


def _mean(values):
    values = np.asarray(list(values), float)
    return float(values.mean()) if len(values) else math.nan


def llm_score(marks):
    """EXPRESS-Bench's C*: the mean of mark / 5, in %."""
    return 100 * _mean(np.asarray(marks, float) / 5)


def llm_match(marks):
    """OpenEQA's LLM-Match: the mean of (mark - 1) / 4, in %."""
    return 100 * _mean((np.asarray(marks, float) - 1) / 4)


def _efficiency(reference, flown):
    longest = max(reference, flown)
    return 1.0 if longest == 0 else reference / longest


def e_path(marks, reference_lengths, flown_lengths):
    """EXPRESS-Bench's E_path with its grounding term at 1: mark / 5 x l / max(p, l), in %."""
    return 100 * _mean(m / 5 * _efficiency(ref_len, flown_len)
                       for m, ref_len, flown_len in zip(marks, reference_lengths,
                                                         flown_lengths, strict=True))


def dtw(a, b):
    """Dynamic time warping between point sequences `a` (N, 3) and `b` (M, 3), Euclidean."""
    a, b = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    acc = np.full((len(a) + 1, len(b) + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        # The diagonal and the step down are known for the whole row; the step along it is not.
        best = np.minimum(acc[i - 1, :-1], acc[i - 1, 1:]) + cost[i - 1]
        row = acc[i]
        for j in range(1, len(b) + 1):
            row[j] = min(best[j - 1], row[j - 1] + cost[i - 1, j - 1])
    return float(acc[-1, -1])


def ndtw(reference, path, threshold=NDTW_THRESHOLD):
    """exp(-DTW / (|R| d_th)), |R| the reference's point count, as nDTW was defined for VLN
    (IndoorUAV writes it L_R). Positions only: the yaw term is IndoorUAV's VLA set's."""
    reference = np.asarray(reference, float).reshape(-1, 3)
    if not len(reference):
        return math.nan
    return math.exp(-dtw(reference, path) / (len(reference) * threshold))


def navigation(path, goal, radius=SUCCESS_RADIUS):
    """IndoorUAV's SR, NE and OSR for one flight `path` (N, 3) to `goal`."""
    distance = np.linalg.norm(np.asarray(path, float).reshape(-1, 3) - goal, axis=1)
    return {'success': bool(distance[-1] <= radius), 'ne': float(distance[-1]),
            'oracle': bool((distance <= radius).any())}
