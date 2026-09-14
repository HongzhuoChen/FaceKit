# FaceKit Feature Glossary

This document explains the **120 geometric phenotype features** that
`facekit extract-features` writes as columns in `phenotypes_all.csv`. It is a
reference for anyone reading a FaceKit feature CSV who wants to know what each
number means.

The descriptions below are derived directly from the extraction code in
`src/facekit/core/geometric/extractor.py` — they are not approximations.

---

## 1. How to read a feature value

### Coordinate frame
MediaPipe returns 478 landmark points per face. Before any feature is
computed, FaceKit **canonicalizes** the landmarks (`_canonicalize` in
`extractor.py`):

1. **De-rotation** — yaw, pitch and roll are undone in the image plane using
   MediaPipe's facial transformation matrix, and features are computed in the
   resulting frontal plane. Undoing the rotation needs a depth per landmark;
   FaceKit takes it from a **fixed canonical table** (an average face) rather
   than MediaPipe's per-image `z`, which is 92.5% that same template and whose
   residual measurably hurts reliability. If the matrix is unavailable,
   FaceKit falls back to correcting roll alone and reports `derotated = False`.
2. **Centering** — the origin is moved to the midpoint of the two inner eye
   corners.
3. **Scale normalization** — most distances are divided by a face-size
   reference so that a near photo and a far photo of the same face give the
   same numbers.

Image axis convention: **x increases to the right, y increases downward.**
"Left" and "right" follow MediaPipe — i.e. the *subject's* left and right.

### Units
Every feature is **unitless**. There are three kinds:

| Kind | What it is | Typical range |
| --- | --- | --- |
| **Ratio / normalized distance** | a length divided by a face-size reference | small positive number, e.g. 0.2–1.0 |
| **Angle** | measured in **degrees** | e.g. 0–180, or signed for slants |
| **Normalized area** | a polygon area divided by a face-size reference squared | small positive number |

Because everything is normalized, the features describe **face shape**, not
face size in pixels. Two photos at different zoom levels produce the same
feature values.

### The four size references ("scales")
Distances are divided by one of these (all measured on the canonicalized
face):

| Scale | Meaning |
| --- | --- |
| `bizyg` | **bizygomatic width** — distance between the left and right cheekbone points. The main width reference. |
| `face_h` | **face height** — vertical distance from hairline point to chin bottom. |
| `midface_h` | **midface height** — vertical distance from the nose root (nasion) to the base of the nose (subnasale). |
| `mouth_w` | **mouth width** — distance between the two mouth corners. |

### Naming suffixes for paired (bilateral) features
Many face parts come in left/right pairs. FaceKit emits up to four columns
for such a feature:

| Suffix | Meaning |
| --- | --- |
| `_r` | the value measured on the **right** side |
| `_l` | the value measured on the **left** side |
| `_mean` | the average of `_r` and `_l` |
| `_asym` | the **asymmetry**: `abs(r - l) / mean`. A relative difference; `0` means perfectly symmetric, larger means more left/right mismatch. |

`gaze_asym_*` and `face_asymmetry` are dedicated asymmetry measures and are
described in their own rows below.

---

## 2. Feature families

The 120 features come from nine extractor methods. Each section below lists
the columns produced by one method.

### 2.1 Eyebrow features — prefix `eb_` (20 columns)

Computed by `_feat_eyebrow`. Eyebrows are described by five MediaPipe points
along the upper edge and five along the lower edge of each brow.

| Column(s) | Meaning | Sign / units |
| --- | --- | --- |
| `eb_thickness_r/_l/_mean/_asym` | Area of the brow region (star polygon through the 10 brow points, NOT a convex hull), normalized by `bizyg²`. Bigger = thicker / bushier brow. | normalized area |
| `eb_inter_distance` | Gap between the inner ends of the two eyebrows, normalized by `bizyg`. Large = widely separated brows; small = brows that nearly meet (synophrys). | normalized distance |
| `eb_arch_r/_l/_mean/_asym` | How "peaked" the brow is: vertical drop from the highest point of the brow to the average height of its two ends, divided by brow length. Larger = more arched; near 0 = flat brow. | ratio |
| `eb_position_r/_l/_mean/_asym` | Vertical gap between the centre of the brow LOWER edge and the top of the eye, normalized by `midface_h`. Larger = brow sits higher above the eye. This matches the clinical brow-to-eye distance, which is measured to the inferior brow border. | ratio |
| `eb_medial_flare_r/_l/_mean` | Slope across the inner 3 of the 5 upper-edge brow points. Positive = the inner end angles upward toward the nose. | signed slope |
| `eb_lateral_thickness_r/_l/_mean/_asym` | Vertical gap between the upper- and lower-edge landmarks at the lateral (outer) end of the brow, normalized by `bizyg`. | normalized distance |

### 2.2 Eye / periocular features — prefixes `eye_`, `iris_`, `gaze_`, plus three canthal distances (43 columns)

Computed by `_feat_eye`. "Canthus" = the corner of the eye; "fissure" = the
eye opening. The iris center landmarks (478-point model) allow gaze and
pupil features.

| Column(s) | Meaning | Sign / units |
| --- | --- | --- |
| `eye_fissure_length_r/_l/_mean/_asym` | Width of the eye opening (outer corner to inner corner), normalized by `bizyg`. | normalized distance |
| `eye_fissure_height_r/_l/_mean/_asym` | Height of the eye opening (top lid to bottom lid), normalized by `bizyg`. | normalized distance |
| `eye_fissure_slant_r/_l/_mean` | Tilt of the eye, in **degrees**. Positive = the outer corner is **higher** than the inner corner ("up-slanting"); negative = outer corner lower ("down-slanting"). | signed angle (deg) |
| `eye_fissure_slant_asym` | Absolute difference between left and right slant, in degrees. | angle (deg) |
| `eye_fissure_aspect_r/_l/_mean` | Eye height divided by eye length. Larger = rounder/more open eye; smaller = narrower eye. | ratio |
| `inter_canthal_distance` | Distance between the two **inner** eye corners, normalized by `bizyg`. | normalized distance |
| `inter_pupillary_distance` | Distance between the two **iris centers** (the pupils), normalized by `bizyg`. A direct measure of how wide-set the eyes are. | normalized distance |
| `outer_canthal_distance` | Distance between the two **outer** eye corners, normalized by `bizyg`. | normalized distance |
| `canthal_to_pupillary_ratio` | `inter_canthal_distance / inter_pupillary_distance`. High = the inner corners are far apart relative to the pupils (a sign of telecanthus or epicanthal folds). | ratio |
| `eye_epicanthus_angle_r/_l/_mean/_asym` | Angle at the inner eye corner formed by two landmarks on the eye contour. Relates to the epicanthal fold (skin fold at the inner eye). | angle (deg) |
| `eye_upper_lid_to_iris_r/_l/_mean/_asym` | Vertical gap from the top eyelid to the iris center, normalized by `bizyg` (matching the clinical margin-reflex distance MRD1). Larger = more eyelid showing above the iris. | normalized distance |
| `eye_lower_lid_to_iris_r/_l/_mean/_asym` | Same, for the bottom eyelid. | normalized distance |
| `eye_area_r/_l/_mean/_asym` | Area enclosed by the full eye contour, normalized by `bizyg²`. Larger = bigger eye opening. | normalized area |
| `eye_fissure_fill_r/_l/_mean` | Eye area divided by eye-length-squared — how "full" or round the opening is for its width. | ratio |
| `iris_offset_x_r/_l`, `iris_offset_y_r/_l` | Position of the iris center relative to the center of the eye opening, normalized by eye length. `x` = horizontal offset, `y` = vertical offset. Captures gaze direction. | ratio |
| `gaze_asym_x`, `gaze_asym_y` | Difference between the right and left iris offsets in x and y — i.e. how differently the two eyes are pointing. | ratio |
| `gaze_asym_norm` | Overall gaze mismatch: the length of the `(gaze_asym_x, gaze_asym_y)` vector. 0 = both eyes aligned the same way. | ratio |

> **Note.** `iris_offset_*` and `gaze_asym_*` partly reflect where the subject
> was looking when photographed, not only anatomy. Treat large `gaze_asym_*`
> values with care.

### 2.3 Nose features — prefix `nose_`, plus columella/ala/nasolabial columns (18 columns)

Computed by `_feat_nose`.

| Column(s) | Meaning | Sign / units |
| --- | --- | --- |
| `nose_length` | Nasion (nose root) to subnasale (nose base) distance, normalized by `bizyg`. | normalized distance |
| `nose_bridge_length` | Nasion to nose tip distance, normalized by `midface_h`. | normalized distance |
| `nose_bridge_width` | Width across the bony bridge (the lateral pair 196/419), normalized by `bizyg`. | normalized distance |
| `nose_tip_width` | Width across the lateral tip points (45/275), normalized by `bizyg`. | normalized distance |
| `nose_base_width` | Width across the two nostril wings (alae), normalized by `bizyg`. | normalized distance |
| `nose_tip_to_bridge_ratio` | `nose_tip_width / nose_bridge_width`. High = a tip that is broad relative to the bridge (a "bulbous" tip). | ratio |
| `nose_tip_midline_dent` | Vertical position of the nose tip relative to two lateral nose points, normalized by `bizyg`. | signed ratio |
| `nose_ala_height_r/_l/_mean/_asym` | Vertical span of each nostril wing, normalized by `bizyg`. | normalized distance |
| `nasolabial_angle` | Angle at the subnasale between the nose tip and the top of the upper lip, in degrees. Describes how the nose meets the lip. | angle (deg) |
| `nose_tip_elevation` | Vertical gap from subnasale to nose tip, normalized by `bizyg`. Positive = tip sits above the base. | signed ratio |
| `ala_flare_angle` | Angle at the subnasale spanned by the two nostril wings — how flared the nostrils are. | angle (deg) |
| `columella_length` | Vertical distance from subnasale to nose tip, normalized by `midface_h`. (The columella is the strip of tissue between the nostrils.) | normalized distance |
| `columella_hang_below_ala` | How far the subnasale sits below the line of the nostril wings, normalized by `midface_h`. | signed ratio |
| `nose_to_face_area_ratio` | Area of a 4-point nose polygon divided by the area of the whole face oval. | ratio |

### 2.4 Philtrum features — prefix `philtrum_` (2 columns)

Computed by `_feat_philtrum`. The philtrum is the vertical groove between the
nose and the upper lip.

| Column | Meaning | Sign / units |
| --- | --- | --- |
| `philtrum_length` | Distance from subnasale to the top of the upper lip, normalized by `midface_h`. Larger = a longer philtrum. | normalized distance |
| `philtrum_width` | Distance between the two Cupid's-bow peaks, normalized by `bizyg`. | ratio |

### 2.5 Mouth / lip features — prefixes `mouth_`, `lip_`, plus vermilion/cupid columns (16 columns)

Computed by `_feat_mouth_lip`. The "vermilion" is the colored part of the lip.

Lengths in this group are normalized by `bizyg`, not by mouth width: mouth width is
itself the *Wide/Narrow mouth* phenotype, so using it as the reference scale would
cancel part of the signal these columns are meant to carry.

| Column(s) | Meaning | Sign / units |
| --- | --- | --- |
| `mouth_width` | Mouth corner to mouth corner, normalized by `bizyg`. | normalized distance |
| `mouth_opening` | Vertical gap between the inner lip edges, normalized by `bizyg`. ~0 for a closed mouth. | ratio |
| `upper_vermilion_height` | Thickness of the upper lip's red part, normalized by `bizyg`. | ratio |
| `lower_vermilion_height` | Thickness of the lower lip's red part, normalized by `bizyg`. | ratio |
| `vermilion_total` | Sum of upper and lower vermilion heights. Overall lip fullness. | ratio |
| `cupid_bow_drop` | How far the central dip of the upper lip sits below the two Cupid's-bow peaks, normalized by `bizyg`. Larger = a more pronounced Cupid's bow. | signed ratio |
| `mouth_corner_drop` | Vertical position of the mouth corners relative to the upper lip top, normalized by `bizyg`. Positive = corners turned down. | signed ratio |
| `mouth_triangularity` | Vertical gap between the mouth corners and the Cupid's-bow peaks, normalized by `bizyg`. | signed ratio |
| `mouth_tenting` | Same vertical gap, normalized instead by the Cupid's-bow peak span — a "tent" shape measure of the upper lip. | signed ratio |
| `mouth_upper_curvature_c2` | Quadratic (curvature) coefficient of a parabola fitted to the upper lip outline, scaled by `bizyg`. Describes how strongly the upper lip curves; this is what carries the *U-shaped upper lip vermilion* claim. | signed |
| `upper_lip_eversion` | Vertical offset between the inner and outer upper-lip outlines, normalized by `bizyg`. Despite the name this is a vermilion-thickness measure (r = 0.98 against `upper_vermilion_height`); true eversion is a sagittal quantity a frontal photo cannot see. | signed ratio |
| `lower_lip_eversion` | Same for the lower lip (r = 0.99 against `lower_vermilion_height`; same caveat). | signed ratio |
| `lip_midline_x_std` | Horizontal scatter of four midline lip points, normalized by `bizyg`. Large = the lip midline is not vertical (asymmetry). | ratio |

### 2.6 Chin / jaw features — prefixes `chin_`, `jaw_` (4 columns)

Computed by `_feat_chin_jaw`.

| Column | Meaning | Sign / units |
| --- | --- | --- |
| `chin_height` | Vertical distance from the mentolabial sulcus (the crease above the chin) to the chin point (menton), normalized by `bizyg`. Larger = a taller chin. | normalized distance |
| `chin_width` | Distance between two lower-jaw points, normalized by `bizyg`. | normalized distance |
| `chin_pointedness_angle` | Angle at the chin point formed by the two side-of-chin points. Smaller angle = a more pointed chin. | angle (deg) |
| `jaw_bigonial_width` | Distance between the two jaw-angle (gonion) points, normalized by `bizyg`. The width of the lower jaw. | normalized distance |

### 2.7 Forehead features — prefix `forehead_` (4 columns)

Computed by `_feat_forehead`.

| Column | Meaning | Sign / units |
| --- | --- | --- |
| `forehead_height` | Vertical distance from the hairline point to the glabella (between the brows), normalized by `bizyg`. The hairline anchor has no reliable image evidence — see `FEATURE_AUDIT.md` §4.7. | normalized distance |
| `forehead_width_upper` | Forehead width measured high up, normalized by `bizyg`. | normalized distance |
| `forehead_width_mid` | Forehead width measured lower down, normalized by `bizyg`. | normalized distance |
| `forehead_taper_ratio` | `forehead_width_upper / forehead_width_mid`. <1 = the forehead narrows toward the top. | ratio |

### 2.8 Face-global features — prefix `face_`, plus `jaw_to_forehead_ratio` (6 columns)

Computed by `_feat_face_global`. These describe the whole-face outline.

| Column | Meaning | Sign / units |
| --- | --- | --- |
| `face_aspect_ratio` | `bizyg / face_h`. >1 = a wide face; <1 = a long/narrow face. | ratio |
| `face_roundness` | A 0–1 "circularity" score of the face oval (1 = a perfect circle). Lower = a more elongated or angular outline. | ratio |
| `face_width_uniformity` | Average of jaw width and forehead width, divided by `bizyg`. Near 1 = the face has a similar width top, middle and bottom. Forehead width here is the mid-forehead pair (54/284), not the hairline-riding pair used by `forehead_width_upper`. | ratio |
| `jaw_to_forehead_ratio` | Jaw width divided by forehead width. >1 = jaw wider than forehead; <1 = forehead wider than jaw. | ratio |
| `face_triangularity` | `(forehead_width - chin_width) / forehead_width`. High = a wide forehead tapering to a narrow chin (a "heart-shaped" face). | ratio |
| `face_asymmetry` | Average left/right mismatch over 13 mirror-paired landmarks, normalized by `bizyg`. 0 = perfectly symmetric face. | ratio |

### 2.9 Midface / cheek features — prefixes `midface_`, `malar_`, `cheek_` (7 columns)

Computed by `_feat_midface_cheek`. "Malar" refers to the cheekbone region.

| Column(s) | Meaning | Sign / units |
| --- | --- | --- |
| `midface_width` | Distance across the two maxilla (upper-jaw) points, normalized by `bizyg`. | normalized distance |
| `malar_bulge_r/_l/_mean/_asym` | How far the cheek point sticks out sideways relative to the face oval edge at the same height, normalized by `bizyg`. Larger = a more prominent / fuller cheek; negative = a flat or sunken cheek. | signed ratio |
| `cheek_area_r/_l/_mean/_asym` | Area of a small polygon around the cheek, normalized by `bizyg²`. | normalized area |

---

## 3. Quick prefix index

| Prefix | Family | Section |
| --- | --- | --- |
| `eb_` | Eyebrow | 2.1 |
| `eye_`, `iris_`, `gaze_`, `*_canthal_*`, `*_pupillary_*` | Eye / periocular | 2.2 |
| `nose_`, `ala_`, `nostril_`, `columella_`, `nasolabial_` | Nose | 2.3 |
| `philtrum_` | Philtrum | 2.4 |
| `mouth_`, `lip_`, `vermilion_`, `cupid_bow_` | Mouth / lip | 2.5 |
| `chin_`, `jaw_` (`jaw_bigonial_width`) | Chin / jaw | 2.6 |
| `forehead_`, `hairline_` | Forehead | 2.7 |
| `face_`, `jaw_to_forehead_ratio` | Face-global | 2.8 |
| `midface_`, `malar_`, `cheek_` | Midface / cheek | 2.9 |

**Total: 120 feature columns** (plus the metadata columns `disease`,
`image_id`, `frontal_ok`, `pose_yaw`, `pose_pitch`, `pose_roll`).

---

## 4. Custom and plugin features

`facekit extract-features-custom` can select a *subset* of these families per
cohort via a user JSON, and can register *additional* columns through
`facekit.api.register_feature`. Plugin columns are appended after the base 120
and cannot collide with them. See the project `README.md` for the plugin API.
