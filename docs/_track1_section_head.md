
## 11. Official submission (NVIDIA V2D Challenge Track 1, added 2026-09-29)

**Format.** One Parquet (`row_id, x, y, z, code_commit_url`; 740,780 rows) built by the kit's
`eval_reconstruction.py` from one NPZ + one mesh per scored episode. The NPZ holds Meta MHR v1.0.1
parameters indexed by video frame: `pose [T,136]` (root translation in MHR units of 10 cm, root
euler Rz·Ry·Rx, body pose), `scales [68]` and `shape [45]` per episode, and the object Sim(3)
`object_rotation [T,3,3]`, `object_translation [T,3]`, `object_scale` with
`p_scene = s·R·p_mesh + t`, plus `episode_XXXXXX_object.obj`. The host re-poses the parameters with
MHR (`verts = diag(1,-1,-1)·MHR/100`, metres). All 30 episodes are scored: **9,877 frames** inside the
hand–object contact spans (1–7 contiguous stretches per episode). The object mesh budget is
**4,096 vertices / 4,096 faces** (decimated + padded by the packer).

**Metrics (all cm, lower is better).** One Sim(3) is fitted from human vertices at the first scored
frame and applied to the whole episode. CD-H / CD-O: symmetric Chamfer of body / posed-object surface
samples. ACC-H: error of the second frame difference of 22 MHR body joints within contiguous scored
stretches. ACC-O: same for object translation. PEN: mean depth of the 512 MHR hand points inside our
*own* object mesh (scaled by the alignment scale).

**How CHROME maps to them.**
1. *Human* (`chrome/mhr_fit.py`). CHROME v2's optimised MediaPipe-33 joints (metric ruler applied) are
   resampled to 30 fps. Because MediaPipe world landmarks are hip-centred, a per-frame camera-frame
   root translation is recovered from re-extracted image landmarks (`scripts/run_stage_a_image2d.py`,
   `chrome/root_translation.py`: linear pinhole LS per frame, focal `0.9·W` refined by a floor-plane
   height constraint, median + Savitzky–Golay). Joints where CHROME's optimiser drifted > 10 cm from
   Stage A are replaced by Stage A (this caught an episode-12 forearm stretched to 0.9 m by the attach
   term). MHR is fitted on the GPU to all frames at once: 6D root rotation outside MHR's gimbal-prone
   euler root (converted exactly on export), Huber data term on 25 mapped landmarks (+3 virtual head
   markers), CHROME's acceleration prior on the 22 scored joints, parameter-velocity smoothing, MHR's
   own parameter limits, weak pose/scale priors; one `scales` per episode, `shape = 0`, fingers at 0.
   A vectorised-by-tree-level FK reproduces the TorchScript skeleton to 3e-5 cm; exported params
   reproduce the fitted joints to ≤ 2e-4 cm.
2. *Object* (`chrome/object_templates.py`, `chrome/object_track1.py`). One closed template per object
   class in `chrome/assets/objects/` (torus hoop Ø80 cm, bowl shell Ø36×15 cm, iron 26×11×13 cm, desk
   100×50×73 cm, pan Ø26 cm + handle, foam block 30×20×15 cm, paint roller, foam roll Ø15×45 cm, stool
   Ø30×45 cm, laptop cart 60×45×90 cm; catalogue-typical sizes chosen from the task text/videos, **not**
   measured from any reference), `object_scale = 1`. Translation = CHROME's object track in the same
   frame as the body (+ root translation), SG-smoothed. Rotation = resting pose (template z along
   CHROME's floor normal) outside grasps; rigid attachment to the MHR wrist rotation (single hand) or
   to the two-wrist frame (bimanual, re-yawed so the template axis matching the hand gap lies between
   the hands) during grasps, held after release. PEN post-step: hand vertices inside the mesh push the
   object out along the deepest point's surface direction (+3 mm), spread with a temporal Gaussian over
   8 rounds; per-frame fixes only for residuals > 5 mm (millimetre pushes cost more ACC-O than PEN).
3. *Export / packing* (`scripts/export_track1_submission.py`) then the kit packer. Self-checks with the
   kit's own scorer code (`scripts/track1_selfcheck.py`).
