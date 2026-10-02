import os, sys
import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg
from dynamic_inputs import map_panorama_to_network


# Unified objects.csv slot layout (matches Simulator/src/src/dataset_generator.cpp):
#   2  obj_id, type
#   5  cx, cy, cz, r, h
#   6  vx_sim, vy_sim, vz_sim, ax_sim, ay_sim, az_sim
#   6  vx_trk, vy_trk, vz_trk, ax_trk, ay_trk, az_trk
#   6  vx_unc, vy_unc, vz_unc, ax_unc, ay_unc, az_unc
#   1  age
#   15 future_traj (3 * T_future)
#   6  bbox_cx,cy,cz, bbox_sx,sy,sz
# Total per slot: 47
T_FUTURE = 5
SLOT_FIELDS = 2 + 5 + 6 + 6 + 6 + 1 + 3 * T_FUTURE + 6
assert SLOT_FIELDS == 47, f"unexpected slot field count {SLOT_FIELDS}"

# CPA inverse placement: bucket id encoding — in sync with BucketId in dataset_generator.cpp
BUCKET_UNSET       = 0   # legacy / missing drone_state CSV → fall back to the old random distribution
BUCKET_NO_DYNAMIC  = 1   # empty mask, static baseline (still random sampling)
BUCKET_INTERCEPT   = 2   # small miss + t* inside the planning window
BUCKET_NEAR_MISS   = 3   # miss within the ball-radius + body narrow band
BUCKET_NO_THREAT   = 4   # large miss or cannot catch up within the window
BUCKET_ALL_BLOCKED = 5   # 4~6 balls with near-zero miss covering 360°

# Threat buckets (velocity/goal read directly from drone_state CSV, no random sampling): everything except no_dynamic / unset
_CSV_BUCKETS = frozenset({BUCKET_INTERCEPT, BUCKET_NEAR_MISS, BUCKET_NO_THREAT, BUCKET_ALL_BLOCKED})


# Per-object indices within a slot (offsets):
IDX_OBJ_ID = 0
IDX_TYPE   = 1
IDX_CX     = 2
IDX_R      = 5
IDX_H      = 6
IDX_VSIM   = 7   # vx_sim,vy_sim,vz_sim
IDX_ASIM   = 10  # ax_sim..az_sim
IDX_VTRK   = 13
IDX_ATRK   = 16
IDX_VUNC   = 19
IDX_AUNC   = 22
IDX_AGE    = 25
IDX_FUT    = 26  # 3*T_FUTURE = 15 floats
IDX_BBOX_C = 26 + 3 * T_FUTURE   # = 41
IDX_BBOX_S = IDX_BBOX_C + 3      # = 44
# end = 47


def _parse_objects_csv(path, expected_rows):
    """Return (sphere_legacy, cylinder_legacy, obj_vel) where:
       sphere_legacy:   (R, S_sph, 7)   (cx,cy,cz,r,vx,vy,vz) — for safety_loss
       cylinder_legacy: (R, S_cyl, 8)   (cx,cy,cz,r,h,vx,vy,vz)
       obj_vel:         (R, max_oid+1, 3) world-frame v_sim indexed by obj_id
                        (row 0 = obj_id 0 = static/no-hit → zero). For the
                        dynamic head's per-pixel velocity image (obj_id mask
                        lookup). Invalid slots (obj_id == 0) are kept zero-filled.
    """
    if not os.path.exists(path):
        # caller will fall back to legacy dynamic_obstacles-{m}.csv loader
        return None, None, None
    # Robust load: tolerate truncated rows from an interrupted dataset run.
    # We discover the expected column count from the header, then drop any
    # body row whose field count doesn't match (e.g. half-written last line).
    with open(path, "r") as f:
        header = f.readline().rstrip("\n")
        expected_cols = header.count(",") + 1
        good_rows = []
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != expected_cols:
                continue
            try:
                good_rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not good_rows:
        return None, None, None
    rows = np.asarray(good_rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    n_rows, n_cols = rows.shape
    assert n_cols % SLOT_FIELDS == 0, (
        f"objects.csv {path} has {n_cols} cols, not divisible by {SLOT_FIELDS}")
    S = n_cols // SLOT_FIELDS
    slots = rows.reshape(n_rows, S, SLOT_FIELDS)

    sphere_mask = (slots[..., IDX_OBJ_ID] > 0) & (slots[..., IDX_TYPE] == 0)
    cyl_mask    = (slots[..., IDX_OBJ_ID] > 0) & (slots[..., IDX_TYPE] == 1)

    # Build legacy per-frame arrays so existing safety_loss still works without
    # restructuring. We keep a fixed max-slot layout (zeros for invalid).
    S_sph = int(sphere_mask.sum(axis=1).max()) if sphere_mask.any() else 0
    S_cyl = int(cyl_mask.sum(axis=1).max())    if cyl_mask.any()    else 0

    sphere_legacy   = np.zeros((n_rows, S_sph, 7), dtype=np.float32)
    cylinder_legacy = np.zeros((n_rows, S_cyl, 8), dtype=np.float32)

    for r in range(n_rows):
        sphs = slots[r][sphere_mask[r]]
        for k, sl in enumerate(sphs[:S_sph]):
            sphere_legacy[r, k] = np.array([
                sl[IDX_CX], sl[IDX_CX + 1], sl[IDX_CX + 2],
                sl[IDX_R],
                sl[IDX_VSIM], sl[IDX_VSIM + 1], sl[IDX_VSIM + 2],
            ], dtype=np.float32)
        cyls = slots[r][cyl_mask[r]]
        for k, sl in enumerate(cyls[:S_cyl]):
            cylinder_legacy[r, k] = np.array([
                sl[IDX_CX], sl[IDX_CX + 1], sl[IDX_CX + 2],
                sl[IDX_R], sl[IDX_H],
                sl[IDX_VSIM], sl[IDX_VSIM + 1], sl[IDX_VSIM + 2],
            ], dtype=np.float32)

    # obj_id → world-frame v_sim, for the dynamic head's per-pixel velocity image.
    valid_any = (slots[..., IDX_OBJ_ID] > 0)
    max_oid = int(slots[..., IDX_OBJ_ID].max()) if valid_any.any() else 0
    obj_vel = np.zeros((n_rows, max_oid + 1, 3), dtype=np.float32)
    for r in range(n_rows):
        for si in np.where(slots[r, :, IDX_OBJ_ID] > 0)[0]:
            oid = int(slots[r, si, IDX_OBJ_ID])
            obj_vel[r, oid] = slots[r, si, IDX_VSIM:IDX_VSIM + 3]

    return sphere_legacy, cylinder_legacy, obj_vel


class OmniRiskDataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(OmniRiskDataset, self).__init__()
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        # v5 hyper-params (with safe defaults if cfg keys missing)
        def _cfg(key, default):
            try:
                return cfg[key]
            except (KeyError, TypeError):
                return default
        self.mask_aug     = bool(_cfg("mask_aug", True))
        self.mask_fp_rate = float(_cfg("mask_fp_rate", 0.02))
        self.mask_fn_rate = float(_cfg("mask_fn_rate", 0.05))

        # ─── LiDAR re-projection: undo body roll/pitch, keep yaw ───────────
        # The simulator renders the 96×360 spherical range/mask under the full body attitude; to match the inference-side
        # "gravity-aligned stabilized frame" (R_yaw in omnirisk_node.py), loading back-projects to
        # body FLU → R_y(pitch).T @ R_x(roll).T rotation → reproject spherical.
        self.lidar_H        = 96
        self.lidar_W        = 360
        self.lidar_elev_min = np.radians(float(_cfg("lidar_elev_min_deg", -7.0)))
        self.lidar_elev_max = np.radians(float(_cfg("lidar_elev_max_deg", 52.0)))
        self.lidar_max_dist = float(_cfg("lidar_max_dist", 20.0))
        self.yaw_level_enabled = bool(_cfg("yaw_level_compensation", True))
        # Omnidirectional + world-azimuth frame. depth/mask/vel_img use the full panorama (sim already de-yaws
        # into world azimuth), obs/goal are in world frame, rot is identity (endstate already in world-azimuth frame).
        self.omnidirectional = bool(_cfg("omnidirectional", False))
        _rows = np.arange(self.lidar_H, dtype=np.float32)
        _cols = np.arange(self.lidar_W, dtype=np.float32)
        _elev = (self.lidar_elev_max - _rows / (self.lidar_H - 1) *
                 (self.lidar_elev_max - self.lidar_elev_min))
        _az   = _cols / self.lidar_W * 2 * np.pi - np.pi
        self._bp_cos_el = np.cos(_elev).astype(np.float32)
        self._bp_sin_el = np.sin(_elev).astype(np.float32)
        self._bp_cos_az = np.cos(_az).astype(np.float32)
        self._bp_sin_az = np.sin(_az).astype(np.float32)

        # random state distributions (unchanged)
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        self.vx_lognorm_mean = np.log(1 - cfg["vx_mean_unit"])
        self.vx_logmorm_sigma = np.log(cfg["vx_std_unit"])
        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]])
        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]])
        self.goal_length = cfg['goal_length']
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        # under omni, random sampling for the no_dynamic bucket draws goal/vel around the same heading → positively correlated
        self.heading_vel_std = np.radians(float(_cfg("heading_vel_std_deg", 25.0)))
        if mode == 'train': self.print_data()

        # dataset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.img_list, self.mask_list, self.map_idx = [], [], []
        self.positions   = np.empty((0, 3), dtype=np.float32)
        self.quaternions = np.empty((0, 4), dtype=np.float32)
        self.dynamic_obstacles = []   # legacy per-frame (N,7)
        self.dynamic_cylinders = []   # legacy per-frame (N,8)
        self.obj_vel           = []   # per-frame (max_oid+1, 3) world-frame v by obj_id
        self.buckets           = []   # v3: per-frame bucket id (int)
        self.drone_states      = []   # CPA: per-frame (6,) world vel(3)+goal(3); missing → zeros

        datafolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        datafolders.sort(key=lambda x: int(os.path.basename(x)))
        if mode == 'train':
            print("Datafolders:")
            for folder in datafolders:
                print("    ", folder)

        print("Loading", mode, "dataset")
        for data_idx in range(len(datafolders)):
            datafolder = datafolders[data_idx]

            image_file_names = [datafolder + "/" + filename
                                for filename in os.listdir(datafolder)
                                if os.path.splitext(filename)[1] == '.png'
                                and os.path.basename(filename).startswith('range_')]
            image_file_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[1]))

            # corresponding mask paths (may not exist for legacy datasets)
            mask_file_names = []
            for rng_path in image_file_names:
                idx_str = os.path.basename(rng_path).split('.')[0].split('_')[1]
                m_path  = os.path.join(datafolder, f"mask_{idx_str}.png")
                mask_file_names.append(m_path if os.path.exists(m_path) else "")

            # Robust pose load: tolerate truncated rows from interrupted dataset
            # generation (e.g. map crashed mid-write).
            _pose_path = data_dir + f"/pose-{data_idx}.csv"
            _rows = []
            with open(_pose_path, "r") as _f:
                next(_f)  # header
                for _line in _f:
                    _parts = _line.strip().split(",")
                    if len(_parts) != 7:
                        continue
                    try:
                        _rows.append([float(p) for p in _parts])
                    except ValueError:
                        continue
            states = np.asarray(_rows, dtype=np.float32)
            if states.ndim == 1 or states.size == 0:
                print(f"[skip] pose-{data_idx}.csv has no valid rows")
                continue
            # Truncate image lists to match available pose rows (orphan PNGs ignored)
            n_valid = states.shape[0]
            if len(image_file_names) > n_valid:
                image_file_names = image_file_names[:n_valid]
                mask_file_names  = mask_file_names[:n_valid]
            elif len(image_file_names) < n_valid:
                states = states[:len(image_file_names)]
            positions   = states[:, 0:3]
            quaternions = states[:, 3:7]

            # unified objects-{m}.csv (sphere/cylinder legacy arrays + obj_vel map)
            objects_path = os.path.join(data_dir, f"objects-{data_idx}.csv")
            sph_legacy, cyl_legacy, obj_vel = _parse_objects_csv(objects_path, len(image_file_names))

            if sph_legacy is not None:
                # Align all per-frame arrays to a common length (handles
                # interrupted dataset runs where pose / objects / PNGs end at
                # different row counts).
                n_common = min(sph_legacy.shape[0], len(image_file_names),
                               states.shape[0])
                sph_legacy   = sph_legacy[:n_common]
                cyl_legacy   = cyl_legacy[:n_common]
                obj_vel      = obj_vel[:n_common]
                if len(image_file_names) > n_common:
                    image_file_names = image_file_names[:n_common]
                    mask_file_names  = mask_file_names[:n_common]
                if states.shape[0] > n_common:
                    states      = states[:n_common]
                    positions   = states[:, 0:3]
                    quaternions = states[:, 3:7]
                dynamic_obstacles = sph_legacy
                dynamic_cylinders = cyl_legacy
            else:
                # legacy fallback
                dynamic_obstacles = self._load_dynamic_objects(
                    os.path.join(data_dir, f"dynamic_obstacles-{data_idx}.csv"),
                    len(image_file_names), dims_with_velocity=7, dims_without_velocity=4)
                dynamic_cylinders = self._load_dynamic_objects(
                    os.path.join(data_dir, f"dynamic_cylinders-{data_idx}.csv"),
                    len(image_file_names), dims_with_velocity=8, dims_without_velocity=5)
                obj_vel = np.zeros((len(image_file_names), 1, 3), dtype=np.float32)

            # v3: per-frame bucket id; missing means all UNSET (old distribution)
            n_frames = len(image_file_names)
            buckets = self._load_buckets(
                os.path.join(data_dir, f"bucket-{data_idx}.csv"), n_frames)

            # CPA: per-frame drone intent velocity+goal; missing drone_state CSV → fall back for the whole map:
            #   ignore buckets too, all frames UNSET use old random sampling (old datasets are only for smoke/regression checks).
            drone_states = self._load_drone_state(
                os.path.join(data_dir, f"drone_state-{data_idx}.csv"), n_frames)
            if drone_states is None:
                buckets = [BUCKET_UNSET] * n_frames
                drone_states = np.zeros((n_frames, 6), dtype=np.float32)

            (file_names_train, file_names_val,
             mask_names_train, mask_names_val,
             positions_train,  positions_val,
             quaternions_train, quaternions_val,
             dynamic_train, dynamic_val,
             cylinders_train, cylinders_val,
             obj_vel_train, obj_vel_val,
             buckets_train, buckets_val,
             drone_states_train, drone_states_val) = train_test_split(
                image_file_names, mask_file_names, positions, quaternions,
                dynamic_obstacles, dynamic_cylinders, obj_vel,
                buckets, drone_states,
                test_size=val_ratio, random_state=0)

            if mode == 'train':
                self.img_list.extend(file_names_train)
                self.mask_list.extend(mask_names_train)
                self.positions   = np.vstack((self.positions,   positions_train.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_train.astype(np.float32)))
                self.dynamic_obstacles.extend(list(dynamic_train.astype(np.float32)))
                self.dynamic_cylinders.extend(list(cylinders_train.astype(np.float32)))
                self.obj_vel.extend(list(obj_vel_train.astype(np.float32)))
                self.buckets.extend(list(buckets_train))
                self.drone_states.extend(list(drone_states_train.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(file_names_train))
            elif mode == 'valid':
                self.img_list.extend(file_names_val)
                self.mask_list.extend(mask_names_val)
                self.positions   = np.vstack((self.positions,   positions_val.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_val.astype(np.float32)))
                self.dynamic_obstacles.extend(list(dynamic_val.astype(np.float32)))
                self.dynamic_cylinders.extend(list(cylinders_val.astype(np.float32)))
                self.obj_vel.extend(list(obj_vel_val.astype(np.float32)))
                self.buckets.extend(list(buckets_val))
                self.drone_states.extend(list(drone_states_val.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(file_names_val))
            else:
                raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.img_list):<3} |  Shape: {self.width},{self.height}")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")

    def __len__(self):
        return len(self.img_list)

    def _load_buckets(self, path, expected_rows):
        """Read bucket-{m}.csv (one int column), truncated to expected_rows.
        Missing/empty file → all BUCKET_UNSET, falling back to the old state/goal distribution."""
        if not os.path.exists(path):
            return [BUCKET_UNSET] * expected_rows
        try:
            arr = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.int32)
        except Exception:
            return [BUCKET_UNSET] * expected_rows
        if arr.ndim == 0:
            arr = arr[None]
        ids = arr.tolist()
        if len(ids) >= expected_rows:
            return ids[:expected_rows]
        # short file: pad with UNSET (keeps image alignment, does not break the existing fallback)
        return ids + [BUCKET_UNSET] * (expected_rows - len(ids))

    def _load_mask_objid(self, path, H, W):
        """Load the raw obj_id mask (uint16; 0 = static/no-hit, >0 = obstacle id).
        Used to build the per-pixel velocity image; binary mask derives from it."""
        if not path or not os.path.exists(path):
            return np.zeros((H, W), dtype=np.int32)
        m = cv2.imread(path, -1)
        if m is None:
            return np.zeros((H, W), dtype=np.int32)
        return m.astype(np.int32)

    def _yaw_level_spherical(self, range_img_norm, mask_img, R_body_to_lvl):
        """
        Reproject the spherical range+mask (rendered in body frame) into the yaw-only "gravity-aligned stabilized frame"
        — removes roll/pitch, keeps yaw.

        Args
            range_img_norm: (H, W) float32, normalized range (0~1 maps to 0~max_dist)
            mask_img:       (H, W) float32, binary mask
            R_body_to_lvl:  scipy Rotation, body → yaw-only frame (after applying it
                            the cloud is yaw-locked to the horizontal plane).

        Returns
            (range_out, mask_out) with the same shape as the input; holes have range=1.0 (max_dist), mask=0.
        """
        H, W = range_img_norm.shape
        max_d = self.lidar_max_dist
        valid = range_img_norm < 0.999
        rows, cols = np.where(valid)
        if rows.size == 0:
            return range_img_norm.copy(), mask_img.copy()
        r = range_img_norm[rows, cols] * max_d
        m = mask_img[rows, cols]
        # backproject to body FLU
        cos_el = self._bp_cos_el[rows]; sin_el = self._bp_sin_el[rows]
        cos_az = self._bp_cos_az[cols]; sin_az = self._bp_sin_az[cols]
        pts_body = np.stack([r * cos_el * cos_az,
                             r * cos_el * sin_az,
                             r * sin_el], axis=1)              # (N, 3)
        # rotate body → yaw-only frame
        pts_lvl = R_body_to_lvl.apply(pts_body).astype(np.float32)
        r_lvl = np.linalg.norm(pts_lvl, axis=1).clip(min=1e-6)
        el_lvl = np.arcsin(pts_lvl[:, 2] / r_lvl)
        az_lvl = np.arctan2(pts_lvl[:, 1], pts_lvl[:, 0])
        # only keep points inside LiDAR vertical FOV
        in_fov = (el_lvl >= self.lidar_elev_min) & (el_lvl <= self.lidar_elev_max)
        if not np.any(in_fov):
            return (np.ones((H, W), dtype=np.float32),
                    np.zeros((H, W), dtype=np.float32))
        el_lvl = el_lvl[in_fov]; az_lvl = az_lvl[in_fov]
        r_lvl  = r_lvl[in_fov];  m      = m[in_fov]
        # forward-project to spherical grid
        row_f = (self.lidar_elev_max - el_lvl) / (self.lidar_elev_max - self.lidar_elev_min) * (H - 1)
        col_f = (az_lvl + np.pi) / (2 * np.pi) * W
        row_i = np.clip(np.round(row_f).astype(np.int32), 0, H - 1)
        col_i = np.round(col_f).astype(np.int32) % W
        # depth-aware scatter: for each (row, col) take the minimum-r source point
        dst_id = row_i.astype(np.int64) * W + col_i
        order  = np.lexsort((r_lvl, dst_id))
        dst_s  = dst_id[order];  r_s = r_lvl[order]
        rr_s   = row_i[order];   cc_s = col_i[order];  m_s = m[order]
        _, first_idx = np.unique(dst_s, return_index=True)
        range_out = np.ones((H, W), dtype=np.float32)
        mask_out  = np.zeros((H, W), dtype=np.float32)
        range_out[rr_s[first_idx], cc_s[first_idx]] = r_s[first_idx] / max_d
        mask_out [rr_s[first_idx], cc_s[first_idx]] = m_s[first_idx]
        return range_out, mask_out

    def _augment_mask(self, mask):
        if not self.mask_aug:
            return mask
        # morphological jitter
        ksize = int(np.random.choice([1, 2, 3]))
        kernel = np.ones((ksize, ksize), np.uint8)
        if np.random.rand() < 0.5:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
        # FP/FN random bit-flips
        if self.mask_fp_rate > 0:
            fp = (np.random.rand(*mask.shape) < self.mask_fp_rate).astype(np.float32)
            mask = np.clip(mask + fp, 0.0, 1.0)
        if self.mask_fn_rate > 0:
            fn_keep = (np.random.rand(*mask.shape) >= self.mask_fn_rate).astype(np.float32)
            mask = mask * fn_keep
        return mask.astype(np.float32)

    def __getitem__(self, item):
        # 1. read the full 96×360 range image (16-bit PNG, 0–65535 → 0–max_lidar_dist)
        raw = cv2.imread(self.img_list[item], -1).astype(np.float32) / 65535.0  # (96, 360)
        H, W = raw.shape

        # 2. raw obj_id mask (for per-pixel velocity image + binary localization mask)
        mask_oid = self._load_mask_objid(
            self.mask_list[item] if item < len(self.mask_list) else "", H, W)

        # W: world frame; B/b: body frame
        q_wxyz = self.quaternions[item, :]
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)
        # R_Bw: yaw-only world → body, applied to a world vector gives its body
        # representation (used below for vel/acc/goal). Its inverse (= body →
        # yaw-only frame) roll/pitch-compensates the range image.
        R_Bw = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False).inv()

        # 3. roll/pitch motion compensation: reproject range + obj_id mask into
        #    yaw-only "gravity-aligned" frame so the only orientation cue is yaw.
        if self.yaw_level_enabled:
            R_body_to_lvl = R_Bw.inv()
            raw, mask_oid_f = self._yaw_level_spherical(
                raw, mask_oid.astype(np.float32), R_body_to_lvl)
            mask_oid = np.round(mask_oid_f).astype(np.int32)

        # 4. Drone intent velocity/goal + acc. CPA: threat buckets (intercept/near_miss/no_threat/
        #    all_blocked) read drone_state CSV directly (sim placed obstacles from it, so geometry holds), acc is still random;
        #    no_dynamic / unset use the old random sampling (keeps velocity augmentation).
        bucket = (self.buckets[item]
                  if item < len(self.buckets) else BUCKET_UNSET)
        if bucket in _CSV_BUCKETS:
            ds = self.drone_states[item]
            vel_w, goal_w = ds[0:3].astype(np.float32), ds[3:6].astype(np.float32)
            heading_az = float(np.arctan2(vel_w[1], vel_w[0]))
            acc_w = self._sample_random_acc(heading_az if self.omnidirectional else None)
        else:
            # omni: the no_dynamic bucket picks a random heading, goal/vel are sampled around it → positively correlated.
            heading_az = (float(np.random.uniform(-np.pi, np.pi))
                          if self.omnidirectional else None)
            vel_w, acc_w = self._get_random_state(bucket, heading_az)
            goal_w = self._get_random_goal(bucket, heading_az)
        vel_b, acc_b = R_Bw.apply(vel_w), R_Bw.apply(acc_w)

        # Ball velocity / velocity image use ground truth baked in by the sim (CPA fixed at the source, no re-labeling during training).
        ov, sph_frame = self.obj_vel[item], self.dynamic_obstacles[item]

        # 5. per-pixel velocity image: obj_id → world v_o (lookup) → stable frame.
        mid = np.clip(mask_oid, 0, ov.shape[0] - 1)
        vimg_world = ov[mid]                                     # (H, W, 3) world
        vimg = R_Bw.apply(vimg_world.reshape(-1, 3)).reshape(H, W, 3).astype(np.float32)

        # binary localization mask (noisy via augmentation); velocity image stays clean GT.
        mask_full = self._augment_mask((mask_oid > 0).astype(np.float32))

        if self.omnidirectional:
            # Full panorama inputs share the exact runtime 360→network column LUT.
            depth = map_panorama_to_network(raw, W, self.width, True)[None]
            mask = map_panorama_to_network(mask_full, W, self.width, True)[None]
            vel_img = map_panorama_to_network(vimg, W, self.width, True).transpose(2, 0, 1)
        else:
            # front branch: center cols (±FOV/2 around forward, col=W/2)
            col_center = W // 2
            col_half   = self.width // 2
            c0, c1 = col_center - col_half, col_center + col_half
            depth   = raw[:, c0:c1][None]
            mask    = mask_full[:, c0:c1][None]
            vel_img = vimg[:, c0:c1, :].transpose(2, 0, 1)

        # 6. goal (read from CSV or randomly sampled in step 4) → stable frame
        goal_b = R_Bw.apply(goal_w)

        random_obs = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)
        # World-azimuth frame: endstate is already in the gravity-aligned world-azimuth frame, body→world no longer applies heading yaw.
        # After sim de-yaw the pose is always level → R_WB≈I; use identity explicitly to stay yaw-independent.
        rot_wb = (np.eye(3, dtype=np.float32) if self.omnidirectional
                  else R_WB.as_matrix().astype(np.float32))

        return (depth.astype(np.float32), mask.astype(np.float32), vel_img.astype(np.float32),
                self.positions[item], rot_wb, random_obs, self.map_idx[item],
                sph_frame, self.dynamic_cylinders[item])

    def _sample_s1_vy(self):
        """vy sampling for S1+S2 buckets: three tiers S1a/S1b/S1c (in m/s, not scaled by vel_max)."""
        u = np.random.rand()
        if u < 0.4:
            # S1a: straight flight → |vy| < 0.3
            return np.random.uniform(-0.3, 0.3)
        elif u < 0.8:
            # S1b: already flying sideways but no threat → |vy| in [0.5, 2]
            mag = np.random.uniform(0.5, 2.0)
        else:
            # S1c: high lateral speed, no threat → |vy| in [2, 4]
            mag = np.random.uniform(2.0, 4.0)
        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        return sign * mag

    def _load_drone_state(self, path, expected_rows):
        """Read drone_state-{m}.csv (vx,vy,vz,gx,gy,gz), truncated/aligned to expected_rows.
        Missing/empty/unparsable file → None (the caller falls back to UNSET + old random sampling for the whole map)."""
        if not os.path.exists(path):
            return None
        try:
            arr = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != 6:
            return None
        if arr.shape[0] >= expected_rows:
            return arr[:expected_rows]
        pad = np.zeros((expected_rows - arr.shape[0], 6), dtype=np.float32)
        return np.vstack([arr, pad])

    def _sample_random_acc(self, heading_az=None):
        """Randomly sample acceleration (world frame), same as the acc part of _get_random_state.
        Threat buckets use CSV velocity but still sample acc randomly; rotated around heading so it correlates with the course."""
        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:
                break
        if self.omnidirectional and heading_az is not None:
            a = heading_az + np.random.randn() * self.heading_vel_std
            ca, sa = np.cos(a), np.sin(a)
            Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            acc = Rz @ acc
        return acc.astype(np.float32)

    def _get_random_state(self, bucket=BUCKET_UNSET, heading_az=None):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max
            vel[0] = right_skewed_vx
            # v3: S1+S2 buckets → vy uses the uniform S1a/b/c tiers, other buckets keep Gaussian
            if bucket == BUCKET_NO_DYNAMIC:
                vel[1] = self._sample_s1_vy()
            if np.linalg.norm(vel) < 1.2 * self.vel_max:
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:
                break
        if self.omnidirectional:
            # World-azimuth frame: velocity azimuth sampled around heading (positively correlated with goal); heading=None falls back to uniform
            a = (np.random.uniform(-np.pi, np.pi) if heading_az is None
                 else heading_az + np.random.randn() * self.heading_vel_std)
            ca, sa = np.cos(a), np.sin(a)
            Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            vel, acc = Rz @ vel, Rz @ acc
        return vel, acc

    def _load_dynamic_objects(self, path, expected_rows, dims_with_velocity, dims_without_velocity):
        if not os.path.exists(path):
            return np.zeros((expected_rows, 0, dims_with_velocity), dtype=np.float32)
        rows = np.loadtxt(path, delimiter=',', skiprows=1).astype(np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        values = rows[:, 1:]
        if values.size == 0:
            return np.zeros((rows.shape[0], 0, dims_with_velocity), dtype=np.float32)
        if values.shape[1] % dims_with_velocity == 0:
            dynamic_dim = dims_with_velocity
        elif values.shape[1] % dims_without_velocity == 0:
            dynamic_dim = dims_without_velocity
        else:
            raise ValueError(f"cannot parse dynamic object file {path}: value columns={values.shape[1]}")
        max_objects = values.shape[1] // dynamic_dim
        objects = values.reshape(-1, max_objects, dynamic_dim).astype(np.float32)
        if dynamic_dim == dims_without_velocity:
            zero_velocity = np.zeros((*objects.shape[:2], 3), dtype=np.float32)
            objects = np.concatenate([objects, zero_velocity], axis=2)
        return objects

    def _get_random_goal(self, bucket=BUCKET_UNSET, heading_az=None):
        # Omnidirectional world-azimuth frame: goal azimuth sampled around heading (threat buckets → obstacle blocks the path); heading=None falls back to uniform
        if self.omnidirectional:
            goal_pitch_angle = np.radians(np.random.normal(0.0, self.goal_pitch_std))
            goal_yaw_angle = (np.random.uniform(-np.pi, np.pi) if heading_az is None
                              else heading_az + np.radians(np.random.normal(0.0, self.goal_yaw_std)))
            goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                                   np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle),
                                   np.sin(goal_pitch_angle)])
            if np.random.rand() < 0.1:
                goal_w_dir = np.random.rand() * 10 * goal_w_dir
            return self.goal_length * goal_w_dir
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        # v3: S1+S2 buckets → 50/50 split
        #   half keep Gaussian goal_yaw (→ S1)
        #   half force a sideways goal at ±[40°, 60°] (→ S2), a control with the same sideways goal as S5 but a different mask
        if bucket == BUCKET_NO_DYNAMIC and np.random.rand() < 0.5:
            mag = np.random.uniform(40.0, 60.0)
            sign = 1.0 if np.random.rand() < 0.5 else -1.0
            goal_yaw_angle = sign * mag
        else:
            goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir
        return self.goal_length * goal_w_dir

    def print_data(self):
        import scipy.stats as stats
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max

        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")
