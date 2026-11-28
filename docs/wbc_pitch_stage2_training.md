# Go2+Piper 18-DoF WBC 两阶段全自动训练与部署指南（Arm 2 默认方案）

本文档提供 **18-DoF Unitree Go2 + Piper 机械臂全身控制（WBC, `210D -> 18D`）** 的完整可复现训练流程，对应主线胜出方案 **`go2_supermimic_pitch_s2_arm2_it6500`**。仓库默认参数已对齐该方案：直接按默认配置运行 **Stage-1（`0 -> 3500`）+ Stage-2（`3500 -> 6500`）** 即可复现。

---

## 1. 方案概述与默认参数清单

### 1.1 核心设计动机
1. **拆分滚转/俯仰姿态惩罚（`flat_roll_l2=-1.5`, `flat_pitch_l2=-0.4`）**：
   - 相比各向同性 `flat_orientation_l2=-0.4`（Arm 1），强约束横向 `roll`（`-1.5`）抑制侧向晃动，同时保持适度 `pitch` 顺应性（`-0.4`），允许机身在低蹲（`z=0.08–0.12 m`）或高举（`z=0.80–0.92 m`）时通过合理俯仰泄载 Piper `joint2` 力矩（真实部署限幅 **`11.8125 Nm`**），显著降低高速/慢走时的 `joint2` 饱和率；相比更宽松的 `pitch=-0.20`（Arm 3），Arm 2 功耗与慢走低位 `roll_rms` 更低且综合追踪更稳。
   - CLI 兼容性：当用户仅传入 `--wbc-flat-orientation-l2 <val>` 且未传入 `--wbc-flat-roll-l2` / `--wbc-flat-pitch-l2` 时，`resolve_wbc_reward_weights` 会自动清除未覆写的拆分项，使各向同性覆写仍能按预期生效；不传任何覆写时默认生效 `flat_roll_l2=-1.5, flat_pitch_l2=-0.4`。
2. **去除站立 5× `leg_joint_pos` 放大**：
   - 统一采用 `leg_joint_pos = -0.1`（站立与行走同权），避免站立工况下过度惩罚腿部关节偏离默认站姿而阻碍深蹲/仰身可达性；配合 `ee_pos_exp = 4.5` 与 `standing_feet_air_soft = -1.0` 抑制零速踏步。
3. **扩展工作空间与 15 桶 `(x, z)` 自适应采样（`ε=0.3`）**：
   - 末端位置工作空间扩展为 `pos_x=(0.45, 0.70)`、`pos_y=(-0.35, 0.35)`、`pos_z=(0.10, 0.95)`，配合 `max_xy_radius = 0.69`（`link0` 系下水平半径解析截断，避免不可达角点样本）。
   - 启用 15 桶（`3 × 5`）归一化 `(x, z)` 误差加权自适应采样（`adaptive_sampling=True`, `adaptive_uniform_prob=0.3`, `adaptive_beta=1.25`, `adaptive_err_clip=(0.02, 0.20)`, `adaptive_ema_alpha=0.05`, `adaptive_min_hold_s=2.0`, `adaptive_init_err=0.05`）。
   - 仅使用非探索环境（`eval_env_mask = ~explore`）、静态保持（`~_traj_active`）、驻留时长 `age >= 2.0 s` 且非跌倒重置的样本更新各桶 EMA 误差，防止探索噪声与 Bezier 轨迹瞬态污染采样分布。
4. **Stage-1 → Stage-2 两阶段设计**：
   - **Stage-1（`0 -> 3500`）**：末端姿态限定在 `±20°`（`STAGE1_ORI_ABS_MAX`），全静态指令（`trajectory_mode_prob=0.0`），前 `3000 iter` 线性拉满速度与末端工作空间课程，后 `500 iter` 在满课程下巩固；
   - **Stage-2（`3500 -> 6500`）**：解锁大姿态范围 `roll ±100° / pitch ±65° / yaw ±35°`（`STAGE2_ORI_ABS_MAX`）与 `40%` 连续轨迹模式（`trajectory_mode_prob=0.40`, `trajectory_duration_range=(2.0, 5.0)`, `trajectory_curve_prob=0.5`）。实测可将 `Actor.norm`（`RunningNorm`）四元数 `qw` 标准差从 Stage-1 的 `0.0100` 恢复至 **`0.1164`**（提升 ~11.6×），彻底消除零速大姿态 OOD 截断诱发的对角踏步漂移。

### 1.2 默认超参对照表（已写入代码默认值）

| 模块 | 参数名 | 默认值 | 说明 |
|---|---|---|---|
| `WbcRewardWeights` (`wbc/imagine.py`) | `flat_roll_l2` / `flat_pitch_l2` | `-1.5` / `-0.4` | Arm 2 拆分滚转/俯仰惩罚（`flat_orientation_l2=-0.4` 作显式覆写回退） |
| `WbcRewardWeights` | `ee_pos_exp` / `ee_ori_huber` | `4.5` / `-1.5` | 末端位置指数奖励（`ee_pos_sigma=0.25`）与姿态 Huber 惩罚（`ee_ori_delta=0.5`） |
| `WbcRewardWeights` | `leg_joint_pos` / `standing_feet_air_soft` | `-0.1` / `-1.0` | 无站立 5× 放大；站立悬空软惩罚 |
| `WbcRewardWeights` | `track_lin_vel_xy` / `track_ang_vel_z` / `track_base_height_exp` | `3.0` / `1.0` / `1.0` | 线速度、角速度与基座高度（`0.30 m`）跟踪 |
| `WbcRewardWeights` | `gait_bandpass` / `gait_peak_lse` / `gait_diag_corr` | `1.20` / `-220.0` / `0.45` | Route 2B 窗级步态频段、峰值与对角相关塑形 |
| `WbcEePoseCommandCfg` (`wbc/commands.py`) | `limit_ranges` (`x, y, z`) | `(0.45, 0.70)`, `(-0.35, 0.35)`, `(0.10, 0.95)` | `link0` 系下 x/y 与世界系 z 范围，`max_xy_radius=0.69` |
| `WbcEePoseCommandCfg` | `adaptive_*` | `3×5` 桶, `ε=0.3`, `β=1.25`, `clip=(0.02, 0.20)` | 15 桶误差加权自适应采样（`α=0.05`, `min_hold=2.0s`, `init=0.05`） |
| `WbcTrainCfg` (`wbc/trainer.py`) | `task` / `wbc_stage` | `Go2Piper-WBC-Flat` / `1` | Stage-1 默认平地任务与阶段 |
| `WbcTrainCfg` | `num_envs` / `buffer_len` / `imag_batch` / `horizon` | `2048` / `600` / `4096` / `32` | Replay buffer 与 BPTT 想象批量 |
| `WbcTrainCfg` | `warmup_iterations` / `explore_frac` / `explore_noise` | `300` / `0.35` / `0.35` | 初期 World Model 预热与 OU 探索噪声 |
| `WbcTrainCfg` | `iterations` / `target_curriculum_iterations` / `save_interval` | `3500` / `3000` / `500` | Stage-1 默认总轮数、课程满载步数与保存周期 |
| `WbcTrainCfg` | `lr` / `lr_decay_iterations` / `lr_final_frac` | `2e-4` / `12000` / `0.03` | 余弦退火学习率调度 |
| `WbcTrainCfg` | `ee_adaptive_*` / `ee_max_xy_radius` / `compile` | `True` / `0.3` / `0.69` / `False` | 自适应采样 CLI 透传与关闭 `torch.compile` |

> **契约不变性**：`210D` Actor 观测（`3 × 70D` 历史帧）、`114D` Critic 输入（`WBC_CRITIC_BASE_DIM = 114`）、`18D` 动作输出及 `WbcReplayBuffer` 张量布局均保持零改动。

---

## 2. 环境准备与硬件要求

### 2.1 安装依赖
需要 Python `>= 3.10`、CUDA GPU 与 `mjlab >= 1.6.0`（MuJoCo Warp）：

```bash
# 推荐在独立虚拟环境中安装（uv 或 venv）
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

无头服务器训练与评测必须开启 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
```

### 2.2 显存与耗时参考
- **单卡显存占用**：`num_envs=2048, buffer_len=600, imag_batch=4096` 单进程约占 **`~8.1 GB`** 显存（48 GB 显卡支持单卡双开并行）。
- **单阶段训练耗时**（以 NVIDIA RTX PRO 5000 Blackwell 为例）：
  - **Stage-1（`0 -> 3500 iter`）**：约 **`62–70 min`**（`6.16M` env steps）；
  - **Stage-2（`3500 -> 6500 iter`）**：约 **`62 min`**（单阶段 `+3000 iter`）。

---

## 3. Step-by-Step 完整训练流程

### 3.1 阶段 1：Stage-1 纯从零训练（`0 -> 3500 iter`）

使用仓库新默认值直接启动，无需手动传入一长串 `--wbc-flat-*` 或 `--ee-*` 覆写：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1

python scripts/train.py   --task Go2Piper-WBC-Flat   --wbc-stage 1   --log-root logs/wbc_stage1   --run-name s1_arm2
```

- **内部等价行为**：`iterations=3500`、`target_curriculum_iterations=3000`（前 `3000 iter` 将速度与末端位置课程从 `0.0` 拉满至 `1.0`，`3000 -> 3500` 在满课程下巩固）、`flat_roll_l2=-1.5`、`flat_pitch_l2=-0.4`、`ee_adaptive_sampling=True`、`ee_adaptive_uniform_prob=0.3`、`ee_max_xy_radius=0.69`、`save_interval=500`。
- **产出检查点**：`logs/wbc_stage1/<timestamp>_s1_arm2/model_3500.pt`。

### 3.2 阶段 2：Stage-2 续训（`3500 -> 6500 iter`）

从 Stage-1 的 `model_3500.pt` 切入 `--wbc-stage 2` 并续训至 `6500 iter`：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1

python scripts/train.py   --task Go2Piper-WBC-Flat   --wbc-stage 2   --resume logs/wbc_stage1/<timestamp>_s1_arm2/model_3500.pt   --continue-run true   --iterations 6500   --log-root logs/wbc_stage2   --run-name s2_arm2
```

- **续训机制说明**：
  - 自动切换 `WbcEePoseCommandCfg(stage=2)`：解锁 `roll ±100° / pitch ±65° / yaw ±35°` 末端目标姿态与 `40%` 连续轨迹模式（`trajectory_mode_prob=0.40`）；
  - `--continue-run true` 自动延续 `iteration=3500`、`common_step_counter`、课程状态 `curriculum_state`、World Model/Actor/Critic 优化器状态及 `ee_adaptive_state`（15 桶 EMA 误差与计数）；
  - 采集阶段通过 `~_traj_active` 过滤 Bezier 轨迹过渡步，防止轨迹瞬态误差污染 15 桶采样权重。
- **产出检查点**：`logs/wbc_stage2/<timestamp>_s2_arm2/model_6500.pt`。

### 3.3 可选断点续训与覆写参数速查

| 场景 | 追加 / 修改参数 | 说明 |
|---|---|---|
| **中断后断点续训** | `--resume <run>/model_XXXX.pt --continue-run true --iterations 6500` | 严格沿用原 `iteration`、课程时钟与优化器动量 |
| **切回 Arm 1（各向同性姿态惩罚）** | `--wbc-flat-orientation-l2 -0.4` | 未传 `--wbc-flat-roll-l2` / `--wbc-flat-pitch-l2` 时自动禁用拆分，回退为 `-0.4 * (gx^2 + gy^2)` |
| **切换为 Arm 3（更低 pitch 惩罚）** | `--wbc-flat-roll-l2 -1.5 --wbc-flat-pitch-l2 -0.20` | 进一步放宽俯仰以降低慢走 `joint2` 饱和，但功耗略高 |
| **纯均匀采样消融** | `--ee-adaptive-uniform-prob 1.0` | 保持统计日志但将 15 桶采样概率强制置为均匀 `1/15`（或 `--ee-adaptive-sampling false`） |
| **微扰地形或重置 Critic 微调** | `--task Go2Piper-WBC-MicroRough --reset-critic true --initial-curriculum-progress 1.0` | 重置 Critic（不继承旧 `ee_adaptive_state`）并冻结 Actor `critic_warmup_iterations=500` 步 |

---

## 4. TorchScript 导出与部署侧 LEAPP 边界对接

### 4.1 导出 210D -> 18D TorchScript 策略

```bash
python scripts/export_wbc_policy.py   --checkpoint logs/wbc_stage2/<timestamp>_s2_arm2/model_6500.pt   --output exported/policy_s2_arm2_6500.pt
```

脚本会自动执行前向一致性校验（`batch_sizes=(1, 4)`，`atol=1e-6`）；Arm 2 实测导出文件大小为 `1,113,957` bytes，`max_parity_err = 0.000e+00`。

### 4.2 部署侧 LEAPP (`wbc_mixed`) 边界与力矩注意项
1. **LEAPP Bundle 裁剪边界**：在 `LeggedManip_Lab` 侧导出 `wbc_mixed` LEAPP Bundle 时，必须显式对齐本方案工作空间裁剪范围：
   ```text
   --position-clip-low 0.45,-0.35,0.05 --position-clip-high 0.70,0.35,0.95
   ```
2. **部署侧契约放宽**：在 `Go2_Piper_Deploy` 中，`wbc_mixed_contract.py` 需同步将世界系高度 `z` 合法范围放宽至 `[0.05, 0.95]`（避免高位 `0.80–0.92 m` 与低蹲 `0.08–0.12 m` 指令被旧契约裁剪）。
3. **Piper `joint2` 真实力矩与速率限幅**：本仓库 `go2_supermimic/robot/go2_piper/go2_piper.py` 已将 Piper 前三轴 `ARM_EFFORT_LIMIT` 对齐真实部署口径 **`11.8125 Nm`**（后三轴 `9.5844 Nm`），`ARM_VELOCITY_LIMIT = 3.0 rad/s`（`PiperSlewJointPositionAction`）。

---

## 5. 实测收敛参考与验收基线（Arm 2）

以下指标均来自 `shenzhen5000` 上 Arm 2 两阶段实训日志与双端对比验收实测：

### 5.1 训练期关键指标演化
- **Stage-1 末态（`@3500`）**：
  - `wm_loss = 1.657`，`imagined_return = 12.51`，`imagined_survival = 0.9904`，`Episode_Termination/fell_over = 0.161`
  - `ee_adaptive`: `e_lo = 0.066 m` (`6.6 cm`), `e_hi = 0.097 m` (`9.7 cm`), `p_lo = 0.182`, `p_hi = 0.262`, `p_max = 0.128`, `cur_ee = 1.00`
- **Stage-2 末态（`@6500`）**：
  - `wm_loss = 1.1483`，`imagined_return = 12.3376`，`imagined_survival = 0.9921`，`Episode_Termination/fell_over = 0.3333`
  - `Metrics/ee_pose/position_error = 0.2475`，`Metrics/ee_pose/orientation_error = 0.4025`
  - `ee_adaptive`: `e_lo = 0.0640 m` (`6.4 cm`), `e_hi = 0.0671 m` (`6.7 cm`), `p_lo = 0.2052`, `p_hi = 0.2150`, `p_max = 0.0913`
  - `imag_gait`: `f_implied_hz = 3.1362`, `diag_corr = 0.5907`

### 5.2 `RunningNorm` 方差恢复与零速大姿态抗漂移（11 组探针）
- **`Actor.norm` 末尾帧 `pos_commands[3:7]` (`[206:210]`: `qw, qx, qy, qz`)**：
  - Stage-1 `@3500`：`std = (0.0100, 0.0625, 0.0938, 0.1046)`（`std_qw` 方差坍缩）
  - Stage-2 `@6500`：`std = (0.1164, 0.2910, 0.2087, 0.1686)`（`std_qw` 恢复 **11.6×**）
- **零速大姿态 11 组探针（`vel=(0,0,0), pos=[0.55, 0, 0.45]`，含 `±40°/±90° roll`, `±40°/±60° pitch`, `±30° yaw`）**：
  - `max |disp_xy| = 0.0368 m`（`3.68 cm`），`max |dyaw| = 2.43°`，稳态悬空率 `max air_ss = 0.0050`，11 组总计 `steps_5s = 8`，`mean ee_ori_err = 12.56°`，`max fell_steps = 0`。

### 5.3 30s 6 区间 Pitch Reach Probe（`x=0.65m, y=0.0m, z=(0.50, 0.12, 0.08, 0.50, 0.80, 0.92)`）
- **Standing (`vx = 0.0 m/s`)**：`falls = 0`，pitch 范围 `-15.53° ~ +16.65°`（摆幅 **`32.18°`**），低位 `z=0.08 m` 误差 **`5.04 cm`**，高位 `z=0.92 m` 误差 **`2.49 cm`**，`j2_tau_mean = 9.92 Nm`，`sat@11.81 = 22.4%`（`sat@35.44 = 0.0%`）。
- **Slow Walk (`vx = 0.30 m/s`)**：`falls = 0`，pitch 范围 `-14.49° ~ +16.59°`（摆幅 `31.08°`），低位 `z=0.08 m` 误差 **`4.77 cm`**，高位 `z=0.92 m` 误差 `2.02 cm`，`j2_tau_mean = 10.00 Nm`，`sat@11.81 = 37.7%`（`sat@35.44 = 0.0%`，显著低于 Arm 1 的 `67.7%` 与 Arm 3 的 `42.9%`）。

### 5.4 双端仿真验收（MuJoCo 32-env/20s & Isaac Sim `64 envs × 800 steps`）
- **MuJoCo 32-env / 20s (`seed=123`, Stage-2 满课程口径)**：
  - `survival_rate_pct = 100.0%` (`0` falls)
  - `vel_xy_mae_mps = 0.1337`，`vel_yaw_mae_rps = 0.1826`
  - `ee_pos_mae_cm = 6.94`，`ee_ori_mae_deg = 15.95`
  - `stance_slip_mean = 0.195 m/s`，`force_gt_1N_cycle_freq_hz = 3.16 Hz`，`swing_peak_foot_z_mean = 7.91 cm`
  - `arm_j2_tau_sat@11.81 = 58.31%`（`@35.44 = 0.00%`），`total_robot_power_mean = 138.27 W`
- **Isaac Sim (`64 envs × 800 steps`, `seed=42`)**：
  - **`A-Hard`**（`--stage1_cur 0.376`）：`survival_rate_pct = 100.00%` (`0` falls)，`pos_rmse_m = 0.0767`，`ori_rmse_rad = 0.1723`，`ee_pos_mae_cm = 5.65`，`ee_ori_mae_deg = 8.63`，`total_robot_power_mean = 105.64 W`
  - **`B-Hard`**（`--stage1_cur 1.0 --ori_limit_deg 20.0`）：`survival_rate_pct = 70.31%` (`19` falls, `completed_ep = 27`, `mean_len = 433.9`)，`pos_rmse_m = 0.1264`，`ori_rmse_rad = 0.2183`，`ee_pos_mae_cm = 8.16`，`ee_ori_mae_deg = 10.59`，`total_robot_power_mean = 174.55 W`
