## Replay (goal = rendered scripted flight, ground truth known)

| space | episodes | pos RMSE m | final err m | hover RMSE m | success | flown dir ok | err / hover |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel | 10 | 0.19 | 0.16 | 1.33 | 9/10 | 10/10 | 0.05 |
| vae | 10 | 0.21 | 0.19 | 1.33 | 9/10 | 10/10 | 0.29 |
| flash | 10 | 0.14 | 0.11 | 1.33 | 10/10 | 10/10 | 0.16 |

## Wan2.2 imagined videos (goal = the imagination)

| space | episodes | flown dir ok | ...where the imagination itself was ok | err / hover |
|---|---:|---:|---:|---:|
| vae | 10 | 9/10 | 8/8 | 0.66 |
| flash | 10 | 2/10 | 2/8 | 0.93 |

## Per episode

| scenario | motion | task | space | seed | fwd m | left m | up m | yaw ° | ref fwd/left/up/yaw | pos RMSE | flown ok | err/hover |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|:-:|---:|
| s01_room_forward | forward | replay | flash | 0 | +2.22 | -0.02 | +0.00 | +0 | 2.2/-0.0/0.0/0 | 0.08 | ✅ | 0.11 |
| s01_room_forward | forward | replay | pixel | 0 | +2.31 | +0.29 | +0.00 | -3 | 2.2/-0.0/0.0/0 | 0.26 | ✅ | 0.15 |
| s01_room_forward | forward | replay | vae | 0 | +2.34 | +0.34 | +0.00 | -3 | 2.2/-0.0/0.0/0 | 0.31 | ✅ | 0.41 |
| s01_room_forward | forward | wan | flash | 0 | +0.58 | +0.10 | -0.29 | -23 |  |  | ✅ | 0.87 |
| s01_room_forward | forward | wan | vae | 0 | +2.99 | +0.39 | +0.08 | -15 |  |  | ✅ | 0.74 |
| s02_hallway | hallway | replay | flash | 0 | +3.72 | -0.04 | +0.00 | +0 | 3.8/-0.0/0.0/0 | 0.07 | ✅ | 0.09 |
| s02_hallway | hallway | replay | pixel | 0 | +3.72 | -0.04 | +0.00 | +0 | 3.8/-0.0/0.0/0 | 0.07 | ✅ | 0.03 |
| s02_hallway | hallway | replay | vae | 0 | +3.72 | -0.04 | +0.00 | +0 | 3.8/-0.0/0.0/0 | 0.08 | ✅ | 0.11 |
| s02_hallway | hallway | wan | flash | 1 | +0.28 | -0.14 | +0.03 | +0 |  |  | ❌ | 0.92 |
| s02_hallway | hallway | wan | vae | 1 | +2.80 | -0.11 | -0.08 | -4 |  |  | ✅ | 0.51 |
| s03_doorway | doorway | replay | flash | 0 | +2.65 | +0.08 | +0.00 | -3 | 2.7/-0.0/0.0/0 | 0.13 | ✅ | 0.25 |
| s03_doorway | doorway | replay | pixel | 0 | +2.69 | +0.02 | +0.00 | +0 | 2.7/-0.0/0.0/0 | 0.11 | ✅ | 0.06 |
| s03_doorway | doorway | replay | vae | 0 | +2.65 | +0.08 | +0.00 | -3 | 2.7/-0.0/0.0/0 | 0.14 | ✅ | 0.35 |
| s03_doorway | doorway | wan | flash | 1 | +0.04 | -0.00 | -0.03 | +0 |  |  | ❌ | 0.97 |
| s03_doorway | doorway | wan | vae | 1 | +2.41 | +0.05 | +0.26 | +0 |  |  | ✅ | 0.46 |
| s04_turn_left | turn left | replay | flash | 0 | +1.35 | +1.03 | +0.00 | +87 | 1.2/1.1/0.0/86 | 0.14 | ✅ | 0.16 |
| s04_turn_left | turn left | replay | pixel | 0 | +1.51 | +0.98 | +0.00 | +91 | 1.2/1.1/0.0/86 | 0.21 | ✅ | 0.05 |
| s04_turn_left | turn left | replay | vae | 0 | +1.46 | +0.99 | +0.00 | +91 | 1.2/1.1/0.0/86 | 0.29 | ✅ | 0.38 |
| s04_turn_left | turn left | wan | flash | 0 | +0.28 | -0.02 | -0.18 | +15 |  |  | ✅ | 0.97 |
| s04_turn_left | turn left | wan | vae | 0 | +1.26 | +0.81 | -0.13 | +19 |  |  | ✅ | 0.69 |
| s05_ascend | ascend | replay | flash | 0 | +0.72 | -0.05 | +0.63 | +0 | 0.7/0.0/0.6/0 | 0.08 | ✅ | 0.20 |
| s05_ascend | ascend | replay | pixel | 0 | +0.76 | -0.01 | +0.66 | +0 | 0.7/0.0/0.6/0 | 0.07 | ✅ | 0.02 |
| s05_ascend | ascend | replay | vae | 0 | +0.68 | +0.22 | +0.66 | -3 | 0.7/0.0/0.6/0 | 0.22 | ✅ | 0.51 |
| s05_ascend | ascend | wan | flash | 0 | +0.39 | +0.27 | -0.03 | +0 |  |  | ❌ | 0.98 |
| s05_ascend | ascend | wan | vae | 0 | +0.57 | -0.65 | +0.32 | -34 |  |  | ✅ | 0.92 |
| s06_bed_approach | approach | replay | flash | 0 | +2.62 | +0.19 | +0.00 | -3 | 2.7/0.0/0.0/0 | 0.19 | ✅ | 0.13 |
| s06_bed_approach | approach | replay | pixel | 0 | +2.54 | +0.14 | +0.03 | -3 | 2.7/0.0/0.0/0 | 0.18 | ✅ | 0.02 |
| s06_bed_approach | approach | replay | vae | 0 | +2.62 | +0.19 | +0.00 | -3 | 2.7/0.0/0.0/0 | 0.19 | ✅ | 0.19 |
| s06_bed_approach | approach | wan | flash | 0 | +0.36 | +0.05 | +0.03 | +0 |  |  | ❌ | 0.97 |
| s06_bed_approach | approach | wan | vae | 0 | +4.15 | +0.56 | -0.21 | -11 |  |  | ✅ | 0.54 |
| s07_turn_right | turn right | replay | flash | 0 | +1.22 | -1.08 | +0.00 | -86 | 1.2/-1.1/0.0/-86 | 0.09 | ✅ | 0.09 |
| s07_turn_right | turn right | replay | pixel | 0 | +1.21 | -1.05 | +0.00 | -86 | 1.2/-1.1/0.0/-86 | 0.20 | ✅ | 0.04 |
| s07_turn_right | turn right | replay | vae | 0 | +1.23 | -1.10 | -0.03 | -86 | 1.2/-1.1/0.0/-86 | 0.13 | ✅ | 0.18 |
| s07_turn_right | turn right | wan | flash | 0 | +0.30 | +0.03 | +0.00 | -11 |  |  | ❌ | 0.92 |
| s07_turn_right | turn right | wan | vae | 0 | +4.60 | -2.16 | +0.03 | -64 |  |  | ✅ | 0.54 |
| s08_strafe_right | strafe right | replay | flash | 0 | +0.05 | -1.21 | +0.00 | -3 | 0.0/-1.5/0.0/0 | 0.31 | ✅ | 0.10 |
| s08_strafe_right | strafe right | replay | pixel | 0 | +0.36 | -1.15 | +0.03 | -3 | 0.0/-1.5/0.0/0 | 0.45 | ✅ | 0.02 |
| s08_strafe_right | strafe right | replay | vae | 0 | -0.02 | -1.07 | +0.00 | -3 | 0.0/-1.5/0.0/0 | 0.36 | ✅ | 0.22 |
| s08_strafe_right | strafe right | wan | flash | 0 | +1.93 | +0.74 | -0.21 | -11 |  |  | ❌ | 0.85 |
| s08_strafe_right | strafe right | wan | vae | 0 | +2.89 | +0.18 | +1.00 | +8 |  |  | ❌ | 0.66 |
| s09_approach_stop | approach+stop | replay | flash | 0 | +1.43 | +0.13 | +0.00 | -3 | 1.5/-0.0/0.0/0 | 0.17 | ✅ | 0.19 |
| s09_approach_stop | approach+stop | replay | pixel | 0 | +1.47 | +0.14 | +0.00 | -3 | 1.5/-0.0/0.0/0 | 0.17 | ✅ | 0.05 |
| s09_approach_stop | approach+stop | replay | vae | 0 | +1.43 | +0.13 | +0.00 | -3 | 1.5/-0.0/0.0/0 | 0.17 | ✅ | 0.27 |
| s09_approach_stop | approach+stop | wan | flash | 0 | +0.06 | -0.11 | -0.03 | -11 |  |  | ❌ | 0.88 |
| s09_approach_stop | approach+stop | wan | vae | 0 | +2.93 | -0.44 | -0.05 | +15 |  |  | ✅ | 0.71 |
| s10_descend | descend | replay | flash | 0 | +1.50 | -0.02 | -0.50 | +0 | 1.5/0.0/-0.5/0 | 0.14 | ✅ | 0.27 |
| s10_descend | descend | replay | pixel | 0 | +1.46 | +0.01 | -0.50 | +0 | 1.5/0.0/-0.5/0 | 0.13 | ✅ | 0.08 |
| s10_descend | descend | replay | vae | 0 | +1.50 | -0.02 | -0.50 | +0 | 1.5/0.0/-0.5/0 | 0.17 | ✅ | 0.31 |
| s10_descend | descend | wan | flash | 0 | +0.08 | +0.11 | -0.08 | -8 |  |  | ❌ | 0.98 |
| s10_descend | descend | wan | vae | 0 | +1.48 | +0.42 | -0.45 | +19 |  |  | ✅ | 0.78 |
