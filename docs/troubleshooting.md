# Troubleshooting

| symptom | cause / fix |
|---|---|
| anything fails on start | `bash scripts/doctor.sh` names the missing piece and the command that fixes it |
| `Ninja is required to load C++ extensions` | gsplat's JIT needs the sim env on `PATH`; run through the provided scripts, which source `scripts/_bootstrap.sh` |
| `[pipeline] WARNING: the capture did not reconstruct as one room` | `--matcher exhaustive --force --only colmap` |
| `align` stops with exit code 3 | the capture could not be aligned automatically; the message lists what to check |
| two tables (photoreal and solid) a hand-span apart | `reseat` skipped: it needs `--table-body` and `--table-height` |
| robot model edits do not show up | built scenes inline the robot; rebuild with `--only robot --force`, then `--only wbc --force` |
| `--force` reset a hand-tuned object scale | `--force` regenerates every stage it covers; tune after the last forced rebuild |
| sim prints `WAITING FOR WBC` forever | the controller is on DDS domain 0; see [docs/teleop_dataset.md](teleop_dataset.md#troubleshooting) |
| `Address already in use` | leftovers from a crashed run: `scripts/realsim/start.sh --stop` |
| `new_table_env`: arms drop onto the basket at every reset | the controller was started without `G1_DEFAULT_ELBOW=-0.5`; use the command `start.sh` prints ([why](../README.md#prebuilt-scenes)) |
| `[layout] no layout clear of the robot in 300 tries` | too much on the plate: name three or four objects at most |
| robot's view mostly black | the splat does not cover the robot's viewpoint: film from the robot's eye height too |
| hybrid / VLA windows stay black on the first start, then `TIMEOUT` | gsplat compiling its CUDA kernels inside every renderer at once. `start.sh` now compiles them first (`scripts/build_gsplat.sh`, about a minute, once per machine); update and re-run |
| `--builtin` ends with `no camera 'cap' in scene_g1.xml` | an older checkout: the prebuilt scenes now carry a `cap` camera, and `start.sh` falls back to `scene_view` for scenes without one; `git pull` |
| arms ignore the bent-elbow pose after `git pull` | the WBC checkout still has an older version of our patch: `bash scripts/setup_external.sh --repatch` |
| controller and teleop / exporter / policy loop do not see each other | one of the container shells is missing `-e ROS_LOCALHOST_ONLY=1`; every one needs it |
| dark patches on the plate where no object is | an object left out of the run still casting shadows; fixed in the renderers, update |
| `EGLError` at interpreter exit | teardown noise, harmless |

