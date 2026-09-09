# MuGS Project Structure

Clean, hierarchical organization for maintainability and clarity.

## Root Directory

```
mugs/
├── README.md                  # Project overview and quick start
├── LICENSE                    # Apache-2.0 license
├── pyproject.toml             # Main package config (mugs)
├── pyproject_mugs_mjlab.toml  # MJLab integration package config
├── pytest.ini                 # Test configuration
│
├── src/                       # Source code
│   ├── mugs/                  # Main package (standalone)
│   └── mugs_mjlab/            # MJLab integration package
│
├── tests/                     # Test suite
├── examples/                  # Example scripts and demos
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
├── assets/                    # Small demo assets (in Git)
└── data/                      # Large assets (gitignored, download separately)
```

## Source Code (`src/`)

```
src/
├── mugs/                      # Standalone package
│   ├── sensors/               # Rendering sensors
│   │   ├── gaussian_sensor.py
│   │   └── base.py
│   ├── utils/                 # Utilities
│   │   ├── rendering.py
│   │   └── mask_config.py
│   ├── postprocess/           # Post-processing
│   │   └── super_resolution.py
│   ├── assets/                # Asset management
│   ├── scene_gen/             # Scene generation
│   └── sr_models/             # SR model definitions
│
└── mugs_mjlab/                # MJLab integration
    └── sensors/               # MJLab-compatible sensors
        └── gaussian_sensor.py
```

## Examples (`examples/`)

```
examples/
├── basic/                     # Basic usage examples
├── demo/                      # Full demonstrations
│   └── kitchen_scene_demo.py
├── advanced/                  # Advanced features
├── configs/                   # Example configurations
│   ├── envs/                  # Environment configs
│   ├── sensors/               # Sensor configs
│   └── sr/                    # Super-resolution configs
├── scenes/                    # Example MuJoCo scenes
│   └── first_person_kitchen.xml
└── *.py                       # Standalone demo scripts
```

## Documentation (`docs/`)

```
docs/
├── API_REFERENCE.md           # Complete API documentation
├── API_QUICKSTART.md          # Quick start guide (中文)
├── OVERVIEW.md                # Project overview
├── PROJECT_STATUS.md          # Current status
│
├── guides/                    # User guides
│   ├── QUICK_START.md
│   ├── ASSET_ACQUISITION.md
│   └── SEGMENT_ID_SYSTEM.md
│
├── design/                    # Design documents
│   ├── DESIGN.md              # System architecture (12k words)
│   └── PROJECT_ARCHITECTURE.md
│
├── technical/                 # Technical details
│   └── COORDINATE_ALIGNMENT.md
│
├── development/               # Development docs
│   ├── PROJECT_MANIFEST.md
│   ├── PROJECT_OVERVIEW.md
│   ├── TODO.md
│   └── TEST_REPORT.md
│
└── images/                    # Documentation images
    ├── showcase.jpg
    └── showcase/              # Showcase materials
        └── androidtwin_g1/
```

## Assets (`assets/`)

Small demonstration assets included in Git (~2MB total).

```
assets/
├── README.md                  # Asset documentation
├── configs/                   # Configuration files
│   └── mask_config_kitchen.yaml
├── objects/                   # Individual 3DGS objects
│   ├── demo_kitchen/          # Kitchen demo objects
│   └── misc/                  # Miscellaneous
├── scenes/                    # Complete scenes
│   └── demo_kitchen/          # 12-object kitchen scene
└── models/                    # MuJoCo models
    └── mjcf/                  # MJCF model files
```

## Data (`data/`)

Large assets (gitignored, download via scripts).

```
data/
├── README.md                  # Download instructions
├── pretrained/                # Pretrained 3DGS scenes
│   └── kitchen/               # INRIA kitchen (5.2GB)
├── external/                  # External datasets
│   ├── DISCOVERSE/            # DISCOVERSE dataset
│   └── gs-playground/         # GS-Playground scenes
└── custom/                    # User custom scenes
```

## Scripts (`scripts/`)

```
scripts/
├── download_external_assets.py    # Download large assets
├── download_sr_models.py          # Download SR models
├── debug/                         # Debug utilities
│   └── debug_camera.py
├── data_collection/               # Data collection tools
├── training/                      # Training scripts
├── evaluation/                    # Evaluation tools
└── utils/                         # Script utilities
```

## Tests (`tests/`)

```
tests/
├── unit/                      # Unit tests
├── integration/               # Integration tests
├── conftest.py                # Pytest configuration
└── test_*.py                  # Test modules
```

## Key Principles

1. **Separation of Concerns**
   - `src/mugs/`: Core functionality, no mjlab dependency
   - `src/mugs_mjlab/`: MJLab integration only
   - `assets/`: Small demo files (Git)
   - `data/`: Large files (download separately)

2. **Clear Hierarchy**
   - Flat root directory (10 items)
   - Logical grouping of related files
   - Consistent naming conventions

3. **Documentation Co-location**
   - READMEs in each major directory
   - Images near their documentation
   - Examples with their configs

4. **Gitignore Strategy**
   - Source code: tracked
   - Small assets: tracked
   - Large data: ignored (download scripts provided)
   - Build artifacts: ignored
   - Test outputs: ignored
