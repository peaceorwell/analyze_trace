# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release

### Features
- Single file GPU kernel analysis
- Dual file comparison analysis
- Web UI with drag-and-drop upload
- Project grouping and management
- Perfetto integration for trace visualization
- `.json.gz` file support (auto-extract on upload, preserve compression on download)
- User isolation via cookie-based session
- Project sharing with password protection

### CLI Options
- `-o, --output-dir` - Output directory
- `-k, --kernel-types` - Custom kernel type keywords
- `-s, --save-triton-csv` - Save per-step Triton kernel details
- `-c, --save-triton-code` - Save Triton kernel source code

### Web Server Options
- `--host` - Listen address
- `--port` - Listen port
- `--no-download` - Disable trace file download

## [0.1.0] - 2024-04-20

### Added
- Initial release with core analysis engine
- Web interface with FastAPI + Vue 3
- SQLite database for job history
- CSV export for all analysis results
- Chart visualization for kernel type breakdown