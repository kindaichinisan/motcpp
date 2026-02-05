<p align="center">
  <img src="docs/images/logo.svg" width="400" alt="motcpp">
</p>

<h1 align="center">motcpp</h1>

<p align="center">
  <strong> Modern C++ Multi-Object Tracking Library</strong>
</p>

<div align="center">

<table>
<tr>
<td align="center"><img src="docs/images/demo_bytetrack.gif" width="400" alt="ByteTrack"><br><b>ByteTrack</b></td>
<td align="center"><img src="docs/images/demo_ocsort.gif" width="400" alt="OC-SORT"><br><b>OC-SORT</b></td>
</tr>
<tr>
<td align="center"><img src="docs/images/demo_boosttrack.gif" width="400" alt="BoostTrack"><br><b>BoostTrack</b></td>
<td align="center"><img src="docs/images/demo_sort.gif" width="400" alt="SORT"><br><b>SORT</b></td>
</tr>
</table>

</div>

<p align="center">
  <a href="https://github.com/Geekgineer/motcpp/actions/workflows/ci.yml">
    <img src="https://github.com/Geekgineer/motcpp/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/Geekgineer/motcpp">
    <img src="https://codecov.io/gh/Geekgineer/motcpp/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/releases">
    <img src="https://img.shields.io/github/v/release/Geekgineer/motcpp" alt="Release">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/Geekgineer/motcpp/stargazers">
    <img src="https://img.shields.io/github/stars/Geekgineer/motcpp?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-trackers">Trackers</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

**motcpp** is a high-performance, production-ready C++ library for multi-object tracking. It provides state-of-the-art tracking algorithms with a clean, modern C++17 API designed for real-time applications.

## ✨ Features

- 🎯 **10 State-of-the-Art Trackers** — SORT, ByteTrack, OC-SORT, DeepOC-SORT, StrongSORT, BoT-SORT, BoostTrack, HybridSORT, UCMCTrack, OracleTrack
- ⚡ **Blazing Fast** — Optimized C++ implementation, 10-100x faster than Python equivalents
- 🔧 **Easy Integration** — Modern CMake, single-header option, vcpkg support
- 🧪 **Well Tested** — Comprehensive unit tests with >90% code coverage
- 📦 **Cross-Platform** — Linux, macOS, Windows
- 🔌 **Flexible ReID** — ONNX Runtime backend for appearance embeddings
- 📊 **MOT Benchmark Ready** — Built-in evaluation tools for MOT17/MOT20

## 📊 Benchmarks

### MOT17 Ablation Split

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
|---------|-------|-------|-------|-----|
| [SORT](https://arxiv.org/abs/1602.00763) | 62.4 | 75.2 | 69.2 | **1250** |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | 66.5 | 76.4 | 77.6 | 1100 |
| [OC-SORT](https://arxiv.org/abs/2203.14360) | 64.6 | 73.9 | 74.4 | 850 |
| [UCMCTrack](https://arxiv.org/abs/2312.08952) | 64.0 | 75.6 | 73.9 | 980 |
| [BoostTrack](https://arxiv.org/abs/2408.13003) | **67.5** | **77.1** | **79.2** | 75 |

<sub>Evaluation on the second half of the [MOT17](https://arxiv.org/abs/1603.00831) training set using [YOLOX](https://arxiv.org/abs/2107.08430) detections and [FastReID](https://github.com/JDAI-CV/fast-reid) embeddings. Pre-generated data available in [releases](https://github.com/Geekgineer/motcpp/releases). FPS measured on Intel i9-13900K.</sub>

### C++ vs Python Performance

| Tracker | C++ (FPS) | Python (FPS) | Speedup |
|---------|-----------|--------------|---------|
| ByteTrack | 1100 | 45 | **24x** |
| OC-SORT | 850 | 32 | **27x** |
| StrongSORT | 95 | 8 | **12x** |

## 🚀 Installation

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+
- OpenCV 4.x
- Eigen3

### Build from Source

```bash
git clone https://github.com/Geekgineer/motcpp.git
cd motcpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

### CMake Integration

```cmake
find_package(motcpp REQUIRED)
target_link_libraries(your_target PRIVATE motcpp::motcpp)
```

## 🎮 Quick Start

```cpp
#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // Create tracker
    motcpp::trackers::ByteTrack tracker;
    
    cv::VideoCapture cap("video.mp4");
    cv::Mat frame;
    
    while (cap.read(frame)) {
        // Your detector outputs: [x1, y1, x2, y2, confidence, class_id]
        Eigen::MatrixXf detections = your_detector(frame);
        
        // Update tracker
        Eigen::MatrixXf tracks = tracker.update(detections, frame);
        
        // tracks: [x1, y1, x2, y2, track_id, confidence, class_id, det_index]
        for (int i = 0; i < tracks.rows(); ++i) {
            int track_id = static_cast<int>(tracks(i, 4));
            cv::Rect box(tracks(i, 0), tracks(i, 1), 
                        tracks(i, 2) - tracks(i, 0), 
                        tracks(i, 3) - tracks(i, 1));
            
            cv::rectangle(frame, box, motcpp::BaseTracker::id_to_color(track_id), 2);
        }
        
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }
    
    return 0;
}
```

## 📋 Trackers

| Tracker | Type | Speed | Paper |
|---------|------|-------|-------|
| **SORT** | Motion | ⚡⚡⚡⚡⚡ | [Bewley et al., 2016](https://arxiv.org/abs/1602.00763) |
| **ByteTrack** | Motion | ⚡⚡⚡⚡⚡ | [Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864) |
| **OC-SORT** | Motion | ⚡⚡⚡⚡ | [Cao et al., CVPR 2023](https://arxiv.org/abs/2203.14360) |
| **UCMCTrack** | Motion | ⚡⚡⚡⚡ | [Yi et al., AAAI 2024](https://arxiv.org/abs/2312.08952) |
| **OracleTrack** | Motion | ⚡⚡⚡⚡ | Kalman + CMC + Cascaded Association |
| **DeepOC-SORT** | ReID | ⚡⚡⚡ | [Maggiolino et al., 2023](https://arxiv.org/abs/2302.11813) |
| **StrongSORT** | ReID | ⚡⚡ | [Du et al., TMM 2023](https://arxiv.org/abs/2202.13514) |
| **BoT-SORT** | ReID | ⚡⚡ | [Aharon et al., 2022](https://arxiv.org/abs/2206.14651) |
| **BoostTrack** | ReID | ⚡⚡ | [Stanojevic et al., MVA 2024](https://arxiv.org/abs/2408.13003) |
| **HybridSORT** | ReID | ⚡⚡ | [Yang et al., AAAI 2024](https://arxiv.org/abs/2308.00783) |

### Tracker Selection Guide

```
Need maximum speed?     → SORT, ByteTrack, OracleTrack
General purpose?        → ByteTrack, OC-SORT, OracleTrack
Heavy occlusions?       → OC-SORT, UCMCTrack, OracleTrack
Moving camera?          → UCMCTrack, BoT-SORT, OracleTrack
Re-identification?      → StrongSORT, BoostTrack
State-of-the-art?       → BoostTrack
```

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| [📖 Getting Started](docs/guides/getting-started.md) | Installation and first steps |
| [🔍 API Reference](docs/api/README.md) | Complete API documentation |
| [📝 Tutorials](docs/tutorials/README.md) | Step-by-step guides |
| [💡 Examples](docs/examples/README.md) | Code examples |
| [🎯 Tracker Guide](docs/guides/trackers.md) | Choosing the right tracker |
| [🏗️ Architecture](docs/guides/architecture.md) | System design |
| [📊 Benchmarking](docs/guides/benchmarking.md) | Run your own benchmarks |

## 🧪 Testing

```bash
cmake -B build -DMOTCPP_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 Citation

If you use motcpp in your research, please cite:

```bibtex
@software{motcpp2026,
  author = {motcpp contributors},
  title = {motcpp: Modern C++ Multi-Object Tracking Library},
  year = {2026},
  url = {https://github.com/Geekgineer/motcpp},
  license = {AGPL-3.0}
}
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Special Thanks

This project draws inspiration from and builds upon the excellent work of:

| Project | Description |
|---------|-------------|
| [**BoxMOT**](https://github.com/mikel-brostrom/boxmot) | The original Python multi-object tracking library by Mikel Broström. Our C++ implementation follows similar architecture patterns and algorithm implementations. |

### Tracking Algorithms

| Algorithm | Paper | Authors |
|-----------|-------|---------|
| SORT | [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) | Bewley et al., 2016 |
| ByteTrack | [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864) | Zhang et al., ECCV 2022 |
| OC-SORT | [Observation-Centric SORT](https://arxiv.org/abs/2203.14360) | Cao et al., CVPR 2023 |
| StrongSORT | [StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514) | Du et al., TMM 2023 |
| BoT-SORT | [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651) | Aharon et al., 2022 |
| UCMCTrack | [UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation](https://arxiv.org/abs/2312.08952) | Yi et al., AAAI 2024 |
| BoostTrack | [BoostTrack: Boosting the Similarity Measure and Detection Confidence](https://arxiv.org/abs/2408.13003) | Stanojevic et al., MVA 2024 |
| HybridSORT | [Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking](https://arxiv.org/abs/2308.00783) | Yang et al., AAAI 2024 |

### Benchmark Data & Tools

| Resource | Source | Citation |
|----------|--------|----------|
| MOT17 Dataset | [MOTChallenge](https://arxiv.org/abs/1603.00831) | Milan et al., 2016 |
| YOLOX Detections | [YOLOX](https://arxiv.org/abs/2107.08430) | Ge et al., 2021 |
| ReID Embeddings | [FastReID](https://github.com/JDAI-CV/fast-reid) | He et al., 2020 |

---

<p align="center">
  Made with ❤️ by the motcpp community
</p>
