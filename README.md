# Curtin CMPE6012 Final Project

## Overview
This repository contains the source for a DeepStream-based animal detection and tracking pipeline that was prepared for the Curtin University CMPE6012 final project. The main application (`ds_track_cam.py`) captures frames from a CSI camera, performs YOLO inference through `nvinfer`, and drives both GPIO-connected indicator lights and an HTTP status endpoint. A second script (`ds_track_cam_with_web.py`) extends the pipeline with a simple web overlay and visualization helpers.

## Repository layout
| File | Description |
| --- | --- |
| `ds_track_cam.py` | Primary DeepStream pipeline with GPIO integration, pause/resume support, and HTTP reporting. |
| `ds_track_cam_with_web.py` | Variant that adds HTML overlay output suitable for dashboards. |
| `pgie_config_yolo.txt` | Sample YOLO primary inference (`pgie`) configuration referenced by the scripts. |
| `metrics.html` | Standalone HTML used to render collected throughput/accuracy metrics.

## Requirements
- Jetson platform with JetPack 5.1.3 (DeepStream 6.3).
- Python 3.8+ with the DeepStream Python bindings available on the device.
- GStreamer, PyGObject (`gi`), and `Jetson.GPIO`.
- YOLO model weights referenced by `pgie_config_yolo.txt` (place them where the config expects).

## Quick start
1. Copy the repository to your Jetson device and ensure the YOLO engine file referenced inside `pgie_config_yolo.txt` exists.
2. Export the `DISPLAY` variable if you plan to use X11 output (`export DISPLAY=:0`).
3. Run the minimal tracker:
   ```bash
   python3 ds_track_cam.py --pgie pgie_config_yolo.txt --tracker tracker_nvdcf_config.yml
   ```
   Replace the tracker path with your preferred NvDCF or IOU config. The script defaults to a CSI camera at 640×480@30 and outputs a muxed 640×640 stream.

## GPIO mapping
- Pin 29 – Cow indicator LED
- Pin 31 – Sheep indicator LED
- Pin 33 – Dog indicator LED
- Pin 32 – Enable switch (toggles `nvinfer` interval to pause/resume inference)

## HTTP status endpoint
`ds_track_cam.py` exposes an HTTP server (default port 8000) with a basic JSON endpoint reporting detected species counts and pipeline FPS. You can query it with:
```bash
curl http://<jetson-ip>:8000/status
```

## Development tips
- Use `tracker_nvdcf_config.yml` or `tracker_iou_config.txt` from the DeepStream samples as a starting point for tuning detection persistence.
- The overlay string in `ds_track_cam.py` shows the current frame number, object count, FPS, and inference state (ON/OFF) derived from the GPIO enable switch.
- When modifying `pgie_config_yolo.txt`, ensure batch size, network mode, and class labels match the model exported in your `.engine` file.

## Troubleshooting
| Symptom | Resolution |
| --- | --- |
| GPIO callbacks never trigger | Confirm that the board pin numbers match your wiring and that `Jetson.GPIO` is run with sufficient permissions (`sudo groupadd -f -r gpio; sudo usermod -a -G gpio $USER`). |
| `gi.repository` import errors | Install PyGObject via the Jetson apt repositories: `sudo apt install python3-gi gir1.2-gst-1.0`. |
| `nvinfer` fails to load engine | Check the `model-engine-file` path in `pgie_config_yolo.txt` and rebuild the engine if switching GPUs or JetPack versions. |

For more background on the DeepStream pipeline structure, consult NVIDIA's [DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) samples, particularly `deepstream-test2` for multi-stream tracking.
