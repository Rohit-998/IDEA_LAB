# ViolenceGuard — Motion-Gated Cascade Detection

Real-time violence detection system using a two-stage cascade approach: a cheap motion-energy gate filters out static footage before passing active segments to a deep learning model trained on the RWF-2000 dataset.

## Architecture

```
┌─────────────┐        /api/*        ┌─────────────────┐
│   Frontend   │ ──────────────────► │     Backend      │
│  (Next.js)   │   SSE + REST        │    (Flask)       │
│  Port 3000   │ ◄────────────────── │   Port 5000      │
└─────────────┘                      └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │  Keras Model     │
                                     │  (keras_model.h5)│
                                     └─────────────────┘
```

## Project Structure

```
├── frontend/              # Next.js dashboard UI
│   ├── app/
│   │   ├── page.tsx       # Main dashboard (upload, timeline, metrics)
│   │   ├── layout.tsx     # Root layout
│   │   └── globals.css    # Styles
│   ├── next.config.ts     # Proxy /api/* → Flask :5000
│   └── package.json
│
├── backend/               # Flask API + inference engine
│   ├── app.py             # REST API, SSE streaming, video segmentation
│   ├── predict_video.py   # Optical flow, model inference, motion gate
│   ├── requirements.txt   # Python dependencies
│   ├── Models/            # Pre-trained Keras model (keras_model.h5)
│   ├── Networks/          # Training notebooks (Flow Gated Network, etc.)
│   ├── Preprocess/        # Data preprocessing notebook
│   └── Images/            # Demo GIFs
│
└── README.md
```

## How It Works

### Stage 1 — Motion-Energy Gate (fast & cheap)
- Samples a few frames at low resolution (56×56)
- Computes optical flow magnitude
- If motion energy is below threshold → skip (no violence in static footage)
- Runs in milliseconds, filters out ~60-70% of segments

### Stage 2 — Deep Model (accurate)
- Only triggered when the gate detects motion
- Processes at full 224×224 resolution with optical flow
- Uses a Flow Gated Network trained on RWF-2000
- Returns violence probability with confidence score

### Video Segmentation
- Long videos are split into 5-second segments (matching RWF-2000 training data)
- Each segment is independently gated and classified
- Results are displayed on an interactive timeline

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip packages: `flask`, `flask-cors`, `psutil`, `numpy`, `opencv-python`, `tensorflow`

### Install & Run

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py                 # Starts on http://localhost:5000

# Frontend (new terminal)
cd frontend
npm install
npm run dev                   # Starts on http://localhost:3000
```

Or use the one-click launcher (Windows):
```bash
start.bat
```

## Dashboard Features

- **Drag-and-drop video upload** — MP4, AVI, MKV supported
- **Motion gate toggle** — compare with/without the gate
- **Real-time SSE streaming** — live progress updates during analysis
- **Segment timeline** — clickable timeline showing per-segment results
- **Process-level CPU tracking** — isolated from background apps
- **Confidence indicators** — color-coded violence probability per segment

## Dataset Citation

Based on the RWF-2000 dataset:

```bibtex
@INPROCEEDINGS{9412502,
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  year={2021},
  pages={4183-4190},
  doi={10.1109/ICPR48806.2021.9412502}}
```

## License

1. Without the approval of the SMIIP Lab, the RWF-2000 database is not allowed to be modified and redistributed.
2. Without the approval of the SMIIP Lab, RWF-2000 database is not allowed for commercial purpose.
3. Images and videos cannot be used in a way that may cause harm to mental health or personal privacy.
