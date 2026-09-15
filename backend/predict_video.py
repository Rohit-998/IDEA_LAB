"""
Violence Detection - Video Prediction Script
=============================================
Uses the pre-trained Flow-Gated Network model (keras_model.h5) to predict
whether a given video contains violent or non-violent behavior.

Usage:
    python predict_video.py --video <path_to_video>

Example:
    python predict_video.py --video sample_fight.mp4
"""

import os
import sys
import argparse
import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# Suppress TensorFlow/Keras warnings for cleaner output
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing utilities (matching the repo's Preprocess/Video2Numpy logic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_optical_flow(frames):
    """Calculate dense optical flow between consecutive frames.

    Args:
        frames: numpy array of shape [N, 224, 224, 3] (RGB, uint8).

    Returns:
        flows: numpy array of shape [N, 224, 224, 2] (float32, 0-255 normalized).
    """
    gray_frames = [
        cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1)
        for f in frames
    ]

    flows = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        # Subtract mean to remove camera motion
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # Normalize each component to 0-255
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)

    # Pad the last frame with zeros
    flows.append(np.zeros((224, 224, 2), dtype=np.float32))

    return np.array(flows, dtype=np.float32)


def load_video(video_path, target_size=(224, 224)):
    """Load a video file and return RGB frames + optical flow as a 5-channel tensor.

    Args:
        video_path: path to the video file.
        target_size: (width, height) to resize frames to.

    Returns:
        data: numpy array of shape [N, 224, 224, 5] where channels are [R,G,B, flow_x, flow_y].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video info: {total_frames} frames, {fps:.1f} FPS")

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Video too short — only {len(frames)} frame(s) extracted.")

    frames = np.array(frames, dtype=np.uint8)
    print(f"  Extracted {len(frames)} RGB frames")

    # Compute optical flow
    print("  Computing optical flow ...")
    flows = compute_optical_flow(frames)

    # Merge into 5-channel tensor: [RGB(3) + OpticalFlow(2)]
    data = np.zeros((len(frames), 224, 224, 5), dtype=np.float32)
    data[..., :3] = frames.astype(np.float32)
    data[..., 3:] = flows

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Sampling & normalization (matching the repo's DataGenerator logic)
# ──────────────────────────────────────────────────────────────────────────────

def uniform_sampling(video, target_frames=64):
    """Uniformly sample `target_frames` from the video.

    Args:
        video: numpy array of shape [N, H, W, C].
        target_frames: number of frames the model expects.

    Returns:
        sampled: numpy array of shape [target_frames, H, W, C].
    """
    n = len(video)
    interval = max(1, int(np.ceil(n / target_frames)))

    sampled = []
    for i in range(0, n, interval):
        sampled.append(video[i])

    # Pad if we got fewer frames than needed
    num_pad = target_frames - len(sampled)
    if num_pad > 0:
        for i in range(-num_pad, 0):
            try:
                sampled.append(video[i])
            except IndexError:
                sampled.append(video[0])

    # Truncate if we overshot
    sampled = sampled[:target_frames]

    return np.array(sampled, dtype=np.float32)


def normalize(data):
    """Zero-mean, unit-variance normalization."""
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-6:
        return data - mean
    return (data - mean) / std


def preprocess_for_model(data, target_frames=64):
    """Full preprocessing pipeline: sample → normalize.

    Args:
        data: raw 5-channel video tensor [N, 224, 224, 5].

    Returns:
        ready: tensor shaped [1, 64, 224, 224, 5] (batch dimension added).
    """
    sampled = uniform_sampling(data, target_frames=target_frames)

    # Normalize RGB and optical flow channels separately
    sampled[..., :3] = normalize(sampled[..., :3])
    sampled[..., 3:] = normalize(sampled[..., 3:])

    # Add batch dimension
    return np.expand_dims(sampled, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & prediction
# ──────────────────────────────────────────────────────────────────────────────

def get_rgb(input_x):
    """Extract RGB channels (used by Lambda layers in the saved model)."""
    return input_x[..., :3]


def get_opt(input_x):
    """Extract optical flow channels (used by Lambda layers in the saved model)."""
    return input_x[..., 3:5]


def build_model():
    """Rebuild the Flow-Gated Network architecture from the original source code.

    This avoids Lambda deserialization issues when loading .h5 files across
    different Python/Keras versions.

    Returns:
        model: uncompiled Keras Model with input shape (64, 224, 224, 5).
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv3D, MaxPooling3D, Dense, Flatten, Dropout,
        Multiply, Lambda,
    )

    inputs = Input(shape=(64, 224, 224, 5))

    rgb = Lambda(get_rgb, output_shape=None)(inputs)
    opt = Lambda(get_opt, output_shape=None)(inputs)

    # ── RGB stream ──
    rgb = Conv3D(16, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(16, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(16, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(16, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(32, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(32, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(32, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(32, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    # ── Optical-flow stream ──
    opt = Conv3D(16, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(16, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(16, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(16, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(32, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(32, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(32, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = Conv3D(32, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    # ── Fusion & Pooling ──
    x = Multiply()([rgb, opt])
    x = MaxPooling3D(pool_size=(8,1,1))(x)

    # ── Merging Block ──
    x = Conv3D(64, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(64, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(64, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(64, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(128, (1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(128, (3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,3,3))(x)

    # ── FC Layers ──
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    pred = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=pred)
    return model


def load_model(model_path):
    """Build the model architecture from code, then load saved weights.

    Args:
        model_path: path to the .h5 model file.

    Returns:
        model: compiled Keras model ready for prediction.
    """
    from tensorflow.keras.optimizers import SGD
    import h5py

    print(f"  Loading model from: {model_path}")
    print("  Rebuilding model architecture ...")
    model = build_model()

    # Load only the weights from the .h5 file
    print("  Loading pre-trained weights ...")
    try:
        model.load_weights(model_path)
    except Exception:
        # Fallback: try loading weights by name for partial compatibility
        model.load_weights(model_path, by_name=True)
        print("  WARNING: Loaded weights by name (some layers may be uninitialized)")

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"  Model loaded — input shape: {model.input_shape}")
    return model


def predict(model, preprocessed_data):
    """Run prediction and return class label + confidence.

    Args:
        model: loaded Keras model.
        preprocessed_data: numpy array of shape [1, 64, 224, 224, 5].

    Returns:
        label: "Violence" or "Non-Violence".
        confidence: float between 0 and 1.
    """
    # Model output is softmax over 2 classes: [Fight, NonFight] (alphabetical)
    probs = model.predict(preprocessed_data, verbose=0)
    class_idx = np.argmax(probs[0])

    # Classes in alphabetical order (as the DataGenerator sorts sub-folders)
    classes = ["Fight (Violence)", "NonFight (Non-Violence)"]
    label = classes[class_idx]
    confidence = probs[0][class_idx]

    return label, confidence, probs[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Violence Detection — Predict on a single video using the "
                    "pre-trained Flow-Gated Network model."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file (e.g., .avi, .mp4, .mkv)."
    )
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(os.path.dirname(__file__), "Models", "keras_model.h5"),
        help="Path to the pre-trained .h5 model (default: Models/keras_model.h5)."
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print("=" * 60)
    print("  VIOLENCE DETECTION — VIDEO PREDICTION")
    print("=" * 60)

    # Step 1: Load the model
    print("\n[1/3] Loading model ...")
    model = load_model(args.model)

    # Step 2: Load and preprocess the video
    print(f"\n[2/3] Processing video: {args.video}")
    raw_data = load_video(args.video)
    preprocessed = preprocess_for_model(raw_data)
    print(f"  Preprocessed shape: {preprocessed.shape}")

    # Step 3: Predict
    print("\n[3/3] Running prediction ...")
    label, confidence, probs = predict(model, preprocessed)

    # Print results
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Video    : {os.path.basename(args.video)}")
    print(f"  Label    : {label}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Raw probs : Fight={probs[0]:.4f}  |  NonFight={probs[1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
