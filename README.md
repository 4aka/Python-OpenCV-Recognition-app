# Python-OpenCV-Recognition-app

# Gesture Control App

This project uses Python, OpenCV, and Mediapipe to recognize head and hand gestures via webcam and perform system actions (like opening browser tabs, closing windows, or simulating play/pause).

---

## ✅ Requirements
- Python 3.11 installed
- A working webcam (used via `cv2.VideoCapture(0)`)
- Virtual environment recommended

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶ How to Run
1. Open terminal or command prompt.
2. Navigate to the project folder (where `main.py` is located).
3. Run the script:
```bash
python main.py
```

---

## 🤖 What It Does
- Opens your webcam and displays a live window titled **Gesture Control**.
- Detects the following gestures:
  - **Head nod** → triggers play/pause (spacebar)
  - **Head shake** → closes active window (Alt + F4)
  - **Right-hand pinch (thumb + index)** → opens a new browser tab (Ctrl + T)
  - **Left-hand pinch (thumb + index)** → closes active window (Alt + F4)

---

## ⚠️ About Warnings and Logs
When you run the script, you might see logs like:
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
```
These **do not indicate errors** — they come from Mediapipe and TensorFlow Lite’s internal libraries.
You can safely ignore them as long as the app works and detects gestures correctly.

If you want, I can help add logging filters or suppression to hide these messages.

---

## 💡 Tips
- Ensure webcam permissions are granted.
- Use in good lighting for best recognition.
- Press `Esc` to safely exit the app.

---

## 📦 Files
- `main.py` → main application script
- `requirements.txt` → list of required Python packages

Let me know if you want example screenshots or a usage GIF added!
