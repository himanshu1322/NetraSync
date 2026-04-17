# NetraSync
NetraSync: A Neural-Visual Hybrid Assistive Interface. A Brain-Computer Interface (BCI) that fuses real time Eye Tracking with EEG Motor Imagery (Cross Attention Mechanism) to empower paralyzed patients with a neural voice using Llama-3.

# NetraSync: Hybrid BCI Assistive Interface

**NetraSync** is a cutting edge research project designed for paralyzed individuals. It integrates **Eye Tracking** and **Electroencephalography (EEG)** signals to provide a "Neural Voice" for patients who cannot move or speak.

## Key Features
- **Hybrid Fusion:** Uses a **Cross Attention Neural Network** to correlate eye gaze with brain intent (Motor Imagery).
- **Neural Voice:** Powered by **Llama-3.2:1b (via Ollama)** to generate natural, context aware speech.
- **Dynamic HUD:** A futuristic Head Up Display showing real-time brainwave streams (8-channels) and gaze targeting.
- **Privacy First:** The system only triggers an action when "Intent" (Brainwaves) matches the "Gaze" (Target), preventing accidental speech.

## The Science
NetraSync identifies a patient's intent by distinguishing between:
1. **Resting State (Run 1):** Patient is scanning the interface; no action is taken.
2. **Motor Imagery (Run 4):** Patient imagines movement to confirm their request.

The system processes **8-channel EEG data** (FC5, FC1, FC2, FC6, CP5, CP1, CP2, CP6) focused on the motor cortex.

## Technology Stack
- **AI/ML:** PyTorch, TensorFlow Lite
- **EEG Processing:** MNE-Python
- **Computer Vision:** MediaPipe, OpenCV
- **LLM:** Ollama (Llama-3.2)
- **Programming:** Python 3.9+

## Interface Preview
- **Emergency Zone:** Dedicated trigger for immediate assistance.
- **Thirst Zone:** Targeted trigger for hydration/food requests.
- **Intent Meter:** Visualizing neural confidence before execution.

## Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/NetraSync.git](https://github.com/yourusername/NetraSync.git)

2. pip install -r requirements.txt
3. ollama serve
4. ollama pull llama3.2:1b
5. python main.py

   
