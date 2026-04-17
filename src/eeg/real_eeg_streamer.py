import mne
import numpy as np

class RealEEGStreamer:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        print("Fetching Intent-Specific EEG Data (PhysioNet Run 4)...")
        
        try:
            # Load Subject 1, Run 4
            data_path = mne.datasets.eegbci.load_data(1, [4], path='./data')[0]
            raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
            
            mne.datasets.eegbci.standardize(raw)
            # Pick the 8 motor channels used in training
            raw.pick(['FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6'])
            
            self.real_brainwaves = raw.get_data().T
            self.total_timesteps = self.real_brainwaves.shape[0]
            
            # Use current_step consistently
            self.current_step = 3200 
            
        except Exception as e:
            print(f"Error loading EEG data: {e}")
            self.real_brainwaves = np.zeros((10000, 8))
            self.current_step = 0

    def get_live_brainwaves(self):
        start_idx = self.current_step
        end_idx = start_idx + self.sequence_length
        
        # Loop logic
        if end_idx >= self.total_timesteps:
            self.current_step = 0
            start_idx = 0
            end_idx = self.sequence_length
            
        chunk = self.real_brainwaves[start_idx:end_idx, :]
        
        # Increment using current_step
        self.current_step += self.sequence_length
        return chunk