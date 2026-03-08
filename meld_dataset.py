import os
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torchaudio
os.environ['TokenizerParallelism'] = 'false'

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        # Added check to help you debug the FileNotFoundError
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {os.path.abspath(csv_path)}")
            
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'sadness': 2, 'joy': 3, 
            'neutral': 4, 'surprise': 5, 'fear': 6
        }
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }
        
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            if not cap.isOpened():
                return torch.zeros((30, 3, 224, 224))
            
            while len(frames) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
                
        finally:
            cap.release()
            
        if len(frames) == 0:
            return torch.zeros((30, 3, 224, 224))
        
        if len(frames) < 30:
            padding = [np.zeros_like(frames[0]) for _ in range(30 - len(frames))]
            frames.extend(padding) 
        else:
            frames = frames[:30]
            
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
     
    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')
        
        try:
            # 1. Extract audio if not exists
            if not os.path.exists(audio_path):
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', video_path, 
                    '-vn', 
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            
            # 2. LOAD AND PROCESS (Moved out of except block)
            waveform, sample_rate = torchaudio.load(audio_path) 
             
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
            
            # Fix padding logic
            if mel_spec.size(2) < 300:
                padding_size = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding_size))
            else:
                mel_spec = mel_spec[:, :, :300]
            
            # Optionally remove temp wav file
            # os.remove(audio_path) 
            
            return mel_spec.squeeze(0) # Returns (64, 300)
                
        except Exception as e:
            # Return zero tensor if extraction/processing fails
            return torch.zeros((64, 300))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        row = self.data.iloc[idx]
        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            path = os.path.join(self.video_dir, video_filename)
            
            if not os.path.exists(path):
                return self.__getitem__((idx + 1) % len(self))
            
            text_input = self.tokenizer(
                str(row['Utterance']),
                padding='max_length',
                truncation=True, 
                max_length=128,
                return_tensors='pt'
            )
            
            audio_features = self._extract_audio_features(path)
            video_frames = self._load_video_frames(path)    
            
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
            
            return {
                'text_input': {
                    'input_ids': text_input['input_ids'].squeeze(),
                    'attention_mask': text_input['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion': torch.tensor(emotion_label),
                'sentiment': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
        
def prepare_dataloader(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    # Check your folder names! Are they 'train-splits' or 'train_splits'?
    train_loader, dev_loader, test_loader = prepare_dataloader(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )
    for batch in train_loader:
        print("Text Input IDs shape:", batch['text_input']['input_ids'].shape)
        print("Video Frames shape:", batch['video_frames'].shape)
        print("Audio Features shape:", batch['audio_features'].shape)
        print("Emotion Label shape:", batch['emotion'].shape)
        print("Sentiment Label shape:", batch['sentiment'].shape)
        break