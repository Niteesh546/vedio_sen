import os
import subprocess
import tempfile
import cv2
import numpy as np
import torch
import torchaudio
import whisper
from transformers import AutoTokenizer
from training.models import MultimodalSentimentModel


class VideoSentimentAnalyzer:
    """End-to-end video sentiment & emotion analyzer."""

    EMOTION_CLASSES = ['anger', 'disgust', 'sadness', 'joy', 'neutral', 'surprise', 'fear']
    SENTIMENT_CLASSES = ['negative', 'neutral', 'positive']

    EMOTION_EMOJIS = {
        'anger': '😠', 'disgust': '🤢', 'sadness': '😢', 'joy': '😄',
        'neutral': '😐', 'surprise': '😲', 'fear': '😨'
    }
    SENTIMENT_EMOJIS = {'negative': '👎', 'neutral': '😐', 'positive': '👍'}

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load sentiment model
        self.model = MultimodalSentimentModel().to(self.device)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained model from {model_path}")
        else:
            print("No trained model found — running with untrained weights (demo mode)")
        self.model.eval()

        # Load Whisper for speech-to-text
        self.whisper_model = whisper.load_model("base", device=self.device)

        # Load BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def extract_text(self, video_path):
        """Extract text from video audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(video_path, language='en')
            return result.get('text', '').strip()
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return ""

    def extract_video_frames(self, video_path, num_frames=30):
        """Extract and preprocess video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                return torch.zeros((num_frames, 3, 224, 224))

            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            return torch.zeros((num_frames, 3, 224, 224))

        # Pad or truncate to num_frames
        if len(frames) < num_frames:
            padding = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))]
            frames.extend(padding)
        else:
            frames = frames[:num_frames]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def extract_audio_features(self, video_path):
        """Extract mel spectrogram audio features from video."""
        audio_path = None
        try:
            # Extract audio to temp wav file
            audio_path = tempfile.mktemp(suffix='.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            # Pad or truncate to 300 time steps
            if mel_spec.size(2) < 300:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, 300 - mel_spec.size(2)))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec.squeeze(0)  # (64, 300)

        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return torch.zeros((64, 300))
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    def analyze(self, video_path):
        """
        Analyze a video file for emotion and sentiment.
        Returns dict with predictions, confidence scores, and transcribed text.
        """
        # 1. Extract all modalities
        text = self.extract_text(video_path)
        video_frames = self.extract_video_frames(video_path)
        audio_features = self.extract_audio_features(video_path)

        # 2. Tokenize text
        if not text:
            text = "no speech detected"
        text_input = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=128, return_tensors='pt'
        )

        # 3. Run inference
        with torch.inference_mode():
            text_dict = {
                'input_ids': text_input['input_ids'].to(self.device),
                'attention_mask': text_input['attention_mask'].to(self.device)
            }
            video_tensor = video_frames.unsqueeze(0).to(self.device)
            audio_tensor = audio_features.unsqueeze(0).to(self.device)

            outputs = self.model(text_dict, video_tensor, audio_tensor)

            emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0].cpu().numpy()
            sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0].cpu().numpy()

        # 4. Build results
        emotion_results = {
            self.EMOTION_CLASSES[i]: float(emotion_probs[i])
            for i in range(len(self.EMOTION_CLASSES))
        }
        sentiment_results = {
            self.SENTIMENT_CLASSES[i]: float(sentiment_probs[i])
            for i in range(len(self.SENTIMENT_CLASSES))
        }

        predicted_emotion = self.EMOTION_CLASSES[emotion_probs.argmax()]
        predicted_sentiment = self.SENTIMENT_CLASSES[sentiment_probs.argmax()]

        return {
            'transcribed_text': text,
            'predicted_emotion': predicted_emotion,
            'predicted_sentiment': predicted_sentiment,
            'emotion_confidence': emotion_results,
            'sentiment_confidence': sentiment_results,
            'emotion_emoji': self.EMOTION_EMOJIS[predicted_emotion],
            'sentiment_emoji': self.SENTIMENT_EMOJIS[predicted_sentiment]
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <video_path> [model_path]")
        sys.exit(1)

    video_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'

    analyzer = VideoSentimentAnalyzer(model_path=model_file)
    results = analyzer.analyze(video_file)

    print(f"\n{'='*50}")
    print(f"Transcribed Text: \"{results['transcribed_text']}\"")
    print(f"\nPredicted Emotion: {results['emotion_emoji']} {results['predicted_emotion']}")
    print(f"Predicted Sentiment: {results['sentiment_emoji']} {results['predicted_sentiment']}")

    print(f"\n--- Emotion Confidence ---")
    for emotion, score in sorted(results['emotion_confidence'].items(), key=lambda x: -x[1]):
        bar = '█' * int(score * 30)
        print(f"  {emotion:>10}: {bar} {score:.1%}")

    print(f"\n--- Sentiment Confidence ---")
    for sentiment, score in sorted(results['sentiment_confidence'].items(), key=lambda x: -x[1]):
        bar = '█' * int(score * 30)
        print(f"  {sentiment:>10}: {bar} {score:.1%}")
    print(f"{'='*50}")
