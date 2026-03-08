import torch
from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultimodalTrainer

def test_logging():
    # 1. Create a single mock sample
    mock_sample = {
        'text_input': {
            'input_ids': torch.ones(128, dtype=torch.long),
            'attention_mask': torch.ones(128, dtype=torch.long)
        },
        'video_frames': torch.ones((30, 3, 224, 224)),
        'audio_features': torch.ones((64, 300)),
        'emotion': torch.tensor(0),
        'sentiment': torch.tensor(0)
    }
    
    # 2. Wrap it in a list and then a DataLoader to satisfy 'len(train_loader.dataset)'
    mock_dataset = [mock_sample]
    mock_loader = DataLoader(mock_dataset, batch_size=1)
    
    # 3. Initialize model and trainer
    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    
    # Test training log
    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }
    trainer.log_metrics(train_losses, phase='train')
    
    # Test validation log
    val_losses = {
        'total': 1.5,
        'emotion': 0.5, 
        'sentiment': 1.0
    }
    
    val_metrics = {
        'emotion_precision': 0.65,
        'emotional_accuracy': 0.7,
        'sentiment_precision': 0.75,    
        'sentiment_accuracy': 0.8
    }
    
    trainer.log_metrics(val_losses, val_metrics, phase='val')
    print("Logging test completed successfully!")

if __name__ == "__main__":
    test_logging()