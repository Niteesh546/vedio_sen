from models import MultimodalSentimentModel

def count_parameters(model):
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,   
        'audio_encoder': 0,
        'fusion_module': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total_params += num
            
            if 'text_encoder' in name:
                params_dict['text_encoder'] += num
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += num
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += num
            elif 'fusion_layer' in name:
                params_dict['fusion_module'] += num
            elif 'emo_classifier' in name:
                params_dict['emotion_classifier'] += num
            elif 'sentiment_classifier' in name:
                params_dict['sentiment_classifier'] += num
                
    return params_dict, total_params

if __name__ == "__main__":
    # Initialize model
    model = MultimodalSentimentModel()
    
    # Get counts
    param_dics, total = count_parameters(model)
    
    print("\n" + "="*40)
    print(f"{'Component':<25} | {'Parameters':<15}")
    print("-" * 40)
    for component, count in param_dics.items():
        print(f"{component:<25} | {count:>15,}")
    print("-" * 40)
    print(f"{'Total Trainable':<25} | {total:>15,}")
    print("="*40 + "\n")