import os
import gradio as gr
import plotly.graph_objects as go
from inference import VideoSentimentAnalyzer

# ── Initialize the analyzer ──────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
analyzer = VideoSentimentAnalyzer(model_path=MODEL_PATH)


# ── Color palettes ───────────────────────────────────────────────────
EMOTION_COLORS = {
    'anger':    '#EF4444',
    'disgust':  '#A855F7',
    'sadness':  '#3B82F6',
    'joy':      '#F59E0B',
    'neutral':  '#6B7280',
    'surprise': '#F97316',
    'fear':     '#6366F1',
}

SENTIMENT_COLORS = {
    'negative': '#EF4444',
    'neutral':  '#6B7280',
    'positive': '#22C55E',
}


def create_emotion_chart(emotion_confidence):
    """Create a horizontal bar chart of emotion scores."""
    emotions = list(emotion_confidence.keys())
    scores = list(emotion_confidence.values())
    colors = [EMOTION_COLORS.get(e, '#6B7280') for e in emotions]

    # Sort by score descending
    sorted_data = sorted(zip(emotions, scores, colors), key=lambda x: x[1])
    emotions, scores, colors = zip(*sorted_data)

    fig = go.Figure(go.Bar(
        x=scores,
        y=[e.capitalize() for e in emotions],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f'{s:.1%}' for s in scores],
        textposition='outside',
        textfont=dict(size=13, color='#E5E7EB'),
    ))

    fig.update_layout(
        title=dict(text='Emotion Analysis', font=dict(size=18, color='#F9FAFB')),
        xaxis=dict(
            range=[0, 1], showgrid=False, zeroline=False,
            tickformat='.0%', tickfont=dict(color='#9CA3AF'),
        ),
        yaxis=dict(tickfont=dict(size=13, color='#D1D5DB')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=50, b=20),
        height=320,
    )
    return fig


def create_sentiment_chart(sentiment_confidence):
    """Create a gauge-style chart for sentiment."""
    sentiments = list(sentiment_confidence.keys())
    scores = list(sentiment_confidence.values())
    colors = [SENTIMENT_COLORS.get(s, '#6B7280') for s in sentiments]

    fig = go.Figure(go.Bar(
        x=[s.capitalize() for s in sentiments],
        y=scores,
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=8,
        ),
        text=[f'{s:.1%}' for s in scores],
        textposition='outside',
        textfont=dict(size=14, color='#E5E7EB'),
    ))

    fig.update_layout(
        title=dict(text='Sentiment Analysis', font=dict(size=18, color='#F9FAFB')),
        yaxis=dict(
            range=[0, 1], showgrid=False, zeroline=False,
            tickformat='.0%', tickfont=dict(color='#9CA3AF'),
        ),
        xaxis=dict(tickfont=dict(size=14, color='#D1D5DB')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=20),
        height=320,
    )
    return fig


def analyze_video(video_path):
    """Main Gradio handler — analyze a video and return results."""
    if video_path is None:
        return (
            None, None,
            "⚠️ Please upload a video first.",
            "—", "—"
        )

    results = analyzer.analyze(video_path)

    emotion_chart = create_emotion_chart(results['emotion_confidence'])
    sentiment_chart = create_sentiment_chart(results['sentiment_confidence'])

    # Summary card
    summary = (
        f"### {results['emotion_emoji']} Detected Emotion: **{results['predicted_emotion'].capitalize()}** "
        f"({results['emotion_confidence'][results['predicted_emotion']]:.1%})\n\n"
        f"### {results['sentiment_emoji']} Detected Sentiment: **{results['predicted_sentiment'].capitalize()}** "
        f"({results['sentiment_confidence'][results['predicted_sentiment']]:.1%})"
    )

    transcript = results['transcribed_text'] if results['transcribed_text'] else "*No speech detected*"

    return emotion_chart, sentiment_chart, summary, transcript


# ── Custom CSS ────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #8B5CF6, #EC4899, #F97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.02em;
}

.sub-title {
    text-align: center;
    color: #9CA3AF !important;
    font-size: 1.05rem !important;
    margin-top: 4px !important;
    margin-bottom: 20px !important;
}

.result-card {
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: rgba(139, 92, 246, 0.04) !important;
}

footer { display: none !important; }
"""


# ── App Layout ────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet"), css=CUSTOM_CSS) as app:
    gr.Markdown("<h1 class='main-title'>🎬 Video Sentiment Analyzer</h1>")
    gr.Markdown("<p class='sub-title'>Upload a video to detect emotions and sentiment using AI — powered by multimodal deep learning</p>")

    with gr.Row(equal_height=True):
        # Left: Input
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Upload Video",
                sources=["upload"],
                include_audio=True,
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Video",
                variant="primary",
                size="lg",
            )

        # Right: Results
        with gr.Column(scale=1):
            summary_output = gr.Markdown(
                value="Upload a video and click **Analyze** to get started.",
                label="Results",
                elem_classes=["result-card"],
            )
            transcript_output = gr.Textbox(
                label="📝 Transcribed Speech",
                lines=3,
                interactive=False,
            )

    with gr.Row():
        emotion_plot = gr.Plot(label="Emotion Breakdown")
        sentiment_plot = gr.Plot(label="Sentiment Breakdown")

    # ── Wire up ───────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[emotion_plot, sentiment_plot, summary_output, transcript_output],
    )

    gr.Markdown(
        "<p style='text-align:center;color:#6B7280;margin-top:30px;font-size:0.85rem'>"
        "Built with ❤️ using Gradio • Multimodal Model (BERT + R3D-18 + Audio CNN) • Whisper Speech-to-Text"
        "</p>"
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
