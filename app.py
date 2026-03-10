import os
import gradio as gr
import plotly.graph_objects as go
from inference import VideoSentimentAnalyzer

# ── Initialize the analyzer ──────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
analyzer = VideoSentimentAnalyzer(model_path=MODEL_PATH)


# ── Cinematic color palette ──────────────────────────────────────────
EMOTION_COLORS = {
    'anger':    '#E74C3C',
    'disgust':  '#8E44AD',
    'sadness':  '#5DADE2',
    'joy':      '#F4D03F',
    'neutral':  '#7F8C8D',
    'surprise': '#E67E22',
    'fear':     '#9B59B6',
}

SENTIMENT_COLORS = {
    'negative': '#E74C3C',
    'neutral':  '#7F8C8D',
    'positive': '#2ECC71',
}


def create_emotion_chart(emotion_confidence):
    """Create a cinematic horizontal bar chart of emotion scores."""
    emotions = list(emotion_confidence.keys())
    scores = list(emotion_confidence.values())
    colors = [EMOTION_COLORS.get(e, '#7F8C8D') for e in emotions]

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
        textfont=dict(size=13, color='#BDC3C7', family='Bebas Neue, Inter, sans-serif'),
    ))

    fig.update_layout(
        title=dict(
            text='<b>🎭 Emotion Breakdown</b>',
            font=dict(size=17, color='#F5E6CA', family='Bebas Neue, Inter, sans-serif'),
            x=0.02,
        ),
        xaxis=dict(
            range=[0, 1], showgrid=False, zeroline=False,
            tickformat='.0%', tickfont=dict(color='#7F8C8D', family='Inter'),
        ),
        yaxis=dict(tickfont=dict(size=13, color='#BDC3C7', family='Inter')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=50, b=20),
        height=340,
    )
    return fig


def create_sentiment_chart(sentiment_confidence):
    """Create a cinematic bar chart for sentiment."""
    sentiments = list(sentiment_confidence.keys())
    scores = list(sentiment_confidence.values())
    colors = [SENTIMENT_COLORS.get(s, '#7F8C8D') for s in sentiments]

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
        textfont=dict(size=14, color='#BDC3C7', family='Bebas Neue, Inter, sans-serif'),
        width=0.45,
    ))

    fig.update_layout(
        title=dict(
            text='<b>🎬 Sentiment Breakdown</b>',
            font=dict(size=17, color='#F5E6CA', family='Bebas Neue, Inter, sans-serif'),
            x=0.02,
        ),
        yaxis=dict(
            range=[0, 1], showgrid=False, zeroline=False,
            tickformat='.0%', tickfont=dict(color='#7F8C8D', family='Inter'),
        ),
        xaxis=dict(tickfont=dict(size=14, color='#BDC3C7', family='Inter')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=20),
        height=340,
    )
    return fig


def analyze_video(video_path):
    """Main Gradio handler — analyze a video and return results."""
    if video_path is None:
        return (
            None, None,
            "⚠️ Please upload a video first.",
            "—"
        )

    results = analyzer.analyze(video_path)

    emotion_chart = create_emotion_chart(results['emotion_confidence'])
    sentiment_chart = create_sentiment_chart(results['sentiment_confidence'])

    emo = results['predicted_emotion']
    sent = results['predicted_sentiment']
    emo_conf = results['emotion_confidence'][emo]
    sent_conf = results['sentiment_confidence'][sent]

    summary = (
        f"<div class='result-inner'>"
        f"<div class='result-row'>"
        f"<span class='result-emoji'>{results['emotion_emoji']}</span>"
        f"<div><span class='result-label'>Detected Emotion</span>"
        f"<span class='result-value'>{emo.capitalize()}</span>"
        f"<span class='result-conf'>{emo_conf:.1%}</span></div>"
        f"</div>"
        f"<div class='result-divider'></div>"
        f"<div class='result-row'>"
        f"<span class='result-emoji'>{results['sentiment_emoji']}</span>"
        f"<div><span class='result-label'>Detected Sentiment</span>"
        f"<span class='result-value'>{sent.capitalize()}</span>"
        f"<span class='result-conf'>{sent_conf:.1%}</span></div>"
        f"</div>"
        f"</div>"
    )

    transcript = results['transcribed_text'] if results['transcribed_text'] else "*No speech detected*"

    return emotion_chart, sentiment_chart, summary, transcript


# ── Cinematic CSS ─────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ────── Global Reset ────── */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container, .main, .wrap, .contain {
    background: #0A0A0A !important;
    color: #D4C5A9 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* ────── Cinematic Ambient Glow ────── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 700px 500px at 15% 10%, rgba(218, 165, 32, 0.07), transparent),
        radial-gradient(ellipse 500px 400px at 85% 85%, rgba(231, 76, 60, 0.05), transparent),
        radial-gradient(ellipse 400px 300px at 50% 40%, rgba(245, 230, 202, 0.03), transparent);
    pointer-events: none;
    z-index: 0;
    animation: ambientShift 16s ease-in-out infinite alternate;
}

@keyframes ambientShift {
    0%   { opacity: 0.6; }
    50%  { opacity: 1;   }
    100% { opacity: 0.75; }
}

/* Film-grain overlay for cinematic texture */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.015'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
}

/* ────── Cinematic Header ────── */
.hero-title {
    text-align: center;
    font-family: 'Bebas Neue', 'Inter', sans-serif !important;
    font-size: 3.5rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #DAA520 0%, #F5E6CA 40%, #E8C872 60%, #DAA520 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
    padding-top: 20px !important;
    text-shadow: 0 0 40px rgba(218, 165, 32, 0.15);
    animation: goldShimmer 6s ease-in-out infinite alternate;
}

@keyframes goldShimmer {
    0%   { filter: brightness(1);    }
    50%  { filter: brightness(1.12); }
    100% { filter: brightness(1.05); }
}

.hero-sub {
    text-align: center;
    color: #7F8C8D !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    margin-top: 6px !important;
    margin-bottom: 28px !important;
    letter-spacing: 0.04em;
}

.hero-sub em {
    color: #D4C5A9 !important;
    font-style: normal;
    font-weight: 500;
}

/* ────── Cinematic Glass Cards ────── */
.glass-card,
.gr-panel,
.gr-box,
.gr-form,
.block {
    background: rgba(18, 18, 18, 0.85) !important;
    backdrop-filter: blur(20px) saturate(1.2) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.2) !important;
    border: 1px solid rgba(218, 165, 32, 0.08) !important;
    border-radius: 12px !important;
    box-shadow:
        0 4px 30px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(245, 230, 202, 0.03) !important;
    transition: border-color 0.4s ease, box-shadow 0.4s ease !important;
}

.glass-card:hover,
.gr-panel:hover {
    border-color: rgba(218, 165, 32, 0.15) !important;
    box-shadow:
        0 8px 40px rgba(218, 165, 32, 0.06),
        inset 0 1px 0 rgba(245, 230, 202, 0.03) !important;
}

/* Make ALL inner containers dark */
.gr-group, .gr-box, .gr-padded, .gr-compact,
div[class*="block"], div[class*="wrap"],
div[class*="panel"], div[class*="form"] {
    background: rgba(18, 18, 18, 0.85) !important;
    border-color: rgba(218, 165, 32, 0.06) !important;
}

/* ────── Section Labels ────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #DAA520 !important;
    margin-bottom: 8px !important;
    margin-top: 4px !important;
}

.section-label .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #DAA520;
    box-shadow: 0 0 8px rgba(218, 165, 32, 0.4);
    animation: dotPulse 3s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 0.4; box-shadow: 0 0 4px rgba(218, 165, 32, 0.2); }
    50%      { opacity: 1;   box-shadow: 0 0 12px rgba(218, 165, 32, 0.6); }
}

/* ────── Video Upload Area ────── */
.upload-area {
    border: 1.5px dashed rgba(218, 165, 32, 0.2) !important;
    border-radius: 12px !important;
    background: rgba(12, 12, 12, 0.6) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    min-height: 260px !important;
}

.upload-area:hover {
    border-color: rgba(218, 165, 32, 0.45) !important;
    background: rgba(218, 165, 32, 0.03) !important;
    box-shadow: 0 0 40px rgba(218, 165, 32, 0.06) !important;
}

/* ────── Analyze Button — Cinematic Gold ────── */
.analyze-btn {
    background: linear-gradient(135deg, #8B6914, #DAA520, #C49B17) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em;
    color: #0A0A0A !important;
    box-shadow:
        0 4px 20px rgba(218, 165, 32, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

.analyze-btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.25) 50%, transparent 70%);
    transform: translateX(-150%);
    transition: transform 0.7s ease;
}

.analyze-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow:
        0 8px 35px rgba(218, 165, 32, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    filter: brightness(1.1);
}

.analyze-btn:hover::before {
    transform: translateX(150%);
}

.analyze-btn:active {
    transform: translateY(-1px) scale(0.98) !important;
}

/* ────── Result Card ────── */
.result-card {
    background: rgba(14, 14, 14, 0.9) !important;
    border: 1px solid rgba(218, 165, 32, 0.12) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    backdrop-filter: blur(16px) !important;
    transition: all 0.3s ease !important;
}

.result-card:hover {
    border-color: rgba(218, 165, 32, 0.25) !important;
    box-shadow: 0 0 30px rgba(218, 165, 32, 0.04) !important;
}

.result-inner {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.result-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.result-emoji {
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 2px 12px rgba(218, 165, 32, 0.2));
}

.result-label {
    display: block;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #7F8C8D;
    margin-bottom: 2px;
}

.result-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F5E6CA;
    margin-right: 10px;
}

.result-conf {
    font-size: 0.82rem;
    font-weight: 600;
    color: #DAA520;
    background: rgba(218, 165, 32, 0.1);
    padding: 2px 10px;
    border-radius: 6px;
    border: 1px solid rgba(218, 165, 32, 0.15);
}

.result-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(218, 165, 32, 0.15), transparent);
}

/* ────── Transcript & Text inputs ────── */
textarea, .gr-textbox textarea, input, .gr-textbox input {
    background: rgba(12, 12, 12, 0.8) !important;
    border: 1px solid rgba(218, 165, 32, 0.08) !important;
    border-radius: 10px !important;
    color: #BDC3C7 !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.3s ease !important;
}

textarea:focus, input:focus {
    border-color: rgba(218, 165, 32, 0.25) !important;
    box-shadow: 0 0 0 2px rgba(218, 165, 32, 0.06) !important;
}

/* placeholder text */
textarea::placeholder, input::placeholder {
    color: #555 !important;
}

/* ────── Labels ────── */
label, .gr-input-label, .label-wrap span {
    color: #8B7D6B !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ────── Plot Containers ────── */
.gr-plot, .plot-container {
    background: rgba(14, 14, 14, 0.8) !important;
    border: 1px solid rgba(218, 165, 32, 0.06) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color 0.3s ease !important;
}

.gr-plot:hover {
    border-color: rgba(218, 165, 32, 0.15) !important;
}

/* ────── Chart Divider ────── */
.chart-divider {
    text-align: center;
    margin: 32px 0 16px 0 !important;
    color: #5A5040 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    font-weight: 400;
}

.chart-divider::before, .chart-divider::after {
    content: '';
    display: inline-block;
    width: 100px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(218, 165, 32, 0.15), transparent);
    vertical-align: middle;
    margin: 0 16px;
}

/* ────── Tech Badges — cinematic style ────── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 0 0 28px 0 !important;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    background: rgba(218, 165, 32, 0.06);
    border: 1px solid rgba(218, 165, 32, 0.1);
    color: #8B7D6B;
    transition: all 0.3s ease;
}

.badge:hover {
    border-color: rgba(218, 165, 32, 0.25);
    color: #D4C5A9;
    background: rgba(218, 165, 32, 0.1);
}

.badge .badge-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
}

/* ────── Footer ────── */
.app-footer {
    text-align: center;
    color: #3A3530 !important;
    font-size: 0.75rem !important;
    padding: 32px 0 12px 0 !important;
    letter-spacing: 0.04em;
}

.app-footer a {
    color: #DAA520 !important;
    text-decoration: none;
    transition: color 0.3s ease;
}

.app-footer a:hover {
    color: #F5E6CA !important;
}

/* ────── Scrollbar ────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(218, 165, 32, 0.12);
    border-radius: 3px;
}

/* ────── Hide default Gradio footer ────── */
footer { display: none !important; }

/* ────── Responsive ────── */
@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem !important; }
    .badge-row { gap: 6px; }
    .badge { font-size: 0.62rem; padding: 3px 8px; }
}
"""


# ── App Layout ────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(
    theme=THEME,
    css=CUSTOM_CSS,
    title="🎬 Video Sentiment Analyzer",
) as app:

    # ─── Cinematic Hero ──────────────────────────────────────────────
    gr.HTML(
        "<h1 class='hero-title'>🎬 VIDEO SENTIMENT ANALYZER</h1>"
        "<p class='hero-sub'>Upload a video to detect <em>emotions</em> and <em>sentiment</em> — "
        "powered by multimodal deep learning</p>"
    )

    # ─── Tech badges ─────────────────────────────────────────────────
    gr.HTML(
        "<div class='badge-row'>"
        "<span class='badge'><span class='badge-dot' style='background:#DAA520'></span>BERT NLP</span>"
        "<span class='badge'><span class='badge-dot' style='background:#E74C3C'></span>R3D-18 Vision</span>"
        "<span class='badge'><span class='badge-dot' style='background:#E67E22'></span>Audio CNN</span>"
        "<span class='badge'><span class='badge-dot' style='background:#2ECC71'></span>Whisper STT</span>"
        "</div>"
    )

    # ─── Main Row ────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        # Left — input
        with gr.Column(scale=1):
            gr.HTML("<div class='section-label'><span class='dot'></span>Input</div>")
            video_input = gr.Video(
                label="Upload Video",
                sources=["upload"],
                include_audio=True,
                elem_classes=["upload-area"],
            )
            analyze_btn = gr.Button(
                "🎬  ANALYZE VIDEO",
                variant="primary",
                size="lg",
                elem_classes=["analyze-btn"],
            )

        # Right — results
        with gr.Column(scale=1):
            gr.HTML("<div class='section-label'><span class='dot'></span>Results</div>")
            summary_output = gr.HTML(
                value=(
                    "<div class='result-inner' style='text-align:center;padding:44px 20px;color:#5A5040;'>"
                    "<p style='font-size:2.8rem;margin:0;'>🎬</p>"
                    "<p style='margin:10px 0 0 0;font-size:0.88rem;letter-spacing:0.03em;'>"
                    "Upload a video and click <b style=\"color:#DAA520;\">Analyze</b> to begin</p>"
                    "</div>"
                ),
                elem_classes=["result-card"],
            )
            transcript_output = gr.Textbox(
                label="📝 Transcribed Speech",
                lines=3,
                interactive=False,
                placeholder="Transcript will appear here...",
            )

    # ─── Charts ──────────────────────────────────────────────────────
    gr.HTML("<div class='chart-divider'>Detailed Analysis</div>")

    with gr.Row():
        emotion_plot = gr.Plot(label="Emotion Breakdown")
        sentiment_plot = gr.Plot(label="Sentiment Breakdown")

    # ─── Wire up ─────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[emotion_plot, sentiment_plot, summary_output, transcript_output],
    )

    # ─── Footer ──────────────────────────────────────────────────────
    gr.HTML(
        "<p class='app-footer'>"
        "Built with ❤️ using <a href='https://gradio.app' target='_blank'>Gradio</a> · "
        "Multimodal Model (BERT + R3D-18 + Audio CNN) · Whisper Speech-to-Text"
        "</p>"
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
