# AIRA Emotion Detection Dashboard

A comprehensive Streamlit-based dashboard for monitoring emotion detection and system diagnostics of the AIRA robo dog companion designed for elderly care.

## 🚀 Features

- **Live Emotion Monitoring**: Real-time primary emotion detection with confidence scores
- **Input Diagnostics**: Video and audio signal quality monitoring
- **Emotion Analytics**: Detailed breakdown of all emotion scores (Happy, Sad, Fear, Angry, Disgust, Neutral)
- **Signal Trends**: Historical visualization of video/audio signal quality
- **Interactive Charts**: Responsive Plotly visualizations for easy data exploration
- **Feed Status**: Real-time indicator of video and audio feed status

## 📋 Project Structure

```
├── app.py                    # Main Streamlit application
├── data/
│   └── emotion_data.csv     # Demo dataset for testing
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Dashboard

Start the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## 📊 Dashboard Sections

### Live Monitoring
- Displays the current detected emotion
- Shows confidence percentage with visual gauge
- Indicates video and audio feed status

### Input Diagnostics
- Video Signal Quality gauge (0-100%)
- Audio Signal Quality gauge (0-100%)
- Historical signal quality trends

### Emotion Analysis
- Current emotion distribution across all emotion categories
- Timeline visualization showing emotion score changes over time
- Raw data viewer for detailed inspection

## 📈 Demo Data

The `emotion_data.csv` file contains 30 sample data points with:
- Timestamps (5-minute intervals on March 31, 2026)
- Primary emotion detection
- Confidence scores
- Individual emotion scores for all categories
- Signal quality metrics for video and audio feeds
- Feed active/inactive status

## 🔮 Future Enhancements

- Integration with live AIRA system data
- Maintenance log tracking
- Development metrics and bug tracking
- Historical data export and reporting
- Multi-unit dashboard support
- Real-time alerts and notifications

## 👥 For Developers

This dashboard is designed to help developers:
- Monitor the emotional AI accuracy
- Track system health and signal quality
- Identify patterns and anomalies in emotion detection
- Debug and troubleshoot sensor inputs
- Maintain and improve the AIRA companion features

## 📝 License

This project is part of the AIRA Robo Dog Companion initiative.

---

**Version**: 1.0  
**Last Updated**: March 31, 2026
