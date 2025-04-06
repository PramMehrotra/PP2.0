# Phone Usage Detector

A Python application that monitors phone usage through webcam and sends email reports. The program uses YOLO for object detection and Ollama for generating reports.

## Features

- Real-time phone detection using YOLO
- Face detection and attention monitoring
- Automatic email reports with detailed analysis
- Customizable detection thresholds
- Timezone-aware reporting (IST)

## Requirements

- Python 3.8+
- Webcam
- Ollama (for local LLM)
- YOLO model file (yolo11x.pt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd phone-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull the required model:
```bash
ollama pull mistral
```

5. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file with your credentials:
- `EMAIL_USER`: Your Gmail address
- `EMAIL_PASS`: Gmail App Password (not your regular password)
- `REPORT_EMAIL`: Email address where reports will be sent
- `OLLAMA_MODEL`: The Ollama model to use (default: mistral)

## Usage

1. Run the program:
```bash
python phone_detector.py
```

2. Click "Start Detection" to begin monitoring
3. The program will detect phones and faces in real-time
4. Click "Stop Detection" to end the session and receive a report

## Report Format

Reports include:
- Total phone detections
- Session duration
- Analysis of usage patterns
- Recommendations for improvement
- Timestamps in Indian Standard Time (IST)

## Notes

- Make sure your webcam is properly connected and accessible
- The program requires a stable internet connection for email functionality
- Reports are generated using the Mistral model through Ollama
- Detection sensitivity can be adjusted in the code

## License

[Your chosen license] 
