import os
from langchain_ollama import OllamaLLM
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

# Configuration
REPORT_EMAIL = os.getenv("REPORT_EMAIL")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def get_ist_time():
    """Get current time in Indian Standard Time (IST)"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def generate_report_content(detections, runtime):
    """Generate a detailed report using Ollama with Mistral model."""
    llm = OllamaLLM(model="mistral")
    
    current_time = get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST")
    
    prompt = f"""
    Create a professional email report about phone usage detection with the following data:
    - Total phone detections: {detections}
    - Total runtime: {runtime}
    - Current time: {current_time}
    
    Format the report with:
    1. A clear subject line
    2. A professional greeting
    3. A summary of the detection data
    4. Analysis of the usage patterns
    5. Recommendations for improvement
    6. A professional closing
    
    Note: All times should be referenced in Indian Standard Time (IST).
    """
    
    response = llm.invoke(prompt)
    return response

def send_report(detections, runtime):
    """Send the generated report via email."""
    try:
        print(f"Preparing to send report...")
        print(f"From: {EMAIL_USER}")
        print(f"To: {REPORT_EMAIL}")
        
        # Generate report content
        report_content = generate_report_content(detections, runtime)
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = REPORT_EMAIL
        msg['Subject'] = f"Phone Detection Report - {get_ist_time().strftime('%Y-%m-%d %H:%M IST')}"
        
        # Add report content
        msg.attach(MIMEText(report_content, 'plain'))
        
        # Send email
        print("Connecting to SMTP server...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            print("Logging in...")
            server.login(EMAIL_USER, EMAIL_PASS)
            print("Sending email...")
            server.send_message(msg)
            print("Email sent successfully!")
            
    except Exception as e:
        print(f"Error sending report: {e}") 