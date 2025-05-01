import os
import json
import datetime
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from dotenv import load_dotenv
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"],
    storage_uri="memory://"
)

# Securely get API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("No DeepSeek API key found")
    raise ValueError("No DeepSeek API key found. Please set DEEPSEEK_API_KEY environment variable.")

# Healthcare context
HEALTHCARE_CONTEXT = """
You are HealthMate, a healthcare assistant chatbot designed to provide general health information, 
answer common medical questions, and help users understand basic health concepts.

IMPORTANT GUIDELINES:
- Always include a disclaimer that you're not a replacement for professional medical advice
- For any serious or emergency symptoms, advise the user to contact a healthcare professional immediately
- Do not provide specific diagnoses or treatment recommendations
- Provide factual, evidence-based health information
- Be compassionate, clear, and concise in responses
- If you don't know the answer, admit it and suggest consulting a healthcare provider
- Only answer health and medical-related questions
- Use a friendly yet professional tone
"""

HEALTHCARE_KEYWORDS = [
    'health', 'medical', 'medicine', 'doctor', 'hospital', 'clinic', 'symptom', 'pain',
    'disease', 'illness', 'condition', 'treatment', 'therapy', 'drug', 'medication',
    'prescription', 'diagnosis', 'cancer', 'diabetes', 'heart', 'blood', 'pressure',
    'fever', 'cough', 'headache', 'allergy', 'vaccine', 'vaccination', 'immunization',
    'surgery', 'emergency', 'infection', 'virus', 'bacteria', 'physician', 'nurse',
    'diet', 'nutrition', 'exercise', 'wellness', 'mental health', 'anxiety', 'depression',
    'stress', 'sleep', 'insomnia', 'dental', 'tooth', 'teeth', 'pregnancy', 'prenatal',
    'postnatal', 'pediatric', 'geriatric', 'elderly', 'chronic', 'acute', 'preventive',
    'prevention', 'healthy', 'checkup', 'screening', 'test', 'scan', 'x-ray', 'MRI',
    'CT scan', 'ultrasound', 'pharmacy', 'pharmaceutical', 'side effect', 'dosage',
    'recovery', 'rehabilitation', 'physical therapy', 'occupational therapy', 'specialist',
    'cardiology', 'neurology', 'oncology', 'dermatology', 'orthopedics', 'gynecology',
    'urology', 'pediatrics', 'psychology', 'psychiatry', 'therapy', 'counseling', 'dizziness',
    'headache', 'stomachache', 'cold', 'tongue', 'throat', 'pain', 'swelling', 'eye', 'ear', 
    'BP', 'strain'
]

# Conversation storage
conversation_store = {}

def validate_input(f):
    """Decorator to validate input data"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        if not data or 'message' not in data or not isinstance(data['message'], str):
            logger.warning("Invalid input received")
            return jsonify({'error': 'Invalid or missing message'}), 400
        if len(data['message'].strip()) > 1000:  # Limit message length
            logger.warning("Message too long")
            return jsonify({'error': 'Message is too long'}), 400
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

def is_healthcare_related(message):
    """Check if message is healthcare-related"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in HEALTHCARE_KEYWORDS)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("5 per minute")  # Rate limit per IP
@validate_input
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data['message'].strip()
    user_id = data.get('user_id', 'default_user')
    
    # Check for emergency keywords
    emergency_keywords = ['heart attack', 'stroke', 'can\'t breathe', 'suicide', 'emergency', 'severe bleeding']
    if any(keyword in user_message.lower() for keyword in emergency_keywords):
        logger.info(f"Emergency detected for user {user_id}")
        return jsonify({
            'response': "🚨 EMERGENCY DETECTED: Please call emergency services (911 in the US) immediately. This chatbot cannot assist with emergencies."
        })
    
    # Check if query is healthcare-related
    if not is_healthcare_related(user_message):
        logger.info(f"Non-healthcare query from user {user_id}")
        return jsonify({
            'response': "I'm HealthMate, designed to answer health-related questions only. Please ask about health, medicine, or wellness."
        })
    
    # Initialize conversation history
    if user_id not in conversation_store:
        conversation_store[user_id] = []
    
    # Add user message to history
    conversation_store[user_id].append({"role": "user", "content": user_message, "timestamp": datetime.datetime.now().isoformat()})
    
    # Prepare messages for API
    messages = [{"role": "system", "content": HEALTHCARE_CONTEXT}] + conversation_store[user_id][-5:]
    
    try:
        response = query_deepseek(messages)
        assistant_message = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not assistant_message:
            logger.warning(f"Empty response from DeepSeek API for user {user_id}")
            assistant_message = "I couldn't generate a response. Please try rephrasing your question."
        
        # Add disclaimer
        if "not a replacement for professional medical advice" not in assistant_message.lower():
            assistant_message += "\n\n⚠️ Reminder: This is general information only. Consult a healthcare provider for professional medical advice."
        
        # Add assistant message to history
        conversation_store[user_id].append({"role": "assistant", "content": assistant_message, "timestamp": datetime.datetime.now().isoformat()})
        
        logger.info(f"Successful response for user {user_id}")
        return jsonify({'response': assistant_message})
    
    except Exception as e:
        logger.error(f"Error processing request for user {user_id}: {str(e)}")
        return jsonify({
            'response': "Sorry, I encountered an error. Please try again later."
        }), 500

def query_deepseek(messages):
    """Query DeepSeek API via OpenRouter"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:5000"),
        "X-Title": "HealthMate"
    }
    
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000  # Added to limit response length
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"DeepSeek API request failed: {str(e)}")
        raise

@app.route('/api/clear_history', methods=['POST'])
@limiter.limit("2 per minute")
@validate_input
def clear_history():
    """Clear conversation history"""
    data = request.json
    user_id = data.get('user_id', 'default_user')
    
    if user_id in conversation_store:
        conversation_store[user_id] = []
        logger.info(f"Conversation history cleared for user {user_id}")
    
    return jsonify({'status': 'success', 'message': 'Conversation history cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)