# app.py - Main application file
# import os
# import json
# import datetime
# import re
# from flask import Flask, request, jsonify, render_template
# import requests
# from dotenv import load_dotenv

# # Import NLP libraries
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# import pickle
# import os.path

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)

# # Securely get API key from environment variables
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# if not DEEPSEEK_API_KEY:
#     raise ValueError("No DeepSeek API key found. Please set DEEPSEEK_API_KEY environment variable.")

# # Healthcare-specific knowledge and context
# HEALTHCARE_CONTEXT = """
# You are a healthcare assistant chatbot designed to provide general health information, 
# answer common medical questions, and help users understand basic health concepts.

# IMPORTANT GUIDELINES:
# - Always include a disclaimer that you're not a replacement for professional medical advice
# - For any serious or emergency symptoms, advise the user to contact a healthcare professional
# - Don't provide specific diagnosis or treatment recommendations
# - Focus on providing factual, evidence-based health information
# - Be compassionate and clear in your responses
# - If you don't know the answer, say so rather than providing potentially incorrect information
# """

# # Standard response for non-medical questions
# NON_MEDICAL_RESPONSE = """I'm a healthcare assistant designed to answer medical and health-related questions only. 
# I'm not able to provide information about {topic}. 

# Please feel free to ask me about health, medicine, wellness, symptoms, treatments, or other medical topics where I can be most helpful."""

# # Track conversation history
# conversation_store = {}

# # Initialize NLTK components if needed
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# # Path to the saved classifier model
# MODEL_PATH = "medical_classifier.pkl"

# # Sample training data for the classifier - expanded list for better accuracy
# MEDICAL_QUESTIONS = [
#     "What are the symptoms of COVID-19?",
#     "How do I treat a headache?",
#     "Is high blood pressure dangerous?",
#     "What causes diabetes?", 
#     "How much vitamin C should I take daily?",
#     "Why does my back hurt when I wake up?",
#     "Is this rash normal?",
#     "What are the side effects of ibuprofen?",
#     "How can I lower my cholesterol?",
#     "What's the recommended blood sugar level?",
#     "How do I know if I have anxiety?",
#     "What vaccines do I need before traveling?",
#     "Is this mole cancerous?",
#     "How do I treat a sunburn?",
#     "What are the early signs of pregnancy?",
#     "How do I deal with allergies?",
#     "What should I do for a sprained ankle?",
#     "How much water should I drink daily?",
#     "What causes high blood pressure?",
#     "How effective is this medicine?",
#     "Should I be worried about my blood test results?",
#     "What does this medical term mean?",
#     "How can I reduce inflammation?",
#     "What vitamins should I take?",
#     "Is it normal to feel dizzy after exercise?",
#     "Can you explain what a cardiologist does?",
#     "What causes migraines?",
#     "Is it safe to take this medication while pregnant?",
#     "What are the signs of dehydration?",
#     "How do I know if I need glasses?",
#     "What should my heart rate be?",
#     "Is it normal to have joint pain?",
#     "What does it mean if my urine is dark?",
#     "How do I manage my diabetes?",
#     "What are the symptoms of iron deficiency?",
#     "What should I eat for better gut health?",
#     "How often should I get a physical?",
#     "What are the symptoms of thyroid problems?",
#     "Should I get the flu shot?",
#     "Is this antibiotic working?",
#     "How do I read my lab results?",
#     "What does this diagnosis mean?",
#     "How can I manage my chronic pain?",
#     "What causes acid reflux?",
#     "How can I improve my sleep?",
#     "What are the symptoms of a heart attack?",
#     "Is my blood pressure reading normal?",
#     "How can I prevent getting sick?",
#     "What does this medical abbreviation mean?"
# ]

# NON_MEDICAL_QUESTIONS = [
#     "What's the weather today?",
#     "Can you recommend a good movie?",
#     "How do I bake a cake?",
#     "What's the capital of France?",
#     "Who won the last World Cup?",
#     "Tell me a joke",
#     "What's 15 * 24?",
#     "How do I reset my phone?",
#     "What's the best laptop to buy?",
#     "Who is the president?",
#     "What's the time in Tokyo?",
#     "How do I learn to code?",
#     "Who wrote Harry Potter?",
#     "What's the stock market doing today?",
#     "What's your favorite color?",
#     "How tall is the Eiffel Tower?",
#     "What's the meaning of life?",
#     "When was the internet invented?",
#     "Help me write an email",
#     "How do I grow tomatoes?",
#     "What is AI?",
#     "What is artificial intelligence?",
#     "What is deep learning?",
#     "Who are you?",
#     "What's your name?",
#     "Tell me about yourself",
#     "How were you made?",
#     "What programming language are you written in?",
#     "Who created you?",
#     "How old are you?",
#     "What do you know about history?",
#     "Help me with my homework",
#     "What's the square root of 144?",
#     "Translate this to Spanish",
#     "What's a good recipe for pancakes?",
#     "How do I fix my computer?",
#     "Tell me about the latest technology",
#     "What's the best way to learn a language?",
#     "What's the meaning of this word?",
#     "Who won the Super Bowl?",
#     "How do I get to the airport?",
#     "What's the latest news?",
#     "Can you recommend a good book?",
#     "What are some vacation destinations?",
#     "What's the best smartphone?",
#     "How do I train my dog?",
#     "What's the best investment strategy?",
#     "What's your opinion on politics?",
#     "Tell me about space exploration"
# ]

# def extract_topic(message):
#     """Extract the likely topic from a non-medical question"""
#     # Simple topic extraction based on noun phrases or question words
#     topic = "that topic"
    
#     # Look for common question patterns
#     patterns = [
#         (r'what is (a |an )?(.*?)\??$', r'\2'),
#         (r'what are (.*?)\??$', r'\1'),
#         (r'who is (.*?)\??$', r'\1'),
#         (r'who are (.*?)\??$', r'\1'), 
#         (r'how do (i|you|we) (.*?)\??$', r'\2'),
#         (r'how (.*?)\??$', r'\1'),
#         (r'why (.*?)\??$', r'\1'),
#         (r'can you (.*?)\??$', r'\1'),
#         (r'tell me about (.*?)[\.\?]?$', r'\1')
#     ]
    
#     message_lower = message.lower()
#     for pattern, replacement in patterns:
#         match = re.search(pattern, message_lower)
#         if match:
#             topic = match.expand(replacement)
#             break
    
#     return topic

# def train_or_load_classifier():
#     """Train a new classifier or load an existing one"""
#     if os.path.exists(MODEL_PATH):
#         try:
#             # Try to load the existing model
#             with open(MODEL_PATH, 'rb') as f:
#                 return pickle.load(f)
#         except:
#             print("Could not load existing model, training a new one")
    
#     # Create and train a new model
#     # Prepare training data
#     all_questions = MEDICAL_QUESTIONS + NON_MEDICAL_QUESTIONS
#     labels = ["medical"] * len(MEDICAL_QUESTIONS) + ["non-medical"] * len(NON_MEDICAL_QUESTIONS)
    
#     # Create a pipeline with TF-IDF and Naive Bayes
#     model = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
#         ('classifier', MultinomialNB())
#     ])
    
#     # Train the model
#     model.fit(all_questions, labels)
    
#     # Save the model
#     with open(MODEL_PATH, 'wb') as f:
#         pickle.dump(model, f)
    
#     return model

# # Load or train the classifier
# classifier = train_or_load_classifier()

# def preprocess_text(text):
#     """Simple text preprocessing"""
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# def is_healthcare_related(message):
#     """Use the classifier to determine if a message is healthcare related"""
#     # Preprocess the text
#     processed_text = preprocess_text(message)
    
#     # Emergency keywords always count as healthcare related
#     emergency_keywords = ['heart attack', 'stroke', 'can\'t breathe', 'suicide', 'emergency', 'severe bleeding']
#     if any(keyword in message.lower() for keyword in emergency_keywords):
#         return True
    
#     # Some explicit medical keywords that should always be classified as medical
#     definite_medical_terms = [
#         'doctor', 'hospital', 'medicine', 'medical', 'symptom', 'disease', 'illness',
#         'pain', 'treatment', 'diagnosis', 'health', 'prescription', 'drug', 'blood',
#         'fever', 'ache', 'allergy', 'vitamin', 'healthcare', 'diet', 'exercise',
#         'vaccine', 'heart', 'diabetes', 'cancer', 'prescription', 'infection'
#     ]
    
#     if any(term in message.lower().split() for term in definite_medical_terms):
#         return True
    
#     # Use the classifier to predict
#     prediction = classifier.predict([processed_text])[0]
#     prediction_proba = classifier.predict_proba([processed_text])[0]
    
#     # Get the probability of being medical
#     medical_idx = list(classifier.classes_).index('medical')
#     medical_prob = prediction_proba[medical_idx]
    
#     # We can adjust this threshold to be more strict
#     # Higher value = more likely to reject borderline queries
#     threshold = 0.6
    
#     return medical_prob >= threshold

# @app.route('/')
# def home():
#     """Render the home page with the chat interface"""
#     return render_template('index.html')

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     """Handle chat messages from the user"""
#     data = request.json
#     user_message = data.get('message', '')
#     user_id = data.get('user_id', 'default_user')
    
#     # Check for emergency keywords and provide immediate response if detected
#     emergency_keywords = ['heart attack', 'stroke', 'can\'t breathe', 'suicide', 'emergency', 'severe bleeding']
#     if any(keyword in user_message.lower() for keyword in emergency_keywords):
#         return jsonify({
#             'response': "EMERGENCY DETECTED: If you're experiencing a medical emergency, please call emergency services "
#                        "(911 in the US) immediately. This chatbot is not designed to handle emergency situations."
#         })
    
#     # Determine if the query is healthcare related
#     is_medical = is_healthcare_related(user_message)
    
#     # Get or initialize conversation history
#     if user_id not in conversation_store:
#         conversation_store[user_id] = []
    
#     # Add user message to history
#     conversation_store[user_id].append({"role": "user", "content": user_message})
    
#     # Handle non-medical questions
#     if not is_medical:
#         # Extract topic from the question for a more personalized response
#         topic = extract_topic(user_message)
#         response_text = NON_MEDICAL_RESPONSE.format(topic=topic)
        
#         # Add response to history
#         conversation_store[user_id].append({"role": "assistant", "content": response_text})
        
#         return jsonify({'response': response_text})
    
#     # If medical, proceed with normal workflow
#     try:
#         # Prepare the messages for DeepSeek API, including history
#         messages = [{"role": "system", "content": HEALTHCARE_CONTEXT}] + conversation_store[user_id][-5:]  # Include last 5 messages
        
#         # Call DeepSeek API
#         response = query_deepseek(messages)
#         assistant_message = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
#         # Check for no response
#         if not assistant_message:
#             assistant_message = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
#         # Add disclaimer if not already included
#         if "not a replacement for professional medical advice" not in assistant_message.lower():
#             assistant_message += "\n\nReminder: This information is for educational purposes only and is not a replacement for professional medical advice."
        
#         # Add assistant message to history
#         conversation_store[user_id].append({"role": "assistant", "content": assistant_message})
        
#         return jsonify({'response': assistant_message})
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({
#             'response': "I'm sorry, I encountered an error processing your request. Please try again later."
#         }), 500

# def query_deepseek(messages):
#     """Send a query to OpenRouter API using DeepSeek R1 model and return the response"""
#     headers = {
#         "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": "http://localhost:5000",  # Update with your actual site URL in production
#         "X-Title": "Healthcare Chatbot"  # Update with your actual site name
#     }
    
#     payload = {
#         "model": "deepseek/deepseek-r1:free",  # Using DeepSeek R1 via OpenRouter
#         "messages": messages,
#         "temperature": 0.7  # Balances creativity and accuracy
#     }
    
#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",  # OpenRouter endpoint
#         headers=headers,
#         json=payload,
#         timeout=30  # Timeout after 30 seconds
#     )
    
#     if response.status_code != 200:
#         raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
#     return response.json()

# # Route to clear conversation history
# @app.route('/api/clear_history', methods=['POST'])
# def clear_history():
#     """Clear the conversation history for a user"""
#     data = request.json
#     user_id = data.get('user_id', 'default_user')
    
#     if user_id in conversation_store:
#         conversation_store[user_id] = []
    
#     return jsonify({'status': 'success', 'message': 'Conversation history cleared'})

# # Route to retrain the classifier with new examples
# @app.route('/api/retrain', methods=['POST'])
# def retrain_classifier():
#     """Admin endpoint to retrain the classifier with new examples"""
#     data = request.json
#     password = data.get('password', '')
    
#     # Simple password protection (use a proper auth system in production)
#     if password != os.getenv("ADMIN_PASSWORD", "admin_password"):
#         return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    
#     # Get new examples
#     new_medical = data.get('medical_examples', [])
#     new_non_medical = data.get('non_medical_examples', [])
    
#     # Update global examples
#     global MEDICAL_QUESTIONS, NON_MEDICAL_QUESTIONS
#     MEDICAL_QUESTIONS.extend(new_medical)
#     NON_MEDICAL_QUESTIONS.extend(new_non_medical)
    
#     # Retrain classifier
#     global classifier
#     classifier = train_or_load_classifier()
    
#     return jsonify({'status': 'success', 'message': 'Classifier retrained successfully'})

# if __name__ == '__main__':
#     app.run(debug=True)