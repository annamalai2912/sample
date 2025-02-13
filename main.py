import streamlit as st
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import random
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Enhanced intents with more training examples
CHATBOT_INTENTS = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Hey", "How are you", "What's up", "Good morning", 
                "Good evening", "Hi there", "Hello there", "Hey there"
            ],
            "responses": [
                "Hello! I'm an educational chatbot. Ask me how I work!",
                "Hi there! Want to learn about chatbots?",
                "Hello! I can explain how chatbots process messages."
            ]
        },
        {
            "tag": "chatbot_basics",
            "patterns": [
                "How do you work", "How do chatbots work", "Explain yourself",
                "What makes you work", "How do you understand me", "How do you process text",
                "What's your process", "How do you function", "Tell me about chatbot processing",
                "Explain chatbot functionality"
            ],
            "responses": [
                "I process messages in several steps:\n1. Break text into words (tokenization)\n2. Convert words to base form (lemmatization)\n3. Convert words to numbers\n4. Use neural network to understand intent\n5. Select appropriate response",
                "Let me explain my process:\n1. First, I split your message into words\n2. Then I simplify these words to their basic form\n3. Next, I convert them to numbers my neural network understands\n4. Finally, I determine your intent and respond accordingly"
            ]
        },
        {
            "tag": "neural_network",
            "patterns": [
                "What is a neural network", "How does AI work", "Explain AI",
                "How do you learn", "What's machine learning", "How does the AI part work",
                "Tell me about neural networks", "Explain machine learning",
                "How does artificial intelligence work", "What's deep learning"
            ],
            "responses": [
                "Neural networks are like my brain:\n1. They take input (your words as numbers)\n2. Process it through layers of connections\n3. Each layer learns to recognize patterns\n4. Finally, they predict the best response\n\nIt's similar to how human brains learn patterns!",
                "I use a neural network that:\n1. Receives your processed text\n2. Passes it through multiple layers\n3. Each layer learns different patterns\n4. The final layer predicts your intent\n\nThis helps me understand what you're asking about!"
            ]
        },
        {
            "tag": "nlp_explanation",
            "patterns": [
                "How do you understand text", "What is NLP", "Explain text processing",
                "How do you read messages", "What's natural language processing",
                "How do you parse text", "Explain language processing",
                "How do you understand language", "What's text analysis",
                "How do you process language"
            ],
            "responses": [
                "I understand text through Natural Language Processing (NLP):\n1. First, I break your message into individual words\n2. Then I simplify words to their base form (e.g., 'running' → 'run')\n3. I convert these words into numbers\n4. This lets my neural network understand the meaning",
                "My text processing works like this:\n1. Split your message into words\n2. Remove unnecessary parts like punctuation\n3. Convert words to their simplest form\n4. Transform them into numbers for processing\n\nThis helps me understand human language!"
            ]
        }
    ]
}

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Chatbot:
    def __init__(self):
        self.intents = CHATBOT_INTENTS
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_chars = ['?', '!', '.', ',']
        
        # Process all patterns
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize and lemmatize
                word_list = word_tokenize(pattern)
                self.words.extend([self.lemmatizer.lemmatize(word.lower()) 
                                 for word in word_list 
                                 if word not in self.ignore_chars])
                # Add to documents
                self.documents.append((word_list, intent['tag']))
                # Add to classes
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        # Initialize and train neural network
        self.model = self.train_model()
        
    def prepare_input(self, text, show_steps=False):
        """Convert input text to bag of words"""
        if show_steps:
            st.sidebar.markdown("### Processing Steps")
            st.sidebar.write("Original text:", text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens 
                 if word not in self.ignore_chars]
        
        if show_steps:
            st.sidebar.write("Processed words:", tokens)
        
        # Create bag of words
        bag = [1 if word in tokens else 0 for word in self.words]
        
        if show_steps:
            st.sidebar.write("Converted to numbers (1 means word is present):")
            word_presence = {word: present for word, present in zip(self.words, bag) if present}
            st.sidebar.write(word_presence)
        
        return torch.FloatTensor(bag)
    
    def train_model(self):
        # Prepare training data
        training_data = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) 
                           for word in doc[0]]
            
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training_data.append([bag, output_row])
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor([x[0] for x in training_data])
        y_train = torch.FloatTensor([x[1] for x in training_data])
        
        # Create and train model
        model = SimpleNeuralNet(len(self.words), 8, len(self.classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.argmax(1))
            loss.backward()
            optimizer.step()
        
        return model
    
    def get_response(self, text):
        """Get chatbot response with explanation"""
        # Process input
        input_data = self.prepare_input(text, show_steps=True)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
            predicted_class = self.classes[output.argmax().item()]
            probabilities = torch.softmax(output, dim=0)
            confidence = probabilities[output.argmax().item()].item()
        
        # Show confidence scores
        st.sidebar.markdown("### Intent Detection")
        for idx, intent_class in enumerate(self.classes):
            prob = probabilities[idx].item() * 100
            st.sidebar.write(f"{intent_class}: {prob:.1f}%")
        
        # Get response
        for intent in self.intents['intents']:
            if intent['tag'] == predicted_class:
                response = random.choice(intent['responses'])
                return response
        
        return "I'm not sure I understand. Could you rephrase that?"

def main():
    st.set_page_config(page_title="Educational Chatbot", page_icon="🤖", layout="wide")
    
    st.title("🤖 Educational Chatbot: Understanding How AI Chat Works")
    st.markdown("""
    This chatbot demonstrates how AI chat systems work! Try asking:
    - "How do chatbots work?"
    - "What is NLP?"
    - "How do neural networks work?"
    - "How do you process text?"
    
    Watch the sidebar to see how your message is processed in real-time!
    """)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = Chatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask me about how chatbots work!"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display response
        response = st.session_state.chatbot.get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == '__main__':
    main()