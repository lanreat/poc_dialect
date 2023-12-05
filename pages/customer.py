import streamlit as st
import joblib
from datetime import datetime
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
import logging
from scipy.stats import mode
import pandas as pd
import numpy as np

# Setting the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 900px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<p style='font-size: 35px; font-weight: bold; color: #1e90ff;text-align: center;'>DIALECT UNDERSTANDING SENTIMENT #2</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 35px; font-weight: bold; color: #fcc305; text-align: center;'>Say Hello!</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 12px; color: white;text-align: center;'>DIALECT Understanding sentiment #2.</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 12px; color: white;text-align: center;'>Northumbria University AP 2023</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 11px; color: white;text-align: center;'>Proof-of-concept</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 35px; font-weight: bold; color: #1e90ff;text-align: center;'>CONTACT CENTER</p>", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Set up scikit-learn stop words
sklearn_stopwords = ENGLISH_STOP_WORDS

# Load the fitted vectorizer from the file
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the trained ExtraTreesClassifier model with error handling
model_filename = 'extra_trees_model.joblib'
try:
    extra_trees_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found.")
    extra_trees_model = None

# Initialize variables before the block
all_timestamps, all_messages, all_sentiments = [], [], []

# Function to preprocess input line text, according to the training of the model
def preprocess_message(message):
    # Substitute 's with ' is '
    message = re.sub(r"'s\b", " is", message)

    # Replace "DM" with "email"
    message = re.sub(r'\bDM\b', 'email', message)

    # Remove ellipsis
    message = re.sub(r'\.\.\.', '', message)

    # Remove hyperlinks
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)

    # Remove punctuations, special characters, and digits
    message = re.sub(r'[^a-zA-Z\s]', '', message)

    # Tokenize, lowercase, and remove stop words using scikit-learn
    words_sklearn = word_tokenize(message.lower())
    filtered_words_sklearn = [word for word in words_sklearn if word not in sklearn_stopwords]

    # Lemmatize each word
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_words_sklearn]

    # rejoining the text
    processed_words = ' '.join(lemmatized_tokens)

    return processed_words

# The function to predict the sentiment
def predict_sentiment(message):
    processed_message = preprocess_message(message)
    # Transform the processed message using the loaded vectorizer
    vectored_message = tfidf_vectorizer.transform([processed_message]).toarray()
    # Make predictions using the loaded model
    prediction = extra_trees_model.predict(vectored_message)
    return prediction[0]

# The function to send notifications
def send_email_notification():
    try:
        # Load environment variables
        load_dotenv('.env')
        
        sender_email: str = os.getenv('SENDER_EMAIL')
        sender_password: str = os.getenv('SENDER_PASSWORD')
        recipient_email: str = os.getenv('RECIPIENT_EMAIL')
        smtp_server: str = os.getenv('SMTP_SERVER')
        smtp_port: str = os.getenv('SMTP_PORT')

        # Create a MIMEText object with the email content
        msg = MIMEText("The average sentiment has remained negative for 3 consecutive times for client ID 1234 in the active session.")

        # Set the sender, recipient, and subject of the email
        msg['Subject'] = 'Negative Sentiment Notification'
        msg['From'] = sender_email
        msg['To'] = recipient_email

        # Set up the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, msg.as_string())

        # Close the SMTP server
        server.quit()
    except Exception as e:
        logger.error(f"Error sending email notification: {str(e)}")

def display_sidebar(sentiment_prediction_customer):
    # Display predicted sentiment in the Streamlit sidebar
    with st.sidebar:
        # st.info(f"Predicted Real-time Sentiment: {sentiment_prediction_customer}")

        # Accumulate predicted sentiments for real-time line plot
        if "predict_line_plot" not in st.session_state:
            st.session_state.predict_line_plot = {'timestamps': [], 'sentiments': []}
        st.session_state.predict_line_plot['timestamps'].append(datetime.now())
        st.session_state.predict_line_plot['sentiments'].append(sentiment_prediction_customer)

        # Plot line chart in real-time
        df_line_plot = pd.DataFrame(st.session_state.predict_line_plot)
        
        df_line_plot['timestamps'] = pd.to_datetime(df_line_plot['timestamps']) # Ensure the DataFrame has the appropriate structure for scatter_chart
        df_line_plot = df_line_plot.set_index('timestamps')

        # line_chart = st.line_chart(df_line_plot)
        # st.scatter_chart(df_scatter_plot) #Suppressed scatter chart

        
        # st.sidebar.text(f"Total Messages Sent: {len(st.session_state.messages)}")

        # Caculate Mean, Mode(Avg, Highest occuring)
        if st.session_state.messages:
            sentiment_predictions = [predict_sentiment(msg[1]) for msg in st.session_state.messages]
            formatted_mode_sentiment = mode(sentiment_predictions).mode
            avg_sentiment = np.mean(sentiment_predictions)
            formatted_avg_sentiment = round(avg_sentiment)

            # st.sidebar.subheader(":blue[Sentiment]")

            # Display formatted sentiment statistics 
            # st.sidebar.text(f"Real-time Sentiment(Highest occurring): {formatted_mode_sentiment}")
            # st.sidebar.text(f"Real-time Sentiment(Mix predictions): {formatted_avg_sentiment}")

            # Bar chart for sentiment distribution
            sentiment_counts = {0: 0, 1: 0, 2: 0}  
            for sentiment in sentiment_predictions:
                sentiment_counts[sentiment] += 1

            # st.bar_chart(sentiment_counts, width=70, height=80)
        
            # Display tone based on sentiment with color formatting
            tone_text = ""
            if sentiment_prediction_customer == 0:
                tone_text = 'Current msg.tone: <span style="color:red"> Negative</span>'
            elif sentiment_prediction_customer == 1:
                tone_text = 'Current msg. tone:<span style="color:white"> Neutral</span>'
            elif sentiment_prediction_customer == 2:
                tone_text = 'Current msg. tone:<span style="color:green"> Positive</span>'
            
            # st.sidebar.markdown(tone_text, unsafe_allow_html=True)

            # Check if the tone is negative for 3 consecutive times
            
            if sentiment_prediction_customer == 0:
                if "consecutive_negative_count" not in st.session_state:
                    st.session_state.consecutive_negative_count = 1
                else:
                    st.session_state.consecutive_negative_count += 1
                
                if st.session_state.consecutive_negative_count >= 3:
                    # send_email_notification()
                    # st.success("Email notification sent for 3 consecutive negative sentiments. :e-mail:")
                    pass
            else:
                # Reset the count if sentiment is not negative
                st.session_state.consecutive_negative_count = 0

    # Word cloud(as theme)
    current_text = " ".join([msg[1] for msg in st.session_state.messages])

    # Check if current_text is not empty before generating the word cloud
    if current_text:
        current_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(current_text)
        # st.subheader(":blue[Theme for Current Conversation]")
        # st.image(current_wordcloud.to_image())
    else:
        # st.warning("Cannot generate theme for an empty conversation.")
        pass

def customer_page():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    st.title(":male-office-worker: Customer Conversation Page")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    new_message = st.text_input("Type a message as a customer :speech_balloon:")

    if st.button("Send"):
        if new_message.strip():
            customer_message = ("Customer", new_message, datetime.now())
            st.session_state.messages.append(customer_message)
        else:
            st.warning("Blank Message 🗅.Please type a message before sending")

        # Predict sentiment for the customer's message
        sentiment_prediction_customer = predict_sentiment(new_message)

        if sentiment_prediction_customer is not None:
            # Set query param for agent page
            st.experimental_set_query_params(predicted_sentiment=sentiment_prediction_customer)

            display_sidebar(sentiment_prediction_customer) 
    
    if st.button("End conversation (Customer)"):
        if new_message.strip():
            customer_message = ("Customer", new_message, datetime.now())
            st.session_state.messages.append(customer_message)
        else:
            st.warning("Blank Message 🗅.Please type a message before sending")

        # Add a special message to indicate that the customer has ended the conversation
        end_message = ("Customer", "Customer has ended the conversation", datetime.now())
        st.session_state.messages.append(end_message)
        
        # Predict sentiment for the customer's message
        sentiment_prediction_customer = predict_sentiment(new_message)

        if sentiment_prediction_customer is not None:
            # Set query param for agent page
            st.experimental_set_query_params(predicted_sentiment=sentiment_prediction_customer)

    # Display customer messages
    for role, message, timestamp in st.session_state.messages:
        time = timestamp.strftime("%H:%M:%S")
        st.write(f"{time} - {role}: {message}") 

if __name__ == '__main__':
    customer_page()
    st.divider()
    
