import streamlit as st
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Set up scikit-learn stop words
sklearn_stopwords = ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from pages.customer import predict_sentiment
from wordcloud import WordCloud
from scipy.stats import mode
from dotenv import load_dotenv
from email.mime.text import MIMEText
import smtplib
import logging
import difflib

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
st.sidebar.markdown("<p style='font-size: 35px; font-weight: bold; color: #1e90ff;'>Dashboard</p>", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Function to check agent compliance
def check_compliance(agent_transcript, script_text):
    agent_transcript_lower = agent_transcript.lower()
    script_text_lower = script_text.lower()

    matcher = difflib.SequenceMatcher(None, agent_transcript_lower, script_text_lower)
    match_ratio = matcher.ratio()

    compliance_threshold = 0.8

    return match_ratio >= compliance_threshold

# The function to send notifications
def send_email_notification(agent_id):
    try:
        # Load environment variables
        load_dotenv('.env')
        
        sender_email: str = os.getenv('SENDER_EMAIL')
        sender_password: str = os.getenv('SENDER_PASSWORD')
        recipient_email: str = os.getenv('RECIPIENT_EMAIL')
        smtp_server: str = os.getenv('SMTP_SERVER')
        smtp_port: str = os.getenv('SMTP_PORT')

        # Create a MIMEText object with the email content
        msg = MIMEText(f"Sentiment has remained negative for 3 consecutive times for client {agent_id} in the active session.")

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

# Count sentence for opening compliance
def count_sentences(text):
    sentences = re.findall(r"\.|\?|\!", text) #tracking periods to separate sentences
    return len(sentences)
       

def agent_page():
    # global counter
    if "counter" not in st.session_state:
        st.session_state.counter = 0  # Initialize counter
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize counters for consecutive negative sentiments
    if "consecutive_negative_counts" not in st.session_state:
        st.session_state.consecutive_negative_counts = {}

    st.title(":male-office-worker: Dialect-Agent DC001")
    
    new_message = st.text_input("Type a message as an agent :speech_balloon:", key="cus")

    # Script for compliance check
    opening_script = "Thank you for calling Dialect. This is Indhu, how may I assist you today?"
    closing_scripts = [
        "Thank you for calling Dialect. Is there anything else I can help you with today?",
        "Thank you for choosing Dialect. We appreciate your business. Have a great day!"
    ]
    # Initialize sentiment_prediction_customer 
    sentiment_prediction_customer = None

    if st.button("Send"):
        agent_message = ("Agent", new_message, datetime.now())
        st.session_state.messages.append(agent_message)

        st.session_state.counter += 1 
        if st.session_state.counter == 1:    # checking for compliance for only the first message
            # Check compliance for agent opening
            opening_compliance = check_compliance(new_message, opening_script)

            if opening_compliance:
                st.success("Agent's opening message is compliant with the script.")
                st.sidebar.info(f"Opening exchange agent compliance: {opening_compliance}")
            else:
                st.warning("Agent's opening message deviates from the script.")
                st.sidebar.markdown(f"Opening exchange agent compliance: Non-compliant")

        # Get the predicted sentiment from the query parameters
        query_params = st.experimental_get_query_params()
        sentiment_prediction_customer = query_params.get("predicted_sentiment", None)       

    if sentiment_prediction_customer is not None:
        agent_id = "Indhu" 
        if agent_id not in st.session_state.consecutive_negative_counts:
            st.session_state.consecutive_negative_counts[agent_id] = 0

        # Display predicted sentiment in the Streamlit sidebar
        with st.sidebar:
            # Convert sentiment_prediction_customer[0] to int
            sentiment_prediction= int(sentiment_prediction_customer[0])
            st.info(f"Predicted Real-time Sentiment: {sentiment_prediction}")

            # Accumulate predicted sentiments for real-time scatter plot
            if "predict_scatter_plot" not in st.session_state:
                st.session_state.predict_scatter_plot = {'timestamps': [], 'sentiments': []}
            st.session_state.predict_scatter_plot['timestamps'].append(datetime.now())
            st.session_state.predict_scatter_plot['sentiments'].append(sentiment_prediction)

            # Check if the tone is negative for 3 consecutive times
            if sentiment_prediction == 0:
                st.session_state.consecutive_negative_counts[agent_id] += 1
                if st.session_state.consecutive_negative_counts[agent_id] >= 3:
                    send_email_notification(agent_id)
                    st.success(f"Email notification sent for 3 consecutive negative sentiments for agent ID {agent_id}. :e-mail:")
            else:
                # Reset the count if sentiment is not negative
                st.session_state.consecutive_negative_counts[agent_id] = 0

            # Plot line plot in real-time
            df_scatter_plot = pd.DataFrame(st.session_state.predict_scatter_plot) 
            df_scatter_plot['timestamps'] = pd.to_datetime(df_scatter_plot['timestamps'])  # Ensure the DataFrame has the appropriate structure for scatter_chart
            df_scatter_plot = df_scatter_plot.set_index('timestamps')

            scatter_chart = st.line_chart(df_scatter_plot)
            # st.scatter_chart(df_scatter_plot): # Suppressed scatter plot
        
            st.sidebar.text(f"Total Messages Sent: {len(st.session_state.messages)}") 

            # Caculate Mean, Mode(Avg, Highest occuring) for Customer Sentiments
            if st.session_state.messages:
                customer_sentiments = [predict_sentiment(msg[1]) for msg in st.session_state.messages if msg[0] == "Customer"]
                formatted_mode_sentiment_customer = mode(customer_sentiments).mode
                avg_sentiment_customer = np.mean(customer_sentiments)
                formatted_avg_sentiment_customer = round(avg_sentiment_customer)

                if formatted_mode_sentiment_customer == 0:
                    formatted_mode_sentiment_customer_text = "Negative(0)"
                if formatted_mode_sentiment_customer == 1:
                    formatted_mode_sentiment_customer_text = "Neutral(1)"
                if formatted_mode_sentiment_customer == 2:
                    formatted_mode_sentiment_customer_text = "Positive(2)"
                

                st.sidebar.subheader(":blue[Sentiment]")

                # Display formatted sentiment statistics 
                st.sidebar.text(f"Real-time Sentiment(Highest occurring): {formatted_mode_sentiment_customer_text}")
                st.sidebar.text(f"Real-time Sentiment(Mix predictions): {formatted_avg_sentiment_customer}")

                # Bar chart for sentiment distribution
                sentiment_counts_customer = {0: 0, 1: 0, 2: 0}  
                for sentiment_customer in customer_sentiments:
                    sentiment_counts_customer[sentiment_customer] += 1

                st.bar_chart(sentiment_counts_customer, width=70, height=80)
                
                # Display tone based on sentiment with color formatting
                tone_text_customer = ""
                if sentiment_prediction == 0:
                    tone_text_customer = 'Current msg.tone: <span style="color:red"> Negative</span>'
                elif sentiment_prediction == 1:
                    tone_text_customer = 'Current msg. tone:<span style="color:white"> Neutral</span>'
                elif sentiment_prediction == 2:
                    tone_text_customer = 'Current msg. tone:<span style="color:green"> Positive</span>'
                
                st.sidebar.markdown(tone_text_customer, unsafe_allow_html=True)

            # Word cloud(as theme)
            current_text_customer = " ".join([msg[1] for msg in st.session_state.messages if msg[0] == "Customer"])

            # Check if current_text is not empty before generating the word cloud
            if current_text_customer:
                current_wordcloud_customer = WordCloud(width=800, height=400, background_color='black').generate(current_text_customer)
                st.subheader(":blue[Theme for Current Conversation]")
                st.image(current_wordcloud_customer.to_image())
            else:
                st.warning("Cannot generate theme for an empty conversation.")
        

    if st.button("End Conversation (Agent)"): 
        # Check if there is an active conversation
        if st.session_state.messages:
            # Check compliance for agent closing
            closing_compliance = any(check_compliance(new_message, script) for script in closing_scripts)
            
            if closing_compliance:
                st.success("Agent's closing message is compliant with one of the closing scripts.")
                
                # Clear agent-specific metrics
                st.session_state.predict_scatter_plot = {'timestamps': [], 'sentiments': []}
                st.session_state.consecutive_negative_counts = 0 

                conversations = st.session_state.get("previous_conversations", [])
                agent_messages = []
                customer_messages = []

                for role, message, timestamp in st.session_state.messages:
                    if role == "Agent":
                        agent_messages.append((role, message, timestamp))
                    elif role == "Customer":
                        customer_messages.append((role, message, timestamp))
						
				# Ensure both agent and customer messages have the same length
                min_length = min(len(agent_messages), len(customer_messages))
                agent_messages = agent_messages[:min_length]
                customer_messages = customer_messages[:min_length]

                if agent_messages or customer_messages:  # Ensure there's at least one message before appending to conversations
                    conversations.append((agent_messages, customer_messages))
                    st.session_state.previous_conversations = conversations
                    st.session_state.messages = []  # Clear the current conversation

                # Perform calculations and display real-time information
                all_timestamps = []
                all_agent_messages = []
                all_customer_messages = []
                all_agent_sentiments = []
                all_customer_sentiments = []

                for agent_messages, customer_messages in conversations:
                    if agent_messages and customer_messages:
                        agent_role, agent_texts, agent_timestamps = zip(*agent_messages)
                        customer_role, customer_texts, customer_timestamps = zip(*customer_messages)

                        # Ensure that there are at least three elements in the messages
                        if len(agent_texts) >= 3 and len(customer_texts) >= 3:
                            agent_sentiment_predictions = [predict_sentiment(msg) for msg in agent_texts]
                            customer_sentiment_predictions = [predict_sentiment(msg) for msg in customer_texts]

                            # Accumulate timestamps, messages, and sentiments for all conversations
                            all_timestamps.extend(agent_timestamps)
                            all_agent_messages.extend(agent_texts)
                            all_customer_messages.extend(customer_texts)
                            all_agent_sentiments.extend(agent_sentiment_predictions)
                            all_customer_sentiments.extend(customer_sentiment_predictions)

                if all_timestamps:
                    # Add this debug print to check if this block is being executed
                    print("Debug: Inside 'if all_timestamps:' block")
                    # Plot sentiment trend for agent
                    st.line_chart(pd.DataFrame({'Agent Sentiment': all_agent_sentiments}, index=all_timestamps))

                    # Average sentiment score on the same plot as the trend plot
                    agent_average_sentiment = np.mean(all_agent_sentiments)
                    formatted_agent_avg_sentiment = round(agent_average_sentiment)
                    st.write(f"Agent sentiment score(Avg): {formatted_agent_avg_sentiment}")

                    # Plot sentiment trend for customer
                    st.line_chart(pd.DataFrame({'Customer Sentiment': all_customer_sentiments}, index=all_timestamps))

                    # Average sentiment score on the same plot as the trend plot
                    customer_average_sentiment = np.mean(all_customer_sentiments)
                    formatted_customer_avg_sentiment = round(customer_average_sentiment)
                    st.write(f"Customer sentiment score(Avg): {formatted_customer_avg_sentiment}")

                    agent_sentiment_description = ""
                    agent_sentiment_color = ""
                    if formatted_agent_avg_sentiment == 0:
                        agent_sentiment_description = "Agent sentiment score is mainly Negative"
                        agent_sentiment_color = "red"
                    elif formatted_agent_avg_sentiment == 1:
                        agent_sentiment_description = "Agent sentiment score is mainly Neutral"
                        agent_sentiment_color = "white"
                    elif formatted_agent_avg_sentiment == 2:
                        agent_sentiment_description = "Agent sentiment score is mainly Positive"
                        agent_sentiment_color = "green"

                    customer_sentiment_description = ""
                    customer_sentiment_color = ""
                    if formatted_customer_avg_sentiment == 0:
                        customer_sentiment_description = "Customer sentiment score is mainly Negative"
                        customer_sentiment_color = "red"
                    elif formatted_customer_avg_sentiment == 1:
                        customer_sentiment_description = "Customer sentiment score is mainly Neutral"
                        customer_sentiment_color = "white"
                    elif formatted_customer_avg_sentiment == 2:
                        customer_sentiment_description = "Customer sentiment score is mainly Positive"
                        customer_sentiment_color = "green"

                    # Display sentiment description with color formatting for agent
                    st.markdown(f'<span style="color:{agent_sentiment_color}">{agent_sentiment_description}</span>', unsafe_allow_html=True)

                    # Display sentiment description with color formatting for customer
                    st.markdown(f'<span style="color:{customer_sentiment_color}">{customer_sentiment_description}</span>', unsafe_allow_html=True)

                    # Line-by-line transcription accumulated for all chat on a date basis for agent
                    agent_date_accumulated_transcription = {}
                    for timestamp, msg, sentiment in zip(all_timestamps, all_agent_messages, all_agent_sentiments):
                        date_str = timestamp.date().isoformat()
                        if date_str not in agent_date_accumulated_transcription:
                            agent_date_accumulated_transcription[date_str] = []
                        agent_date_accumulated_transcription[date_str].append(f"{role}: {msg} - Sentiment: {sentiment}")

                    # Plot word cloud for accumulated transcription on a daily basis for agent
                    for date_str, transcription_list in agent_date_accumulated_transcription.items():
                        text = " ".join(transcription_list)
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        st.header(f"Agent Conversation theme {date_str}")
                        st.image(wordcloud.to_image())

                    # Line-by-line transcription accumulated for all chat on a date basis for customer
                    customer_date_accumulated_transcription = {}
                    for timestamp, msg, sentiment in zip(all_timestamps, all_customer_messages, all_customer_sentiments):
                        date_str = timestamp.date().isoformat()
                        if date_str not in customer_date_accumulated_transcription:
                            customer_date_accumulated_transcription[date_str] = []
                        customer_date_accumulated_transcription[date_str].append(f"{role}: {msg} - Sentiment: {sentiment}")

                    # Plot word cloud for accumulated transcription on a daily basis for customer
                    for date_str, transcription_list in customer_date_accumulated_transcription.items():
                        text = " ".join(transcription_list)
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        st.header(f"Customer Conversation theme {date_str}")
                        st.image(wordcloud.to_image())
            else:
                st.warning("Agent's closing message deviates from the script.")
        else:
            st.warning("No active conversation to end.")

        
    for role, message, timestamp in st.session_state.messages:  
        time = timestamp.strftime("%H:%M:%S")
        st.write(f"{time} - {role}: {message}") 

if __name__ == '__main__':
    agent_page()
    st.divider()
    st.sidebar.markdown("<p style='font-size: 12px; color: white;'>DIALECT Understanding sentiment #2.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-size: 12px; color: white;'>Northumbria University AP 2023</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-size: 11px; color: white;'>Proof-of-concept</p>", unsafe_allow_html=True)
