# app.py (with improved API key handling)
import streamlit as st
import sqlite3
import datetime as dt
import json
from pathlib import Path
import gtts
from io import BytesIO
import base64
import speech_recognition as sr
import hashlib
import time
from gtts import gTTS
from io import BytesIO
import streamlit as st
from tempfile import NamedTemporaryFile
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import requests
from timezonefinder import TimezoneFinder
import pytz
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Appointment Manager",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini AI with better error handling
def init_gemini():
    """Initialize Gemini AI with API key from .env file"""
    try:
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Check if API key is missing or still has the placeholder
        if not api_key or api_key == "your_actual_api_key_here":
            st.warning("""
            """)
            return None
            
        genai.configure(api_key=api_key)
        
        # Test the API key with a simple request
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            # Use a very simple test that shouldn't trigger content filters
            response = model.generate_content("Hello", safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            })
        except Exception as auth_error:
            error_msg = str(auth_error)
            if "API_KEY_INVALID" in error_msg:
                st.error(f"""
                **Invalid **
            
                
                Error details: {error_msg}
                """)
            else:
                st.error(f"API key validation failed: {error_msg}")
            return None
        
        # Create the model with proper configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        st.success("✓ Gemini AI connected successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {e}")
        return None

# Database setup with migration support
def init_db():
    """Initialize the database with required tables and handle migrations"""
    conn = sqlite3.connect('appointments.db')
    c = conn.cursor()
    
    # Check if users table exists and has the timezone column
    c.execute("PRAGMA table_info(users)")
    users_columns = [column[1] for column in c.fetchall()]
    
    # Create users table if it doesn't exist
    if not users_columns:
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      email TEXT,
                      timezone TEXT DEFAULT 'UTC',
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    elif 'timezone' not in users_columns:
        # Add timezone column if it doesn't exist
        c.execute("ALTER TABLE users ADD COLUMN timezone TEXT DEFAULT 'UTC'")
    
    # Check if appointments table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='appointments'")
    if not c.fetchone():
        c.execute('''CREATE TABLE appointments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      title TEXT NOT NULL,
                      description TEXT,
                      date TEXT NOT NULL,
                      time TEXT NOT NULL,
                      duration INTEGER DEFAULT 60,
                      priority INTEGER DEFAULT 2,
                      category TEXT DEFAULT 'General',
                      location TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
    else:
        # Check if appointments table has all the required columns
        c.execute("PRAGMA table_info(appointments)")
        appointments_columns = [column[1] for column in c.fetchall()]
        
        if 'duration' not in appointments_columns:
            c.execute("ALTER TABLE appointments ADD COLUMN duration INTEGER DEFAULT 60")
        if 'priority' not in appointments_columns:
            c.execute("ALTER TABLE appointments ADD COLUMN priority INTEGER DEFAULT 2")
        if 'category' not in appointments_columns:
            c.execute("ALTER TABLE appointments ADD COLUMN category TEXT DEFAULT 'General'")
        if 'location' not in appointments_columns:
            c.execute("ALTER TABLE appointments ADD COLUMN location TEXT")
    
    # Check if tasks table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
    if not c.fetchone():
        c.execute('''CREATE TABLE tasks
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      title TEXT NOT NULL,
                      description TEXT,
                      due_date TEXT,
                      priority INTEGER DEFAULT 2,
                      status INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Check if notes table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
    if not c.fetchone():
        c.execute('''CREATE TABLE notes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      title TEXT NOT NULL,
                      content TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def create_connection():
    """Create a database connection"""
    return sqlite3.connect('appointments.db')

# User authentication functions
def create_user(username, password, email, timezone='UTC'):
    """Create a new user in the database"""
    conn = create_connection()
    c = conn.cursor()
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        c.execute("INSERT INTO users (username, password, email, timezone) VALUES (?, ?, ?, ?)",
                  (username, hashed_password, email, timezone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials"""
    conn = create_connection()
    c = conn.cursor()
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    c.execute("SELECT id, username, timezone FROM users WHERE username = ? AND password = ?",
              (username, hashed_password))
    user = c.fetchone()
    conn.close()
    
    return user

def update_user_timezone(user_id, timezone):
    """Update user's timezone"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("UPDATE users SET timezone = ? WHERE id = ?",
              (timezone, user_id))
    conn.commit()
    conn.close()

# Appointment functions
def add_appointment(user_id, title, description, date, time_str, duration=60, priority=2, category="General", location=""):
    """Add a new appointment for the user"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("INSERT INTO appointments (user_id, title, description, date, time, duration, priority, category, location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (user_id, title, description, date, time_str, duration, priority, category, location))
    conn.commit()
    conn.close()

def get_user_appointments(user_id, start_date=None, end_date=None):
    """Get appointments for a user within a date range"""
    conn = create_connection()
    c = conn.cursor()
    
    if start_date and end_date:
        c.execute("SELECT id, title, description, date, time, duration, priority, category, location FROM appointments WHERE user_id = ? AND date BETWEEN ? AND ? ORDER BY date, time",
                  (user_id, start_date, end_date))
    else:
        c.execute("SELECT id, title, description, date, time, duration, priority, category, location FROM appointments WHERE user_id = ? ORDER by date, time",
                  (user_id,))
    
    appointments = c.fetchall()
    conn.close()
    
    return appointments

def delete_appointment(appointment_id, user_id):
    """Delete an appointment"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("DELETE FROM appointments WHERE id = ? AND user_id = ?",
              (appointment_id, user_id))
    conn.commit()
    conn.close()

# Task functions
def add_task(user_id, title, description, due_date, priority=2):
    """Add a new task for the user"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("INSERT INTO tasks (user_id, title, description, due_date, priority) VALUES (?, ?, ?, ?, ?)",
              (user_id, title, description, due_date, priority))
    conn.commit()
    conn.close()

def get_user_tasks(user_id, status=None):
    """Get tasks for a user"""
    conn = create_connection()
    c = conn.cursor()
    
    if status is not None:
        c.execute("SELECT id, title, description, due_date, priority, status FROM tasks WHERE user_id = ? AND status = ? ORDER BY due_date, priority",
                  (user_id, status))
    else:
        c.execute("SELECT id, title, description, due_date, priority, status FROM tasks WHERE user_id = ? ORDER BY due_date, priority",
                  (user_id,))
    
    tasks = c.fetchall()
    conn.close()
    
    return tasks

def update_task_status(task_id, user_id, status):
    """Update task status"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("UPDATE tasks SET status = ? WHERE id = ? AND user_id = ?",
              (status, task_id, user_id))
    conn.commit()
    conn.close()

def delete_task(task_id, user_id):
    """Delete a task"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("DELETE FROM tasks WHERE id = ? AND user_id = ?",
              (task_id, user_id))
    conn.commit()
    conn.close()

# Notes functions
def add_note(user_id, title, content):
    """Add a new note for the user"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("INSERT INTO notes (user_id, title, content) VALUES (?, ?, ?)",
              (user_id, title, content))
    conn.commit()
    conn.close()

def get_user_notes(user_id):
    """Get notes for a user"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, title, content, created_at, updated_at FROM notes WHERE user_id = ? ORDER BY updated_at DESC",
              (user_id,))
    notes = c.fetchall()
    conn.close()
    
    return notes

def update_note(note_id, user_id, title, content):
    """Update a note"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
              (title, content, note_id, user_id))
    conn.commit()
    conn.close()

def delete_note(note_id, user_id):
    """Delete a note"""
    conn = create_connection()
    c = conn.cursor()
    
    c.execute("DELETE FROM notes WHERE id = ? AND user_id = ?",
              (note_id, user_id))
    conn.commit()
    conn.close()

# AI Functions
def generate_ai_suggestions(user_id, model):
    """Generate AI suggestions based on user's schedule"""
    if not model:
        return "**AI features are disabled.** Please add a valid Gemini API key to your .env file to enable AI suggestions."
    
    appointments = get_user_appointments(user_id)
    tasks = get_user_tasks(user_id)
    
    if not appointments and not tasks:
        return "I don't have enough data to provide suggestions yet. Please add some appointments or tasks first."
    
    # Prepare data for AI
    schedule_data = "Appointments:\n"
    for appt in appointments:
        schedule_data += f"- {appt[3]} {appt[4]}: {appt[1]} ({appt[7]})\n"
    
    schedule_data += "\nTasks:\n"
    for task in tasks:
        status = "Not started" if task[5] == 0 else "In progress" if task[5] == 1 else "Completed"
        schedule_data += f"- {task[3]}: {task[1]} (Priority: {task[4]}, Status: {status})\n"
    
    prompt = f"""Based on the following schedule and tasks, provide helpful suggestions, reminders, and productivity tips:
    
    {schedule_data}
    
    Please provide:
    1. Any potential scheduling conflicts
    2. Suggestions for time management
    3. Reminders about upcoming important events
    4. Tips for prioritizing tasks
    5. Any other helpful insights
    
    Keep the response concise and actionable."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating suggestions: {e}"

def ai_chat(user_id, message, model, chat_history=[]):
    """Chat with AI about schedule"""
    if not model:
        return "**AI features are disabled.**"
    
    appointments = get_user_appointments(user_id)
    tasks = get_user_tasks(user_id)
    
    # Prepare context
    context = "User's schedule:\n"
    for appt in appointments[:10]:  # Limit to recent 10 appointments
        context += f"- Appointment: {appt[3]} {appt[4]}: {appt[1]}\n"
    
    for task in tasks[:10]:  # Limit to recent 10 tasks
        status = "Not started" if task[5] == 0 else "In progress" if task[5] == 1 else "Completed"
        context += f"- Task: {task[3]}: {task[1]} (Status: {status})\n"
    
    prompt = f"""You are a helpful scheduling assistant. Here is the user's schedule context:
    
    {context}
    
    Current conversation history:
    {chat_history}
    
    User's message: {message}
    
    Please provide a helpful response related to their schedule, tasks, or productivity. If the user is asking to add something to their schedule, clarify the details before proceeding."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in AI chat: {e}"

# Voice functions
def text_to_speech(text):
    """Convert text to speech and return audio data"""
    try:
        tts = gtts.gTTS(text=text, lang='en')
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def record_audio(duration=5, sample_rate=16000):
    """Record audio using sounddevice"""
    st.info(f"Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float64')
    sd.wait()
    
    # Save to temporary file
    with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, recording, sample_rate)
        return tmp_file.name

def speech_to_text():
    """Convert speech to text using microphone input"""
    try:
        # Record audio
        audio_file = record_audio()
        
        # Use SpeechRecognition to process the audio
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        
        # Recognize speech
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return None

def get_today_appointments(user_id):
    """Get today's appointments for a user"""
    today = datetime.now().strftime('%Y-%m-%d')
    appointments = get_user_appointments(user_id, today, today)
    return appointments

def speak_today_schedule(user_id):
    """Generate speech for today's appointments"""
    appointments = get_today_appointments(user_id)
    
    if not appointments:
        text = "You have no appointments scheduled for today."
    else:
        text = f"You have {len(appointments)} appointment{'s' if len(appointments) > 1 else ''} today. "
        for i, appt in enumerate(appointments):
            # Handle different time formats
            time_formats = ['%H:%M:%S.%f', '%H:%M:%S', '%H:%M']
            time_obj = None
            
            for time_format in time_formats:
                try:
                    time_obj = datetime.strptime(appt[4], time_format)
                    break
                except ValueError:
                    continue
            
            if time_obj:
                formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                text += f"Appointment {i+1}: {appt[1]} at {formatted_time}. "
            else:
                text += f"Appointment {i+1}: {appt[1]} at {appt[4]}. "
    
    audio_bytes = text_to_speech(text)
    return audio_bytes, text

# Visualization functions - FIXED VERSION
def create_schedule_visualization(user_id, days=7):
    """Create a visualization of the user's schedule"""
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    appointments = get_user_appointments(user_id, start_date, end_date)
    
    if not appointments:
        return None
    
    # Prepare data for visualization
    viz_data = []
    for appt in appointments:
        try:
            # Parse the date and time - handle different time formats
            appointment_date = appt[3]
            appointment_time = appt[4]
            
            # Handle different time formats
            time_formats = ['%H:%M:%S.%f', '%H:%M:%S', '%H:%M']
            start_datetime = None
            
            for time_format in time_formats:
                try:
                    start_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", f'%Y-%m-%d {time_format}')
                    break
                except ValueError:
                    continue
            
            if start_datetime is None:
                st.error(f"Could not parse time: {appointment_time}")
                continue
            
            # Calculate end time based on duration (default to 60 minutes if not specified)
            duration = appt[5] or 60
            end_datetime = start_datetime + timedelta(minutes=duration)
            
            viz_data.append({
                'Task': appt[1],
                'Start': start_datetime,
                'End': end_datetime,
                'Category': appt[7] or 'General',
                'Priority': appt[6] or 2
            })
        except Exception as e:
            st.error(f"Error processing appointment {appt[0]}: {e}")
            continue
    
    if not viz_data:
        return None
    
    df = pd.DataFrame(viz_data)
    
    # Create Gantt chart
    try:
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="End", 
            y="Task", 
            color="Category",
            title=f"Schedule for Next {days} Days"
        )
        
        fig.update_yaxes(autorange="reversed")
        return fig
    except Exception as e:
        st.error(f"Error creating timeline visualization: {e}")
        return None

def create_productivity_chart(user_id):
    """Create a productivity chart"""
    tasks = get_user_tasks(user_id)
    
    if not tasks:
        return None
    
    status_counts = {'Not Started': 0, 'In Progress': 0, 'Completed': 0}
    priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    
    for task in tasks:
        status = task[5]
        priority = task[4]
        
        if status == 0:
            status_counts['Not Started'] += 1
        elif status == 1:
            status_counts['In Progress'] += 1
        else:
            status_counts['Completed'] += 1
            
        if priority == 1:
            priority_counts['High'] += 1
        elif priority == 2:
            priority_counts['Medium'] += 1
        else:
            priority_counts['Low'] += 1
    
    # Create subplots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Task Status", "Task Priority")
    )
    
    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            name="Status"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=list(priority_counts.keys()),
            values=list(priority_counts.values()),
            name="Priority"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    return fig

# Weather integration
def get_weather_forecast(location="New York", date=None):
    """Get weather forecast for a location (simplified)"""
    # This is a simplified version - in a real app, you'd use a weather API
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Simulate weather data based on date and location
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
    condition = weather_conditions[hash(location + date) % len(weather_conditions)]
    temp = (hash(location + date) % 40) + 10  # Temperature between 10-50°C
    
    return {
        "location": location,
        "date": date,
        "condition": condition,
        "temperature": temp,
        "humidity": (hash(location + date) % 80) + 20,  # 20-100%
        "wind_speed": (hash(location + date) % 30) + 5  # 5-35 km/h
    }

# Initialize database and AI
init_db()
gemini_model = init_gemini()

# Main app
def main():
    # Check if user is logged in
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for navigation
    st.sidebar.title("AI Appointment Manager")
    
    # Show API key status in sidebar
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        st.sidebar.warning("🔴 AI Features: Disabled")
        st.sidebar.info("Add Gemini API key to .env file to enable AI")
    elif gemini_model:
        st.sidebar.success("🟢 AI Features: Enabled")
    else:
        st.sidebar.error("🔴 AI Features: Invalid API Key")
    
    if st.session_state.user is None:
        # User is not logged in
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Logged in as {username}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        elif choice == "Register":
            st.subheader("Create New Account")
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Timezone selection
            timezones = pytz.all_timezones
            default_timezone = "UTC"
            try:
                # Try to detect user's timezone
                tf = TimezoneFinder()
                latitude, longitude = 0, 0  # Default values
                # In a real app, you might get this from the browser or user input
                timezone_name = tf.timezone_at(lat=latitude, lng=longitude)
                if timezone_name:
                    default_timezone = timezone_name
            except:
                pass
                
            selected_timezone = st.selectbox("Timezone", timezones, index=timezones.index(default_timezone) if default_timezone in timezones else 0)
            
            if st.button("Register"):
                if new_username and new_email and new_password:
                    if new_password == confirm_password:
                        if create_user(new_username, new_password, new_email, selected_timezone):
                            st.success("Account created successfully. Please login.")
                        else:
                            st.error("Username already exists")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill all fields")
    
    else:
        # User is logged in
        user_id, username, timezone = st.session_state.user
        
        st.sidebar.write(f"Logged in as **{username}**")
        st.sidebar.write(f"Timezone: **{timezone}**")
        
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.chat_history = []
            st.rerun()
        
        menu = ["Dashboard", "Add Appointment", "View Appointments", "Tasks", "Notes", "Voice Command", "AI Assistant", "Analytics"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Dashboard":
            st.subheader("Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Today's Schedule")
                today = datetime.now().strftime('%Y-%m-%d')
                today_appointments = get_user_appointments(user_id, today, today)
                
                if not today_appointments:
                    st.info("No appointments scheduled for today.")
                else:
                    for appt in today_appointments:
                        with st.expander(f"{appt[4]} - {appt[1]}"):
                            st.write(f"**Description:** {appt[2] or 'No description'}")
                            st.write(f"**Duration:** {appt[5] or 60} minutes")
                            st.write(f"**Priority:** {'High' if appt[6] == 1 else 'Medium' if appt[6] == 2 else 'Low'}")
                            st.write(f"**Category:** {appt[7] or 'General'}")
                            if appt[8]:
                                st.write(f"**Location:** {appt[8]}")
            
            with col2:
                st.write("### Upcoming Tasks")
                tasks = get_user_tasks(user_id, 0)  # Not completed tasks
                
                if not tasks:
                    st.info("No pending tasks.")
                else:
                    for task in tasks[:5]:  # Show only 5 tasks
                        with st.expander(f"{task[3]} - {task[1]}"):
                            st.write(f"**Description:** {task[2] or 'No description'}")
                            st.write(f"**Priority:** {'High' if task[4] == 1 else 'Medium' if task[4] == 2 else 'Low'}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Mark Complete", key=f"complete_{task[0]}"):
                                    update_task_status(task[0], user_id, 2)
                                    st.success("Task marked as complete!")
                                    time.sleep(1)
                                    st.rerun()
                            with col_b:
                                if st.button("Delete", key=f"delete_task_{task[0]}"):
                                    delete_task(task[0], user_id)
                                    st.success("Task deleted!")
                                    time.sleep(1)
                                    st.rerun()
            
            st.write("### Schedule Visualization")
            viz_fig = create_schedule_visualization(user_id)
            if viz_fig:
                st.plotly_chart(viz_fig, use_container_width=True)
            else:
                st.info("No appointments to visualize. Add some appointments to see your schedule.")
            
            if gemini_model:
                st.write("### AI Suggestions")
                with st.spinner("Generating AI suggestions..."):
                    suggestions = generate_ai_suggestions(user_id, gemini_model)
                    st.write(suggestions)
            else:
                st.info("AI suggestions are disabled. Add a valid Gemini API key to your .env file to enable AI features.")
        
        elif choice == "Add Appointment":
            st.subheader("Add New Appointment")
            
            with st.form("appointment_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    title = st.text_input("Title*")
                    description = st.text_area("Description")
                    date = st.date_input("Date*", min_value=datetime.today())
                    time_val = st.time_input("Time*")
                
                with col2:
                    duration = st.slider("Duration (minutes)*", 15, 240, 60, 15)
                    priority = st.selectbox("Priority*", ["High", "Medium", "Low"], index=1)
                    category = st.selectbox("Category", ["Work", "Personal", "Health", "Education", "Social", "Other"])
                    location = st.text_input("Location")
                
                submitted = st.form_submit_button("Add Appointment")
                if submitted:
                    if title:
                        priority_map = {"High": 1, "Medium": 2, "Low": 3}
                        add_appointment(
                            user_id, title, description, str(date), time_val.strftime('%H:%M'), 
                            duration, priority_map[priority], category, location
                        )
                        st.success("Appointment added successfully!")
                    else:
                        st.warning("Please enter a title for the appointment")
        
        elif choice == "View Appointments":
            st.subheader("Your Appointments")
            
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.today())
            with col2:
                end_date = st.date_input("End Date", datetime.today() + timedelta(days=7))
            
            appointments = get_user_appointments(user_id, str(start_date), str(end_date))
            
            if not appointments:
                st.info("No appointments found in the selected date range.")
            else:
                # Group appointments by date
                appointments_by_date = {}
                for appt in appointments:
                    appt_id, title, description, date, time_str, duration, priority, category, location = appt
                    if date not in appointments_by_date:
                        appointments_by_date[date] = []
                    appointments_by_date[date].append((appt_id, title, description, time_str, duration, priority, category, location))
                
                # Display appointments by date
                for date in sorted(appointments_by_date.keys()):
                    st.subheader(date)
                    
                    for appt_id, title, description, time_str, duration, priority, category, location in appointments_by_date[date]:
                        with st.expander(f"{time_str} - {title}"):
                            st.write(f"**Description:** {description or 'No description'}")
                            st.write(f"**Duration:** {duration} minutes")
                            st.write(f"**Priority:** {'High' if priority == 1 else 'Medium' if priority == 2 else 'Low'}")
                            st.write(f"**Category:** {category}")
                            if location:
                                st.write(f"**Location:** {location}")
                            
                            if st.button("Delete", key=f"delete_{appt_id}"):
                                delete_appointment(appt_id, user_id)
                                st.success("Appointment deleted!")
                                time.sleep(1)
                                st.rerun()
        
        elif choice == "Tasks":
            st.subheader("Task Management")
            
            tab1, tab2, tab3 = st.tabs(["Add Task", "View Tasks", "Completed Tasks"])
            
            with tab1:
                with st.form("task_form"):
                    title = st.text_input("Title*")
                    description = st.text_area("Description")
                    due_date = st.date_input("Due Date", min_value=datetime.today())
                    priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
                    
                    submitted = st.form_submit_button("Add Task")
                    if submitted:
                        if title:
                            priority_map = {"High": 1, "Medium": 2, "Low": 3}
                            add_task(user_id, title, description, str(due_date), priority_map[priority])
                            st.success("Task added successfully!")
                        else:
                            st.warning("Please enter a title for the task")
            
            with tab2:
                st.write("### Pending Tasks")
                tasks = get_user_tasks(user_id, 0)  # Not completed tasks
                
                if not tasks:
                    st.info("No pending tasks.")
                else:
                    for task in tasks:
                        with st.expander(f"{task[3]} - {task[1]}"):
                            st.write(f"**Description:** {task[2] or 'No description'}")
                            st.write(f"**Priority:** {'High' if task[4] == 1 else 'Medium' if task[4] == 2 else 'Low'}")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                if st.button("Start", key=f"start_{task[0]}"):
                                    update_task_status(task[0], user_id, 1)
                                    st.success("Task marked as in progress!")
                                    time.sleep(1)
                                    st.rerun()
                            with col_b:
                                if st.button("Complete", key=f"complete_{task[0]}"):
                                    update_task_status(task[0], user_id, 2)
                                    st.success("Task marked as complete!")
                                    time.sleep(1)
                                    st.rerun()
                            with col_c:
                                if st.button("Delete", key=f"delete_{task[0]}"):
                                    delete_task(task[0], user_id)
                                    st.success("Task deleted!")
                                    time.sleep(1)
                                    st.rerun()
            
            with tab3:
                st.write("### Completed Tasks")
                tasks = get_user_tasks(user_id, 2)  # Completed tasks
                
                if not tasks:
                    st.info("No completed tasks.")
                else:
                    for task in tasks:
                        with st.expander(f"{task[3]} - {task[1]}"):
                            st.write(f"**Description:** {task[2] or 'No description'}")
                            st.write(f"**Priority:** {'High' if task[4] == 1 else 'Medium' if task[4] == 2 else 'Low'}")
                            
                            if st.button("Reopen", key=f"reopen_{task[0]}"):
                                update_task_status(task[0], user_id, 0)
                                st.success("Task reopened!")
                                time.sleep(1)
                                st.rerun()
        
        elif choice == "Notes":
            st.subheader("Notes")
            
            tab1, tab2 = st.tabs(["Add Note", "View Notes"])
            
            with tab1:
                with st.form("note_form"):
                    title = st.text_input("Title*")
                    content = st.text_area("Content", height=200)
                    
                    submitted = st.form_submit_button("Save Note")
                    if submitted:
                        if title:
                            add_note(user_id, title, content)
                            st.success("Note saved successfully!")
                        else:
                            st.warning("Please enter a title for the note")
            
            with tab2:
                notes = get_user_notes(user_id)
                
                if not notes:
                    st.info("No notes found.")
                else:
                    for note in notes:
                        with st.expander(f"{note[1]} - {note[3]}"):
                            st.write(note[2])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Edit", key=f"edit_{note[0]}"):
                                    st.session_state.editing_note = note
                                    st.rerun()
                            with col2:
                                if st.button("Delete", key=f"delete_note_{note[0]}"):
                                    delete_note(note[0], user_id)
                                    st.success("Note deleted!")
                                    time.sleep(1)
                                    st.rerun()
            
            # Note editing
            if 'editing_note' in st.session_state:
                note = st.session_state.editing_note
                st.subheader("Edit Note")
                
                with st.form("edit_note_form"):
                    title = st.text_input("Title*", value=note[1])
                    content = st.text_area("Content", value=note[2], height=200)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Save Changes")
                    with col2:
                        if st.button("Cancel"):
                            del st.session_state.editing_note
                            st.rerun()
                    
                    if submitted:
                        if title:
                            update_note(note[0], user_id, title, content)
                            del st.session_state.editing_note
                            st.success("Note updated successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Please enter a title for the note")
        
        elif choice == "Voice Command":
            st.subheader("Voice Command")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Hear Today's Schedule")
                if st.button("Speak Today's Appointments"):
                    audio_bytes, text = speak_today_schedule(user_id)
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')
                        st.write("**Spoken text:**", text)
            
            with col2:
                st.write("### Voice Input")
                st.info("Click below and speak to add a quick appointment or task")
                
                if st.button("Start Voice Input"):
                    text = speech_to_text()
                    if text:
                        st.write("**You said:**", text)
                        
                        # Simple parsing of voice input
                        if "appointment" in text.lower() or "meeting" in text.lower():
                            # Default to today
                            date = datetime.today()
                            time_str = "12:00"  # Default time
                            
                            # Try to extract time
                            if "at" in text:
                                parts = text.split("at")
                                if len(parts) > 1:
                                    time_part = parts[1].strip()
                                    # Simple time extraction
                                    if ":" in time_part:
                                        time_str = time_part.split(":")[0] + ":" + time_part.split(":")[1][:2]
                            
                            # Add appointment with parsed details
                            title = "Voice appointment"
                            if "for" in text:
                                parts = text.split("for")
                                if len(parts) > 1:
                                    title = parts[1].strip()
                            
                            add_appointment(user_id, title, f"Added via voice: {text}", str(date), time_str)
                            st.success("Appointment added from voice input!")
                        
                        elif "task" in text.lower():
                            # Default to today
                            due_date = datetime.today()
                            
                            # Add task with parsed details
                            title = "Voice task"
                            if "for" in text:
                                parts = text.split("for")
                                if len(parts) > 1:
                                    title = parts[1].strip()
                            
                            add_task(user_id, title, f"Added via voice: {text}", str(due_date))
                            st.success("Task added from voice input!")
        
        elif choice == "AI Assistant":
            st.subheader("AI Assistant")
            
            if not gemini_model:
                st.warning("Is not configured. Please add to enable AI features.")
                if st.button("How to ge"):
                    st.info("""
  
                    """)
                return
            
            st.write("Chat with your AI scheduling assistant")
            
            # Display chat history
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about your schedule..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                # Get AI response
                with st.spinner("Thinking..."):
                    response = ai_chat(user_id, prompt, gemini_model, st.session_state.chat_history)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
        
        elif choice == "Analytics":
            st.subheader("Analytics & Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Productivity Overview")
                productivity_fig = create_productivity_chart(user_id)
                if productivity_fig:
                    st.plotly_chart(productivity_fig, use_container_width=True)
                else:
                    st.info("No tasks to analyze. Add some tasks to see productivity insights.")
            
            with col2:
                st.write("### Upcoming Weather")
                # Simple weather display - in a real app, you'd ask for location
                weather = get_weather_forecast()
                st.write(f"**Location:** {weather['location']}")
                st.write(f"**Condition:** {weather['condition']}")
                st.write(f"**Temperature:** {weather['temperature']}°C")
                st.write(f"**Humidity:** {weather['humidity']}%")
                st.write(f"**Wind Speed:** {weather['wind_speed']} km/h")
            
            st.write("### Schedule Analysis")
            appointments = get_user_appointments(user_id)
            
            if appointments:
                # Category distribution
                categories = {}
                for appt in appointments:
                    category = appt[7] or "General"
                    categories[category] = categories.get(category, 0) + 1
                
                if categories:
                    cat_fig = px.pie(
                        values=list(categories.values()),
                        names=list(categories.keys()),
                        title="Appointments by Category"
                    )
                    st.plotly_chart(cat_fig, use_container_width=True)
                
                # Time analysis
                appointment_times = []
                for appt in appointments:
                    try:
                        # Handle different time formats
                        time_formats = ['%H:%M:%S.%f', '%H:%M:%S', '%H:%M']
                        time_obj = None
                        
                        for time_format in time_formats:
                            try:
                                time_obj = datetime.strptime(appt[4], time_format)
                                break
                            except ValueError:
                                continue
                        
                        if time_obj:
                            appointment_times.append(time_obj.hour + time_obj.minute / 60)
                    except:
                        pass
                
                if appointment_times:
                    time_fig = px.histogram(
                        x=appointment_times,
                        title="Appointment Time Distribution",
                        labels={"x": "Hour of Day"},
                        nbins=24
                    )
                    time_fig.update_layout(xaxis_range=[0, 24])
                    st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("No appointments to analyze. Add some appointments to see analytics.")

if __name__ == "__main__":
    main()
