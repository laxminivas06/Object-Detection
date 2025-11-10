import cv2
import time
import queue
import threading
import pyttsx3
import os
from gtts import gTTS
import speech_recognition as sr
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from AppKit import NSSound  # macOS native sound player
import atexit  # Module for cleanup on program exit

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production!

# Configuration
CONFIG = {
    'min_confidence': 0.5,               # Detection sensitivity (0-1)
    'announce_cooldown': 3,              # Seconds between same-object announcements
    'voice_rate': 150,                   # Speech speed
    'admin_username': 'admin',
    'admin_password_hash': generate_password_hash('password'),  # Change this!
    'target_objects': None               # Set to None to detect all objects, or specify a list
}

# Speech Engine Initialization
def init_speech_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', CONFIG['voice_rate'])
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)  # First available voice
        return engine
    except Exception as e:
        print(f"Error initializing pyttsx3: {e}")
        return None

engine = init_speech_engine()
speech_queue = queue.Queue()

# Audio playback functions
def play_audio_macos(filepath):
    sound = NSSound.alloc().initWithContentsOfFile_byReference_(filepath, True)
    if sound:
        sound.play()

def speak_with_gtts(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "temp_speech.mp3"
        tts.save(filename)
        play_audio_macos(filename)
        time.sleep(0.1)  # Small delay to ensure file is playable
        os.remove(filename)
    except Exception as e:
        print(f"Audio playback error: {e}")

def speak(text):
    try:
        if engine:
            engine.say(text)
            engine.runAndWait()
        else:
            speak_with_gtts(text)
    except Exception as e:
        print(f"Speech error: {e}")
        print(f"[Speech]: {text}")

# Speech thread
def speech_loop():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit signal
            break
        speak(text)
        time.sleep(0.1)  # Prevent speech overlap

speech_thread = threading.Thread(target=speech_loop, daemon=True)
speech_thread.start()

# Initialize YOLO model
try:
    model = YOLO("yolov8n.pt")  # Using nano version for better performance
    class_names = model.names
    print(f"YOLO model loaded with {len(class_names)} classes")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Voice command recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()
detection_active = False

def listen_for_commands():
    global detection_active
    while True:
        try:
            with mic as source:
                print("Calibrating microphone...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Listening for commands...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            command = recognizer.recognize_google(audio).lower()
            print(f"Command heard: {command}")
            
            if "start" in command or "begin" in command:
                detection_active = True
                speech_queue.put("Object detection activated")
            elif "stop" in command or "end" in command:
                detection_active = False
                speech_queue.put("Detection paused")
            elif "what do you see" in command and detection_active:
                speech_queue.put("Scanning surroundings")
                
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            print("Could not understand audio")
        except Exception as e:
            print(f"Error recognizing voice: {e}")

command_thread = threading.Thread(target=listen_for_commands, daemon=True)
command_thread.start()

# Initialize video capture
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    print("Camera initialized successfully")
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

# Detection tracking
spoken_times = {}
last_announcement_time = 0

def generate_frames():
    global detection_active, last_announcement_time
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        if detection_active:
            # Perform detection
            results = model(frame, verbose=False)  # Disable logging for performance
            
            # Process results
            for result in results:
                for box in result.boxes:
                    confidence = box.conf.item()
                    cls = int(box.cls.item())
                    label = class_names[cls]
                    
                    # Filter by confidence and target objects
                    if (confidence > CONFIG['min_confidence'] and 
                        (CONFIG['target_objects'] is None or label in CONFIG['target_objects'])):
                        
                        # Announce object if not recently announced
                        current_time = time.time()
                        if (label not in spoken_times or 
                            current_time - spoken_times.get(label, 0) > CONFIG['announce_cooldown']):
                            
                            speech_queue.put(f"{label} detected")
                            spoken_times[label] = current_time
                            last_announcement_time = current_time
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                   f"{label} {confidence:.2f}",
                                   (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('detection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if (username == CONFIG['admin_username'] and 
            check_password_hash(CONFIG['admin_password_hash'], password)):
            session['logged_in'] = True
            return redirect(url_for('detection'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/detection')
def detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return Response("Unauthorized", status=401)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_detection():
    if not session.get('logged_in'):
        return Response("Unauthorized", status=401)
    
    global detection_active
    action = request.form.get('action')
    
    if action == 'start':
        detection_active = True
        speech_queue.put("Detection activated")
    elif action == 'stop':
        detection_active = False
        speech_queue.put("Detection paused")
    
    return '', 204

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Cleanup function
def cleanup():
    print("Cleaning up resources...")
    speech_queue.put(None)  # Stop speech thread
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_speech.mp3"):
        os.remove("temp_speech.mp3")

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)