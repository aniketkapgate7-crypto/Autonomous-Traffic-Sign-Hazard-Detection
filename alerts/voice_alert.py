import time
import queue
import threading

try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


alert_queue = queue.Queue(maxsize=3)
last_alert_time = 0
worker_started = False


def voice_worker():
    if not VOICE_AVAILABLE:
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)

    while True:
        message = alert_queue.get()

        if message is None:
            break

        engine.say(message)
        engine.runAndWait()
        alert_queue.task_done()


def start_voice_system():
    global worker_started

    if not VOICE_AVAILABLE:
        print("Voice system not available. Install pyttsx3.")
        return

    if worker_started:
        return

    thread = threading.Thread(target=voice_worker, daemon=True)
    thread.start()
    worker_started = True


def speak_warning(message, cooldown=6):
    global last_alert_time

    if not VOICE_AVAILABLE:
        return

    current_time = time.time()

    if current_time - last_alert_time < cooldown:
        return

    start_voice_system()

    try:
        alert_queue.put_nowait(message)
        last_alert_time = current_time
    except queue.Full:
        pass
    