import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 170)

voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

def speak(message):
    engine.say(message)
    engine.runAndWait()

if __name__ == "__main__":
    speak("AI Driver Assistant is ready.")
    