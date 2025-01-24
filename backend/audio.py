import pyaudio
import wave
import whisper
from datetime import datetime
import psycopg2  # Example: Using PostgreSQL as the database
from psycopg2.extras import RealDictCursor


class AudioRecorderWithDatabase:
    def __init__(self, db_connector):
        """
        Initializes the AudioRecorderWithDatabase class with a database connector.

        :param db_connector: A database connection object (e.g., SQLite, PostgreSQL, MySQL).
        """
        self.db_connector = db_connector
        self._setup_database()

    def _setup_database(self):
        """Sets up the database by creating a table if it doesn't already exist."""
        cursor = self.db_connector.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                transcription TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
        """)
        self.db_connector.commit()

    def record_audio(self, output_file, record_seconds=10, sample_rate=44100, chunk_size=1024, channels=1):
        """
        Records audio from the microphone and saves it as a .wav file.

        :param output_file: Name of the output .wav file.
        :param record_seconds: Duration of recording in seconds.
        :param sample_rate: Sampling rate in Hz (default is 44100 Hz).
        :param chunk_size: Number of audio frames per buffer.
        :param channels: Number of audio channels (1 for mono, 2 for stereo).
        """
        audio_format = pyaudio.paInt16  # 16-bit audio format
        audio = pyaudio.PyAudio()

        # Open the audio stream
        stream = audio.open(format=audio_format,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)

        print("Recording started...")
        frames = []

        # Record audio in chunks
        for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
            data = stream.read(chunk_size)
            frames.append(data)

        print("Recording finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a .wav file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Audio recorded and saved as {output_file}")

    def transcribe_audio(self, input_file):
        """
        Transcribes audio using the Whisper model.

        :param input_file: Path to the audio file to transcribe.
        :return: The transcription as a string.
        """
        print("Loading Whisper model...")
        model = whisper.load_model("base")  # Use 'base', 'small', 'medium', or 'large'
        print("Transcribing audio...")
        result = model.transcribe(input_file)
        return result["text"]

    def save_transcription_to_db(self, filename, transcription):
        """
        Saves a transcription to the database.

        :param filename: Name of the audio file.
        :param transcription: Transcription text.
        """
        cursor = self.db_connector.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO transcriptions (filename, transcription, timestamp)
            VALUES (%s, %s, %s)
        """, (filename, transcription, timestamp))
        self.db_connector.commit()
        print(f"Transcription for {filename} saved to database.")

    def get_transcriptions(self):
        """
        Retrieves all transcriptions from the database.

        :return: A list of tuples containing (id, filename, transcription, timestamp).
        """
        cursor = self.db_connector.cursor()
        cursor.execute("SELECT * FROM transcriptions")
        records = cursor.fetchall()
        return records


if __name__ == "__main__":
   

    # Connect to your database
    conn = psycopg2.connect(
        host="localhost",
        database="transcription_db",
        user="your_username",
        password="your_password",
        cursor_factory=RealDictCursor
    )

    # Initialize the class
    recorder = AudioRecorderWithDatabase(conn)

    # Record and transcribe
    output_filename = "conversation.wav"
    duration = 10  # Record for 10 seconds

    recorder.record_audio(output_filename, record_seconds=duration)
    transcript = recorder.transcribe_audio(output_filename)

    # Save the transcription to the database
    recorder.save_transcription_to_db(output_filename, transcript)

    # Retrieve and display all transcriptions
    print("\nAll Transcriptions:")
    for record in recorder.get_transcriptions():
        print(record)
