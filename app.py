from flask import Flask, request, jsonify
import whisper  # Whisper for STT
from TTS.api import TTS  # Coqui TTS for TTS

app = Flask(__name__)

# Load the Whisper model (you can use "base", "small", "medium", or "large")
whisper_model = whisper.load_model("tiny")

# Load Coqui TTS model
# Replace "tts_models/en/ljspeech/tacotron2-DDC" with the appropriate model you want to use
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint for speech-to-text using Whisper"""
    if request.data:
        print("Received audio data, processing...")

        # Save the audio data to a file
        audio_file = 'received_audio.wav'
        try:
            with open(audio_file, 'wb') as f:
                f.write(request.data)
            print("Audio file saved successfully.")

            # Transcribe the audio file using Whisper
            print("Transcribing audio...")
            result = whisper_model.transcribe(audio_file)
            transcription = result['text']
            print("Transcription complete:", transcription)

            return jsonify({"transcription": transcription}), 200

        except Exception as e:
            print("Error during transcription:", str(e))
            return jsonify({"error": str(e)}), 500
    else:
        print("No audio data received")
        return jsonify({"error": "No audio data received"}), 400


@app.route('/synthesize', methods=['POST'])
def synthesize_audio():
    """Endpoint for text-to-speech using Coqui TTS"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        print(f"Received text for synthesis: {text}")

        # Generate speech audio
        output_audio_path = "synthesized_audio.wav"
        tts_model.tts_to_file(text=text, file_path=output_audio_path)
        print("Audio synthesis complete.")

        # Send the synthesized audio file back to the client
        with open(output_audio_path, "rb") as f:
            audio_data = f.read()

        return audio_data, 200, {
            "Content-Type": "audio/wav",
            "Content-Disposition": f"attachment; filename={output_audio_path}"
        }

    except Exception as e:
        print("Error during synthesis:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
