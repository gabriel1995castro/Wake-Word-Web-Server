from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import queue
import pyaudio
import wave
import json
import os
import whisper
from vosk import Model, KaldiRecognizer
from agente_ollama import OllamaAgent
import time

app = Flask(__name__)
CORS(app)

command_queue = queue.Queue()
response_queue = queue.Queue()
status_queue = queue.Queue()

class WakeWordWebServer:
    def __init__(self, model, wake_word=['andador', 'lina']):
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.format = pyaudio.paInt16
        self.channels = 1
        self.wake_words = [w.lower() for w in wake_word]
        self.audio = pyaudio.PyAudio()
        self.vosk_model = Model(model_path=model)
        self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        self.recognizer.SetWords(True)
        self.whisper_model = whisper.load_model("medium")
        self.llm_agent = OllamaAgent(model="qwen2.5:14b")
        
        self.is_running = True
        self.current_status = "Aguardando wake word"
        
    def listening_ambiente(self):
        return self.audio.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=self.format,
            input=True,
            frames_per_buffer=self.chunk_size
        )
    
    def detect_wake_word(self, stream):
        self.current_status = "Aguardando wake word"
        status_queue.put({"status": self.current_status, "timestamp": time.time()})
        
        while self.is_running:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data=data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').lower()

                    for wake_word in self.wake_words:
                        if wake_word in text:
                            print(f"\nWake word detectada: '{wake_word}'")
                            status_queue.put({
                                "status": "Wake word detectada",
                                "wake_word": wake_word,
                                "timestamp": time.time()
                            })
                            return True
            except Exception as e:
                print(f"Erro na detecção: {e}")
                time.sleep(0.1)
                
        return False
    
    def record_command(self, stream, duration=10):
        self.current_status = f"Gravando comando ({duration}s)"
        status_queue.put({"status": self.current_status, "timestamp": time.time()})
        print(f"\n Gravando comando ({duration}s)...")
        
        frames = []
        chunks = int(self.sample_rate / self.chunk_size * duration)

        for i in range(chunks):
            if not self.is_running:
                break
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                progress = int((i + 1) * 100 / chunks)
                if (i + 1) % (chunks // 10) == 0:
                    status_queue.put({
                        "status": "Gravando",
                        "progress": progress,
                        "timestamp": time.time()
                    })
                    print(f"   Progresso: {progress}%", end='\r')
            except Exception as e:
                print(f"Erro na gravação: {e}")
        
        print("\nGravação finalizada!")
        status_queue.put({"status": "Gravação finalizada", "timestamp": time.time()})
        return frames
    
    def create_audio(self, frames, filename="command.wav"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return filename
    
    def executate_transcription(self, audio_file):
        self.current_status = "Transcrevendo áudio"
        status_queue.put({"status": self.current_status, "timestamp": time.time()})
        print("Transcrevendo áudio...")
        
        result = self.whisper_model.transcribe(
            audio_file,
            language='pt',
            fp16=False
        )
        
        transcription = result['text'].strip()
        print(f"Transcrição: '{transcription}'")
        return transcription
    
    def run(self, command_duration=10):
        stream = None
        try:
            stream = self.listening_ambiente()
            print("\n" + "=" * 60)
            print("SISTEMA DE WAKE WORD ATIVO")
            print("=" * 60)
            print(f"Wake words configuradas: {', '.join(self.wake_words)}")
            print(f"Duração de gravação: {command_duration}s")
            print("=" * 60 + "\n")

            while self.is_running:
                try:
                    if self.detect_wake_word(stream):
                        frames = self.record_command(stream=stream, duration=command_duration)
                        audio_file = self.create_audio(frames)
                        command = self.executate_transcription(audio_file=audio_file)
                        
                        command_queue.put({"command": command, "timestamp": time.time()})
                        
                        self.current_status = "Processando com LLM"
                        status_queue.put({"status": self.current_status, "timestamp": time.time()})
                        print("Processando com LLM...")
                        
                        llm_response = self.llm_agent.run(command)
                        print(f"Resposta: {llm_response}\n")
                        
                        response_queue.put({
                            "command": command,
                            "response": llm_response,
                            "timestamp": time.time()
                        })
                        
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        
                        self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                        self.recognizer.SetWords(True)
                        
                        print("\n" + "-" * 60)
                        print("Voltando a escutar wake word...")
                        print("-" * 60 + "\n")
                        
                except Exception as e:
                    print(f"Erro no ciclo: {e}")
                    time.sleep(1)

        except Exception as e:
            print(f"\nErro crítico no sistema: {e}")
            status_queue.put({"status": "Erro", "error": str(e), "timestamp": time.time()})
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            self.audio.terminate()
            print("\n Sistema encerrado")
    
    def stop(self):
        self.is_running = False
        print("\n Parando sistema...")


# Instância global do sistema
wake_word_system = None
system_thread = None


@app.route('/')
def index():
    return send_from_directory('.', 'client.html')

@app.route('/health', methods=['GET'])
def health_check():
    is_alive = system_thread.is_alive() if system_thread else False
    return jsonify({
        "status": "healthy",
        "system_running": is_alive,
        "current_status": wake_word_system.current_status if wake_word_system else "Parado"
    })

@app.route('/status', methods=['GET'])
def get_status():
    statuses = []
    while not status_queue.empty():
        statuses.append(status_queue.get())
    
    is_alive = system_thread.is_alive() if system_thread else False
    
    return jsonify({
        "system_running": is_alive,
        "current_status": wake_word_system.current_status if wake_word_system else "Parado",
        "updates": statuses
    })

@app.route('/commands', methods=['GET'])
def get_commands():
    commands = []
    while not command_queue.empty():
        commands.append(command_queue.get())
    return jsonify({"commands": commands})

@app.route('/responses', methods=['GET'])
def get_responses():
    responses = []
    while not response_queue.empty():
        responses.append(response_queue.get())
    return jsonify({"responses": responses})

@app.route('/start', methods=['POST'])
def start_system():
    global wake_word_system, system_thread
    
    if system_thread and system_thread.is_alive():
        return jsonify({"error": "Sistema já está rodando"}), 400
    
    data = request.json or {}
    model_path = data.get('model_path', '/home/gabriel/Desktop/teste_wake_word/vosk-model-pt-fb-v0.1.1-20220516_2113')
    wake_words = data.get('wake_words', ['andador'])
    command_duration = data.get('command_duration', 15)
    
    try:
        wake_word_system = WakeWordWebServer(
            model=model_path,
            wake_word=wake_words
        )
        
        system_thread = threading.Thread(
            target=wake_word_system.run,
            args=(command_duration,),
            daemon=True
        )
        system_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Sistema iniciado com sucesso",
            "config": {
                "wake_words": wake_words,
                "command_duration": command_duration
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_system():
    global wake_word_system
    
    if wake_word_system:
        wake_word_system.stop()
        return jsonify({
            "status": "success",
            "message": "Sistema parado"
        })
    return jsonify({"error": "Sistema não está rodando"}), 400


def auto_start_system():
    global wake_word_system, system_thread
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'vosk-model-pt-fb-v0.1.1-20220516_2113')    
    WAKE_WORDS = ["andador", "lina"]
    COMMAND_DURATION = 15
    
    print("\n" + "=" * 60)
    print("INICIANDO SISTEMA AUTOMÁTICO DE WAKE WORD")
    print("=" * 60)
    
    try:
        wake_word_system = WakeWordWebServer(
            model=MODEL_PATH,
            wake_word=WAKE_WORDS
        )
        
        system_thread = threading.Thread(
            target=wake_word_system.run,
            args=(COMMAND_DURATION,),
            daemon=True
        )
        system_thread.start()
        
        print(" Sistema de wake word iniciado automaticamente!")
        print(f" Wake words: {', '.join(WAKE_WORDS)}")
        print(f"  Duração de gravação: {COMMAND_DURATION}s")
        print("Servidor web rodando em http://0.0.0.0:5000")
        print("Acesse http://localhost:5000 no navegador")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"Erro ao iniciar sistema: {e}")
        print(" O servidor web continuará rodando, mas o wake word está inativo")


if __name__ == '__main__':
    
    if not os.path.exists('client.html'):
        print("\n  AVISO: Arquivo 'client.html' não encontrado!")
        print(" Certifique-se de que o arquivo está no mesmo diretório do script\n")
    
    
    auto_start_system()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )