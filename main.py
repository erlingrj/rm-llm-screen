import os
import asyncio
import base64
import io
import traceback

from dotenv import load_dotenv
import pyaudio
import PIL.Image
import mss


from google import genai
from google.genai import types

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-exp-native-audio-thinking-dialog"
API_KEY="AIzaSyADOwZuCPzDEf-lx_PmGI_DpVvtA25b9kM"
DEFAULT_MODE = "screen"

SYSTEM_INSTRUCTION = """You are a helpful, CONCISE, creative, and critical assistant embedded within a digital writing notebook.
Your primary focus is the current content of this notebook, including text, diagrams, and doodles.
Interpret all visual elements, even sketches, as integral parts of the user's work.
Your goal is to help the user continue their writing and drawing, offering insightful suggestions, critiques, and creative ideas grounded in what is already present on the page.
Use the audio input to understand the user's spoken thoughts, questions, and instructions, treating it as a supplement to the visual information from the notebook.
Always refer to the notebook's content when formulating your responses.
Be proactive in identifying connections, potential next steps, or areas for elaboration based on the visual and textual context.
Encourage creativity and help the user explore their ideas further within this notebook environment. 
Never summarize the content on the notebook just for the sake of doing it. Provide new insights based on the contents instead.
"""

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ["GEMINI_API_KEY"]
)

tools = [
    types.Tool(google_search=types.GoogleSearch()),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        ),
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    tools=tools,
    system_instruction=SYSTEM_INSTRUCTION
)

pya = pyaudio.PyAudio()


class AudioVideoLoop:
    def __init__(self, screen_number=0):
        self.screen_number = screen_number

        self.audio_in_queue = None
        self.out_queue = None

        self.is_ai_speaking = False
        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_screen(self):

        sct = mss.mss()
        if self.screen_number >= len(sct.monitors):
            raise Exception("Requested screen number not available")
        monitor = sct.monitors[self.screen_number]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        img.save("screen.jpeg", format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            msg_mime_type = msg["mime_type"]
            if msg_mime_type == "image/jpeg":
                image_data_bytes = base64.b64decode(msg['data'])
                image_blob = types.Blob(mime_type=msg_mime_type, data=image_data_bytes)
                await self.session.send_realtime_input(video=image_blob)
            elif msg_mime_type == "audio/pcm":
                audio_blob = types.Blob(mime_type=msg_mime_type, data=msg["data"])
                await self.session.send_realtime_input(audio=audio_blob)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            if self.is_ai_speaking:
                await asyncio.sleep(0.05)
                continue
            
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            first_audio_chunk_in_turn = True
            async for response in turn:
                if data := response.data:
                    if first_audio_chunk_in_turn:
                        self.is_ai_speaking = True
                        first_audio_chunk_in_turn = False
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            self.is_ai_speaking = False
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    main = AudioVideoLoop(screen_number=0)
    asyncio.run(main.run())
