import os
import asyncio
import base64
import io
import traceback

from dotenv import load_dotenv
import pyaudio
import PIL.Image
import mss
import remarkable_web_client
# import rag

from markdown_pdf import MarkdownPdf, Section


from google import genai
from google.genai import types

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"
API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODE = "screen"

SYSTEM_INSTRUCTION = """You are a helpful, CONCISE, creative, and critical assistant embedded within a digital writing notebook.
Your primary focus is the current content of this notebook, including text, diagrams, and doodles.
Interpret all visual elements, even sketches, as integral parts of the user's work.
Your goal is to help the user continue their writing and drawing, offering insightful suggestions, critiques, and creative ideas grounded in what is already present on the page.
Use the audio input to understand the user's spoken thoughts, questions, and instructions, treating it as a supplement to the visual information from the notebook.
Always refer to the notebook's content when formulating your responses.
Encourage creativity and help the user explore their ideas further within this notebook environment. 
Dont summarize or descrive the content on the notebook unless the user explicitly asks for it. Provide new insights based on the contents instead.
"""


client = genai.Client(http_options={"api_version": "v1beta"}, api_key=API_KEY)
try:
    remarkable_web_client = remarkable_web_client.RemarkableWebClient()
except Exception as e:
    print(
        f"Could not connect to the Remarkable Web Server. Is it connected to your laptop, running and have web server enabled in cloud storage settings?"
        f"Error: {e}"
    )
    exit(1)


def create_document(document_name: str, markdown_text: str) -> dict:
    """Create a text document in the user notebook.

    This function enables creating a document directly into the user notebook.
    Should be asked if the user asks the agent to create a document for him.

    Args:
     document_name: The descriptive name of the document
     markdown_text: The text to write into the document, formatted as markdown.

    Returns:
        A dictionary indicating the outcome.
        Example: {"status": "success", "message": "Document 'MyDoc.pdf' uploaded successfully."}
                 or {"status": "error", "message": "Failed to create PDF: <error details>"}
    """
    file_path = "upload.pdf"
    output_filename = f"{document_name}.pdf"
    try:
        pdf = MarkdownPdf()
        pdf.add_section(Section(f"{markdown_text}"))  # Ensure markdown_text is properly formatted if needed
        pdf.save(file_path)
        # print(f"Successfully converted '{document_name}' to PDF: {file_path}") # Keep for local debugging if needed
        remarkable_web_client.upload_file(file_path, output_filename)
        # print(f"Successfully uploaded '{output_filename}' to reMarkable.") # Keep for local debugging
        return {
            "status": "success",
            "message": f"Document '{output_filename}' created and uploaded successfully.",
            "filename": output_filename,
        }
    except Exception as e:
        error_message = f"Failed to create or upload document '{output_filename}': {str(e)}"
        # print(error_message) # Keep for local debugging
        return {"status": "error", "message": error_message, "filename": output_filename}


create_document_func = types.FunctionDeclaration(
    name="create_document",
    description="Creates a new document with the given name and markdown content in the user's notebook, then uploads it as a PDF.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "document_name": types.Schema(
                type=types.Type.STRING,
                description="The desired name for the new document (without file extension).",
            ),
            "markdown_text": types.Schema(
                type=types.Type.STRING,
                description="The content of the document, formatted as Markdown.",
            ),
        },
        required=["document_name", "markdown_text"],
    ),
)


tools = [create_document]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")),
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    tools=tools,
    system_instruction=SYSTEM_INSTRUCTION,
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
        self.receive_response_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True
            )

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
                image_data_bytes = base64.b64decode(msg["data"])
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

    async def receive_response(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn: asyncio.AsyncIterator[genai.LiveServerMessage] = self.session.receive()
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

                print(response)

                if response.tool_call:
                    tool_responses_for_model = []
                    for func_call in response.tool_call.function_calls:
                        id = func_call.id
                        name = func_call.name
                        args = dict(func_call.args) if func_call.args else {}
                        print(f"\n[Handling Function Call]: {name} with args: {args} and id={id}")
                        api_tool_response_data = {}

                        try:
                            # Check if the function exists in the global scope and is callable
                            if name in globals() and callable(globals()[name]):
                                actual_function_to_call = globals()[name]

                                function_execution_result = actual_function_to_call(**args)

                                api_tool_response_data = function_execution_result
                                print(
                                    f"[Function Call Successful]: {name} -> Response to model: {api_tool_response_data}"
                                )

                            else:
                                error_message = (
                                    f"Function '{name}' is not implemented or available in the global scope."
                                )
                                print(f"[Function Call Error]: {error_message}")
                                api_tool_response_data = {"error": error_message}

                        except Exception as e:
                            error_message = f"Error executing function '{name}': {str(e)}"
                            print(f"[Function Call Execution Error]: {error_message}\n{traceback.format_exc()}")
                            api_tool_response_data = {"error": error_message, "traceback": traceback.format_exc()}

                        function_response_for_model = types.FunctionResponse(
                            id=id, name=name, response=api_tool_response_data
                        )
                        tool_responses_for_model.append(function_response_for_model)

                    if tool_responses_for_model:
                        await self.session.send_tool_response(function_responses=tool_responses_for_model)
                        print(f"[INFO] Sent {len(tool_responses_for_model)} tool response(s).")

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

                tg.create_task(self.receive_response())
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
