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
import rag
import click
from pynput import keyboard  # Add this import

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

SYSTEM_INSTRUCTION = """You are an intelligent assistant embedded directly within my digital writing notebook. 
You see what I see on my current page – my text, sketches, diagrams, and doodles – in real-time. You might only see
parts of the current document as it might have multiple pages and the pages might also be longer than what is shown. 
We'll communicate primarily through voice; I'll speak my thoughts and questions, and you'll respond with audio.

Your primary function is to be a REACTIVE assistant. This means:
1.  DO NOT speak or offer suggestions unless I ask you a question or give you a command.
2.  When I do ask something, be concise and directly address my query.

You have tools to help me:
-   You can search my other notebooks for relevant information. If I ask a question where searching might be helpful, you can use this tool.
-   You can create new documents for me. ONLY use this tool if I explicitly ask you to create a document.

Regarding information from tools:
-   NEVER mention or refer to any document, note, or piece of information unless it was explicitly found and returned by the 'notebook_search' tool in the current interaction. Do not assume or hallucinate the existence of documents.
-   If a search yields no results, state that clearly.
-   When referring to another notebook you found, ALWAYS state its name.

Your main role is to help me think, create, and refine my work within this notebook, based on what you see on the current page and what I ask.
Interpret everything visually, even rough sketches, as part of my creative process when I ask for your input.
Your goal is to be a helpful, CONCISE, creative, and critical partner, encouraging me to explore ideas further and enhance my work directly within this notebook environment, but only when prompted.
Avoid summarizing or describing the content unless I specifically ask. Instead, provide new insights or perspectives based on what you see and hear, in response to my questions.
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

db = rag.get_db()


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


def notebook_search(search_text: str, top_k: int = 3) -> dict:
    """Search for a text string in other notebooks using vector similarity.

    This function searches a database of embeddings from other user notebooks.
    It returns a list of the top_k most relevant text chunks. Only text chunks that
    are above a certain relevance threshold are returned. It is important to phrase the
    search_text such that it has rich semantic that we can search for in the other notebooks.

    Args:
        search_text: The text to search for.
        top_k: The maximum number of results to return.

    Returns:
        A dictionary containing the search status and results.
        Example on success:
        {
            "status": "success",
            "message": "Found 2 relevant snippets.",
            "results": [
                {'source_notebook': 'NotebookA', 'snippet': 'Relevant text...', 'page': 1},
                {'source_notebook': 'NotebookB', 'snippet': 'More relevant text...'}
            ]
        }
        Example on error:
        {
            "status": "error",
            "message": "Error during similarity search: <details>",
            "results": []
        }
        Example if DB not available:
        {
            "status": "warning",
            "message": "Database not available for search.",
            "results": []
        }
    """

    RELEVANCE_THRESHOLD = 0.7

    if not db:
        return {"status": "warning", "message": "Database not available for search.", "results": []}

    results_list = []
    try:
        result = db.similarity_search_with_score(search_text, k=top_k)
        for doc, score in result:
            if score > RELEVANCE_THRESHOLD:
                continue
            item = {"source_notebook": doc.metadata.get("source", "Unknown source"), "snippet": doc.page_content}
            if "page" in doc.metadata and doc.metadata["page"] is not None:
                try:
                    item["page"] = int(doc.metadata["page"])
                except ValueError:
                    # Handle case where page is not a valid integer, or log it
                    pass
            results_list.append(item)

        if results_list:
            return {
                "status": "success",
                "message": f"Found {len(results_list)} relevant snippets for '{search_text}'.",
                "results": results_list,
            }
        else:
            return {
                "status": "error",
                "message": "Did not find any relevant snippets for '{search_text}'.",
            }

    except Exception as e:
        print(f"Error during similarity search: {e}")  # Keep for server-side logging
        return {"status": "error", "message": f"Error during similarity search: {str(e)}", "results": []}


pya = pyaudio.PyAudio()


class AudioVideoLoop:
    def __init__(self, config, model, screen_number=0):
        self.screen_number = screen_number
        self.config = config
        self.model = model

        self.audio_in_queue = None
        self.out_queue = None

        self.is_ai_speaking = False
        self.session = None

        self.send_text_task = None
        self.receive_response_task = None
        self.play_audio_task = None
        self.last_screen_bytes = None

        self.is_sending_active = False  # Start with sending deactivated
        self.keyboard_listener_instance = None  # For managing the listener lifecycle

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
            # Consider logging this instead of raising an exception in a continuous loop
            click.echo(f"Error: Requested screen number {self.screen_number} not available.", err=True)
            return None
        monitor = sct.monitors[self.screen_number]

        i = sct.grab(monitor)

        png_image_bytes = mss.tools.to_png(i.rgb, i.size)

        if png_image_bytes == self.last_screen_bytes:
            return None  # Screen has not changed

        self.last_screen_bytes = png_image_bytes  # Update last screen

        img = PIL.Image.open(io.BytesIO(png_image_bytes))
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg", quality=85)
        # img.save("screen.jpeg", format="jpeg", quality=85) # For debugging
        image_io.seek(0)
        jpeg_image_bytes_for_payload = image_io.read()

        mime_type = "image/jpeg"
        return {"mime_type": mime_type, "data": base64.b64encode(jpeg_image_bytes_for_payload).decode()}

    async def get_screen(self):
        while True:
            if not self.is_sending_active:
                await asyncio.sleep(0.1)  # Check frequently if sending becomes active
                # Reset last_screen_bytes so the first frame after activation is always sent
                if self.last_screen_bytes is not None:
                    self.last_screen_bytes = None
                continue

            # Sending is active
            try:
                frame = await asyncio.to_thread(self._get_screen)  # _get_screen handles change detection

                if frame:  # frame is not None if screen changed
                    await self.out_queue.put(frame)
            except Exception as e:
                click.echo(f"Error in _get_screen: {e}", err=True)

            await asyncio.sleep(1.0)  # Interval to check for screen changes when active

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

        click.echo("Audio listener started. Press SPACE to toggle sending audio/video.")

        while True:
            if not self.is_sending_active:  # Check the flag
                await asyncio.sleep(0.1)  # Sleep briefly and re-check
                continue

            # Sending is active
            if self.is_ai_speaking:
                await asyncio.sleep(0.05)
                continue

            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except IOError as e:
                # This can happen if the stream is closed, e.g. during shutdown
                if self.audio_stream and not self.audio_stream.is_stopped():  # type: ignore
                    click.echo(f"Audio read error: {e}", err=True)
                break  # Exit loop if stream error
            except Exception as e:
                click.echo(f"Unexpected error in listen_audio: {e}", err=True)
                break

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
                                # print(
                                #     f"[Function Call Successful]: {name} -> Response to model: {api_tool_response_data}"
                                # )

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
                        # print(f"[INFO] Sent {len(tool_responses_for_model)} tool response(s).")

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

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.space:
                if self.is_sending_active:
                    click.echo("Stopped")
                    self.is_sending_active = False
                else:
                    click.echo("Start streaming .... ", nl=False)
                    self.is_sending_active = True

        except AttributeError:
            pass  # Ignore other key presses (e.g. regular character keys)

    async def start_keyboard_listener(self):
        """Runs the pynput keyboard listener in a separate thread."""

        def listener_thread_target():
            # self.keyboard_listener_instance is set here so it can be stopped
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                self.keyboard_listener_instance = listener
                listener.join()  # This blocks until listener.stop() is called
            self.keyboard_listener_instance = None  # Clear it after stopping
            click.echo("Keyboard listener stopped.")

        # Run the listener in a separate thread managed by asyncio
        await asyncio.to_thread(listener_thread_target)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=self.model, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=10)  # Increased size slightly

                # Start the keyboard listener
                tg.create_task(self.start_keyboard_listener())

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.get_screen())

                tg.create_task(self.receive_response())
                tg.create_task(self.play_audio())

                await send_text_task  # This means loop exits when send_text (e.g. user types 'q') exits
                # When send_text_task completes, it will trigger cancellation of the TaskGroup
                # due to exiting the `async with tg` block if not handled otherwise.
                # To make 'q' in send_text quit the whole app, we can raise CancelledError here.
                click.echo("Exiting application...")
                # The TaskGroup will handle cancelling other tasks.

        except asyncio.CancelledError:
            click.echo("Application was cancelled.")
            pass
        except ExceptionGroup as eg:  # TaskGroup raises ExceptionGroup
            click.echo("An error occurred in one of the tasks:", err=True)
            for i, exc in enumerate(eg.exceptions):
                click.echo(f"  Error {i + 1}: {type(exc).__name__}: {exc}", err=True)
                # traceback.print_exception(type(exc), exc, exc.__traceback__) # For more detail
        finally:
            click.echo("Cleaning up resources...")
            if self.audio_stream and not self.audio_stream.is_stopped():  # type: ignore
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                click.echo("Audio stream closed.")

            if self.keyboard_listener_instance:
                click.echo("Stopping keyboard listener...")
                self.keyboard_listener_instance.stop()

            # PyAudio termination is handled globally by pya.terminate() if needed,
            # but usually individual stream closure is sufficient.


@click.command()
@click.option(
    "--tools",
    "selected_tool_names_str",  # Use a different variable name to avoid conflict with the list
    type=str,
    help="Comma-separated list of tool names to enable (e.g., 'create_document,notebook_search'). Defaults to all available tools if not provided.",
    default=None,
    show_default="all available tools",
)
@click.option(
    "--screen",
    "screen_number",
    type=int,
    default=0,
    show_default=True,
    help="Screen number to capture.",
)
def main_cli(selected_tool_names_str, screen_number):
    """
    Runs the AudioVideoLoop with real-time audio/video streaming to Gemini
    and configurable tools.
    """
    # Define all available tools with their string names as keys
    available_tools_map = {
        "create_document": create_document,
        "notebook_search": notebook_search,
        # Add other tools here if you create more
        # "another_tool": another_tool_function,
    }

    selected_tool_functions = []
    if selected_tool_names_str:
        selected_tool_names_list = [name.strip() for name in selected_tool_names_str.split(",")]
        for tool_name in selected_tool_names_list:
            if tool_name in available_tools_map:
                selected_tool_functions.append(available_tools_map[tool_name])
            else:
                click.echo(f"Warning: Tool '{tool_name}' not found in available tools. Skipping.", err=True)
        if (
            not selected_tool_functions and selected_tool_names_list
        ):  # Check if input was given but no valid tools found
            click.echo("Warning: No valid tools selected via --tools argument. No tools will be enabled.", err=True)
    else:
        click.echo(f"No tools are enabled. See uv run main.py --help for more info.")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")),
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
        tools=selected_tool_functions,  # Use the selected tool functions
        system_instruction=SYSTEM_INSTRUCTION,
    )

    main_loop = AudioVideoLoop(config=config, model=MODEL, screen_number=screen_number)
    asyncio.run(main_loop.run())


if __name__ == "__main__":
    try:
        main_cli()
    except Exception as e:
        click.echo(f"Unhandled error in main_cli: {e}", err=True)
        traceback.print_exc()
    finally:
        if pya:
            pya.terminate()  # Terminate PyAudio system
            click.echo("PyAudio terminated.")
