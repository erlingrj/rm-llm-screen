# rm-screen-share

This project allows you to share your screen and audio with a Gemini model and receive audio responses.

## Prerequisites

*   **macOS:** This application has been tested on macOS.
*   **uv:** This project uses `uv` for package management. If you don't have `uv` installed, you can install it by following the instructions on the [official uv installation guide](https://github.com/astral-sh/uv#installation).
*   **Python:** Version 3.13 or higher (as specified in [`pyproject.toml`](pyproject.toml) and [`.python-version`](.python-version)).
* brew install portaudio


## Getting Started

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd rm-screen-share
    ```

2.  **Create a virtual environment and install dependencies using uv:**
    ```bash
    uv sync
    ```
3.  **Set up your API Key:**
    Create a file named `.env` in the root of the project directory.
    Add your Gemini API key to this file:
    ```
    // filepath: .env
    GEMINI_API_KEY=your_api_key_here
    ```

4.  **Run the application:**
    ```bash
    uv run main.py
    ```
    The application will start, and you can interact with it by typing messages in the terminal. It will capture the screen specified (default is screen 0, see `main.py` at the bottom) and your microphone input.

## How it Works
This works by periodically taking screenshots of your entire window and sending it up to Gemini togheter with your audio.
Gemini is instructed to treat the screenshots as coming from your notebook and to provide helpful guidance based on it.

Have your Remarkable connected to WiFi and turn on screen sharing. Open the Screen Share window and enlarge it so that it
covers the entire screen. Then run the main application. Make sure that the notebook is covering the entire screen
while you are conversing with Gemini.

The screenshots of the notebook are stored to a file called `screen.jpeg` so you can inspect what Gemini sees.


The system instruction for gemini is given at the top of [`main.py`](main.py). This can be experimented with.

## Troubleshooting
Make sure that VSCode or Terminal, or whatever program you are running the script from, has the permissions to
capture the screen and audio.

Refer to [`main.py`](main.py) for the core logic.