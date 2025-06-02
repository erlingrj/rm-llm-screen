import requests

REMARKABLE_ADDR = "10.11.99.1"


class RemarkableWebClient:
    def __init__(self) -> None:
        response = requests.get(f"http://{REMARKABLE_ADDR}/", timeout=5)  # 5 second timeout
        # Check for a successful status code (e.g., 200-299)
        response.raise_for_status()
        print(f"Successfully connected to reMarkable at {REMARKABLE_ADDR}")

    def upload_file(self, file_path: str, document_name: str) -> bool:
        # The URL for the upload
        url = f"http://{REMARKABLE_ADDR}/upload"

        # Open the file in binary read mode
        try:
            with open(file_path, "rb") as f:
                # Define the multipart/form-data payload
                # The key 'file' corresponds to the -F "file=@..." part of the curl command
                files = {"file": (f"{document_name}.pdf", f)}

                # Make the POST request
                response = requests.post(url, files=files)

                # Check the response from the server
                response.raise_for_status()  # This will raise an exception for HTTP errors (4xx or 5xx)

                print(f"File '{document_name}' uploaded successfully!")
                print(f"Server Response: {response.status_code}")
                return True
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return False
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return False
