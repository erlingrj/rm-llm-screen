import requests
import os
import re
import time

# --- Configuration ---
DEVICE_IP = "10.11.99.1"  # Default reMarkable IP when connected via USB
BASE_URL = f"http://{DEVICE_IP}"
DOWNLOAD_FILE_TYPE = "pdf"  # We'll download documents as PDFs


def get_filename_from_headers(headers, default_name="document.pdf"):
    """
    Extracts filename from Content-Disposition header.
    Matches the logic in the provided JavaScript:
    const xx = e => {
        if (!e) return;
        const n = /attachment; filename="(.*)"/gm.exec(e);
        if ((n == null ? void 0 : n.length) === 2) return n[1];
    };
    """
    content_disposition = headers.get("content-disposition")
    if content_disposition:
        match = re.search(r'filename="([^"]+)"', content_disposition, re.IGNORECASE)
        if match:
            return match.group(1)
    return default_name


def sanitize_filename(name):
    """Basic sanitization for filenames to avoid issues with OS."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace potentially problematic characters (you might want to expand this)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Avoid names that are reserved on some systems
    if name.upper() in ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1"]:
        name = "_" + name
    # Limit length if necessary (optional)
    # MAX_FILENAME_LENGTH = 255
    # if len(name) > MAX_FILENAME_LENGTH:
    #     name_part, ext_part = os.path.splitext(name)
    #     name = name_part[:MAX_FILENAME_LENGTH - len(ext_part) - 1] + ext_part
    return name


def download_file_from_device(session, file_id, visible_name, file_type, local_folder_path):
    """Downloads a single file from the device."""
    download_url = f"{BASE_URL}/download/{file_id}/{file_type}"
    print(f"  Downloading {visible_name} (ID: {file_id}, Type: {file_type}) to {local_folder_path}...")
    try:
        response = session.get(download_url, timeout=60)  # Increased timeout for larger files
        response.raise_for_status()

        # Determine filename
        filename = get_filename_from_headers(response.headers, f"{sanitize_filename(visible_name)}.{file_type}")
        sanitized_filename = sanitize_filename(filename)  # Sanitize the final resolved name

        full_local_path = os.path.join(local_folder_path, sanitized_filename)

        with open(full_local_path, "wb") as f:
            f.write(response.content)
        print(f"  Successfully downloaded {sanitized_filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading file {file_id} ({visible_name}): {e}")
        return False
    except IOError as e:
        print(f"  Error saving file {file_id} ({visible_name}) to {local_folder_path}: {e}")
        return False


def get_folder_contents(session, folder_id):
    """Fetches the contents (files and subfolders) of a given folder ID."""
    list_url = f"{BASE_URL}/documents/"
    if folder_id != "root":
        list_url = f"{BASE_URL}/documents/{folder_id}"

    try:
        response = session.get(list_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contents for folder {folder_id}: {e}")
        return []
    except ValueError:  # Includes JSONDecodeError
        print(f"Error decoding JSON for folder {folder_id}. Response: {response.text}")
        return []


def download_all_recursive(session, folder_id, current_local_path):
    """
    Recursively downloads files and creates folders.
    """
    print(f"Processing folder ID: {folder_id} -> {current_local_path}")
    items = get_folder_contents(session, folder_id)

    if not items:
        print(f"  No items found or error fetching folder {folder_id}")
        return

    for item in items:
        item_id = item.get("ID")
        item_visible_name = sanitize_filename(item.get("VissibleName", f"Unnamed_{item_id}"))
        item_type = item.get("Type")

        if not item_id or not item_visible_name or not item_type:
            print(f"  Skipping item with missing critical information: {item}")
            continue

        if item_type == "CollectionType":  # It's a folder
            print(f" Entering folder: {item_visible_name}")
            new_local_path = os.path.join(current_local_path, item_visible_name)
            try:
                os.makedirs(new_local_path, exist_ok=True)
                download_all_recursive(session, item_id, new_local_path)
            except OSError as e:
                print(f"  Could not create directory {new_local_path}: {e}")
        elif item_type == "DocumentType":  # It's a file
            # The JavaScript code implies files also have a 'fileType' attribute.
            # However, the general download endpoint takes a type (pdf, epub, rmdoc).
            # We'll use DOWNLOAD_FILE_TYPE for documents.
            file_export_type = item.get("fileType", DOWNLOAD_FILE_TYPE).lower()
            if file_export_type not in ["pdf", "epub", "rmdoc"]:  # Ensure supported type
                file_export_type = DOWNLOAD_FILE_TYPE

            download_file_from_device(session, item_id, item_visible_name, file_export_type, current_local_path)
        else:
            print(f"  Unknown item type: {item_type} for {item_visible_name}")

        # Small delay to avoid overwhelming the device
        time.sleep(0.1)


def export_all_from_device(local_base_path="remarkable_files"):
    """
    Main function to start the download process from the device's root.
    :param local_base_path: The local directory where files will be saved.
    """
    if not os.path.exists(local_base_path):
        try:
            os.makedirs(local_base_path, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create base directory '{local_base_path}': {e}")
            return

    # Use a session object for connection pooling
    with requests.Session() as session:
        print(f"Starting download of all files to: {local_base_path}")
        print(f"Device URL: {BASE_URL}")
        print("Attempting to connect to device...")
        try:
            # Make a test request to check connectivity
            response = session.get(f"{BASE_URL}/documents/", timeout=5)
            response.raise_for_status()
            print("Successfully connected to device.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not connect to the device at {BASE_URL}.")
            print(f"Please ensure your reMarkable is connected via USB and Web UI is accessible.")
            print(f"Details: {e}")
            return

        download_all_recursive(session, "root", local_base_path)
        print("Download process finished.")


if __name__ == "__main__":
    # --- Set your desired local path here ---
    local_directory = "my_remarkable_backup"

    export_all_from_device(local_directory)
