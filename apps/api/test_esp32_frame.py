
import os
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

stream_url = os.getenv("ESP32_STREAM_URL")

if not stream_url:
    raise RuntimeError("ESP32_STREAM_URL not found in .env")

print("Connecting to ESP32-CAM...")

with httpx.stream("GET", stream_url, timeout=20) as response:
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    match = re.search(r"boundary=([^;]+)", content_type)

    if not match:
        raise RuntimeError("MJPEG boundary not found")

    boundary = match.group(1).strip().strip('"').encode()
    buffer = bytearray()

    for chunk in response.iter_bytes():
        buffer.extend(chunk)

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9", start + 2) if start != -1 else -1

        if start != -1 and end != -1:
            frame = bytes(buffer[start:end + 2])
            break
    else:
        raise RuntimeError("No JPEG frame received")

with open("esp32_test.jpg", "wb") as file:
    file.write(frame)

print("SUCCESS: JPEG frame captured!")
print("Image size:", len(frame), "bytes")
print("Saved as: apps/api/esp32_test.jpg")
