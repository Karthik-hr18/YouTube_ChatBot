import re
from urllib.parse import parse_qs, urlsplit


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_video_id(video_url_or_id: str) -> str:
    """Accept a YouTube video ID or a supported YouTube URL."""
    value = video_url_or_id.strip()
    if VIDEO_ID_PATTERN.fullmatch(value):
        return value

    candidate = value if "://" in value else f"https://{value}"
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")

    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        path_parts = [part for part in parsed.path.split("/") if part]
        video_id = (
            path_parts[1]
            if len(path_parts) >= 2
            and path_parts[0] in {"embed", "shorts", "live"}
            else ""
        )

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("The YouTube URL does not contain a valid video ID.")
    return video_id


def format_timestamp(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
