from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
from pytube import YouTube
from typing import Dict, Any


class YoutubeClient:
    def __init__(self):
        self.client = YouTubeTranscriptApi()

    def get_transcript(self, video_id: str) -> FetchedTranscript:
        transcript = self.client.fetch(video_id)
        return transcript

    def get_video_metadata(self, video_id: str, url: str = None) -> Dict[str, Any]:
        """
        Get video metadata using pytube.
        Returns: title, channel, duration (seconds), URL
        """
        try:
            if url:
                yt = YouTube(url)
            else:
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            
            return {
                "title": yt.title,
                "channel": yt.author,
                "duration": yt.length,  # in seconds
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        except Exception as e:
            # Fallback to basic info if pytube fails
            return {
                "title": "Unknown",
                "channel": "Unknown",
                "duration": 0,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }