from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID = "dQw4w9WgXcQ"  # 테스트 영상 ID (Rickroll)

youtube = build("youtube", "v3", developerKey=API_KEY)

try:
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=VIDEO_ID,
        maxResults=5,
        textFormat="plainText"
    )
    response = request.execute()

    print("API 호출 성공!")
    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        print("댓글:", text)

except Exception as e:
    print("API 호출 실패:", e)
