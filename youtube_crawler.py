from googleapiclient.discovery import build


def fetch_youtube_comments(
    video_id: str,
    api_key: str,
    max_comments: int = 50,
) -> list[str]:
    """
    YouTube Data API v3를 사용하여 특정 영상의 상위 댓글 텍스트를 수집한다.
    """
    comments: list[str] = []

    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="relevance",
    )

    while request and len(comments) < max_comments:
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "").strip()
            if text:
                comments.append(text)
                if len(comments) >= max_comments:
                    break

        if "nextPageToken" in response and len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                textFormat="plainText",
                order="relevance",
            )
        else:
            request = None

    return comments
