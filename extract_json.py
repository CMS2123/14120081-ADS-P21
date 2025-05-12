import pandas as pd
import json

# --- Helper function to convert likes/comments/shares like "94.6K" into numbers ---
def parse_shorthand_number(value):
    if isinstance(value, str):
        value = value.strip().upper()
        if value.endswith("K"):
            return int(float(value[:-1]) * 1_000)
        elif value.endswith("M"):
            return int(float(value[:-1]) * 1_000_000)
        elif value.isdigit():
            return int(value)
        else:
            try:
                return int(float(value))
            except:
                return 0
    elif isinstance(value, (int, float)):
        return int(value)
    return 0

# --- Load TikTok Data ---
tiktok_df = pd.read_csv("".csv")

# Extract video ID from the filename
tiktok_df["video_id"] = tiktok_df["downloaded_file"].str.extract(r"(\d+)\.mp4")

# --- Process TikTok Events ---
tiktok_events = []
for _, row in tiktok_df.iterrows():
    event = {
        "event_name": row.get("description", "")[:50],
        "description": row.get("description", ""),
        "date_time": row.get("date", ""),
        "location": "Manchester",
        "source": "TikTok",
        "link": row.get("url", ""),
        "social_proof": {
            "likes": parse_shorthand_number(row.get("likes", 0)),
            "comments": parse_shorthand_number(row.get("comments", 0)),
            "shares": parse_shorthand_number(row.get("shares", 0))
        }
    }
    tiktok_events.append(event)

# --- Load Twitter Data ---
twitter_df = pd.read_csv("tweets_all.csv")

# --- Process Twitter Events ---
twitter_events = []
for _, row in twitter_df.iterrows():
    text = row.get("content", "")
    if isinstance(text, str) and any(keyword in text.lower() for keyword in ["event", "concert", "meet", "party", "launch"]):
        event = {
            "event_name": text[:50],
            "description": text,
            "date_time": row.get("date", ""),
            "location": "Manchester",
            "source": "Twitter",
            "link": row.get("url", ""),
            "social_proof": {
                "likes": parse_shorthand_number(row.get("likeCount", 0)),
                "comments": parse_shorthand_number(row.get("replyCount", 0)),
                "shares": parse_shorthand_number(row.get("retweetCount", 0))
            }
        }
        twitter_events.append(event)

# --- Merge and Save All Events ---
all_events = tiktok_events + twitter_events

# Save to JSON
with open("events_output_only_csv.json", "w", encoding="utf-8") as f:
    json.dump(all_events, f, indent=4, ensure_ascii=False)

print(f"✅ Done! {len(all_events)} events saved to 'events_output_only_csv.json'")
