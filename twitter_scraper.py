import os, csv, json, time, random, requests, datetime
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def get_today_date():
    return datetime.datetime.now().strftime("%m%d_%H%M%S")

def create_base_folder():
    folder = f"twitterscrape_{get_today_date()}"
    os.makedirs(folder, exist_ok=True)
    return folder

def download_image_task(args):
    url, media_folder, file_id = args
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                file_name = f"media_{file_id}.png"
                save_path = os.path.join(media_folder, file_name)
                img.save(save_path, format="PNG")
                return file_id, file_name
        except:
            pass
        time.sleep(2)
    return file_id, None

def perform_search(driver, keyword):
    print(f"\n🔍 Searching keyword: {keyword}")
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, '//input[@aria-label="Search query"]')))
    driver.execute_script("""
        let input = document.querySelector('input[aria-label="Search query"]');
        if (input) {
            input.focus();
            input.value = '';
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true }));
            input.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
        }
    """)
    time.sleep(1)
    search_box = driver.find_element(By.XPATH, '//input[@aria-label="Search query"]')
    search_box.send_keys(keyword)
    time.sleep(0.5)
    search_box.send_keys(Keys.ENTER)
    time.sleep(5)

    latest_tab = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "Latest"))
    )
    latest_tab.click()
    time.sleep(5)

def scroll_to_load(driver, max_scrolls=500):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(random.uniform(2, 4))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    return driver.page_source

def extract_engagement_stats(tweet):
    stats = {"Replies": "0", "Reposts": "0", "Likes": "0"}
    for btn in tweet.find_all("button", {"aria-label": True}):
        label = btn.get("aria-label", "").lower()
        if "reply" in label:
            match = re.search(r"(\d+)", label.replace(",", ""))
            if match:
                stats["Replies"] = match.group(1)
        elif "repost" in label:
            match = re.search(r"(\d+)", label.replace(",", ""))
            if match:
                stats["Reposts"] = match.group(1)
        elif "like" in label:
            match = re.search(r"(\d+)", label.replace(",", ""))
            if match:
                stats["Likes"] = match.group(1)
    return stats

def extract_tweets(html, media_folder, starting_media_id=1):
    soup = BeautifulSoup(html, 'html.parser')
    tweets = soup.find_all("article", {"role": "article"})
    all_data = []
    media_id = starting_media_id
    download_tasks = []

    for idx, tweet in enumerate(tweets, 1):
        print(f"\n🔍 Processing tweet {idx} ...")
        text_elem = tweet.find("div", {"data-testid": "tweetText"})
        tweet_text = text_elem.get_text(strip=True) if text_elem else ""
        print(f"📌 Tweet content: {tweet_text}")
        save_status = "Yes"
        skip_reason = ""

        user_elem = tweet.find("div", {"dir": "ltr"})
        username = user_elem.get_text(strip=True) if user_elem else "Unknown"

        time_elem = tweet.find("time")
        timestamp = time_elem["datetime"] if time_elem and time_elem.has_attr("datetime") else "Unknown"

        tweet_link_tag = tweet.find("a", href=re.compile(r"^/.+/status/\d+"))
        if tweet_link_tag:
            tweet_path = tweet_link_tag["href"]
            tweet_link = f"https://twitter.com{tweet_path}"
            username = tweet_path.split("/")[1]
        else:
            tweet_link = "Not Found"
            username = "Unknown"

        stats = extract_engagement_stats(tweet)

        media_imgs = tweet.find_all("img")
        for img in media_imgs:
            url = img.get('src', '')
            if not url or url.startswith("data:") or url.endswith(".svg"):
                continue
            if "emoji" in url or "profile_images" in url or "abs.twimg.com" in url:
                continue
            download_tasks.append((url, media_folder, media_id))
            media_id += 1

        quoted = tweet.find("div", {"data-testid": "tweet"})
        if quoted:
            quote_imgs = quoted.find_all("img")
            for img in quote_imgs:
                url = img.get("src", "")
                if not url or url.startswith("data:") or url.endswith(".svg"):
                    continue
                if "emoji" in url or "profile_images" in url or "abs.twimg.com" in url:
                    continue
                download_tasks.append((url, media_folder, media_id))
                media_id += 1

        all_data.append({
            "description": tweet_text,
            "location": "Manchester",
            "source": "Twitter",
            "link": tweet_link,
            "comments_count": stats.get("Replies", "0"),
            "reposts_count": stats.get("Reposts", "0"),
            "likes_count": stats.get("Likes", "0"),
            "media_files": [],
            "saved": save_status,
            "skip_reason": skip_reason,
            "username": username,
            "timestamp": timestamp
        })

    print(f"\n📥 Downloading images (total {len(download_tasks)})...")
    id_to_file = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image_task, task) for task in download_tasks]
        for future in as_completed(futures):
            file_id, file_name = future.result()
            if file_name:
                id_to_file[file_id] = file_name

    current_id = starting_media_id
    for tweet in all_data:
        count = 0
        while current_id in id_to_file:
            tweet["media_files"].append(id_to_file[current_id])
            count += 1
            current_id += 1
        if count == 0:
            tweet["skip_reason"] = "No image (kept)"

    return all_data, media_id

def save_to_csv(results, path, append=False):
    write_mode = 'a' if append else 'w'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, write_mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "description", "location", "source", "link",
            "comments_count", "reposts_count", "likes_count",
            "media_files", "saved", "username", "timestamp"
        ])
        if not append:
            writer.writeheader()
        for row in results:
            writer.writerow({
                "description": row["description"],
                "location": row["location"],
                "source": row["source"],
                "link": row["link"],
                "comments_count": row.get("comments_count", "0"),
                "reposts_count": row.get("reposts_count", "0"),
                "likes_count": row.get("likes_count", "0"),
                "media_files": ", ".join(row["media_files"]),
                "saved": row["saved"],
                "username": row.get("username", "Unknown"),
                "timestamp": row.get("timestamp", "Unknown")
            })

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    keywords = ["manchester event"]
    folder = create_base_folder()
    media_folder = os.path.join(folder, "media")
    os.makedirs(media_folder, exist_ok=True)
    csv_path = os.path.join(folder, "tweets_all.csv")
    json_path = os.path.join(folder, "tweets_all.json")

    driver = webdriver.Chrome()
    driver.get("https://twitter.com/login")
    input("🔐 Please log in and then press Enter to continue...")

    all_results = []
    media_id_counter = 1
    save_to_csv([], csv_path, append=False)

    for keyword in keywords:
        perform_search(driver, keyword)
        html = scroll_to_load(driver, max_scrolls=500)
        time.sleep(10)
        results, media_id_counter = extract_tweets(html, media_folder, media_id_counter)
        all_results.extend(results)
        save_to_csv(results, csv_path, append=True)
        print(f"✅ Saved {len(results)} results for keyword [{keyword}]")
        time.sleep(random.randint(10, 15))

    driver.quit()
    save_json(all_results, json_path)
    saved_count = sum(1 for r in all_results if r["saved"] == "Yes")
    print(f"\n✅ Total tweets scraped: {len(all_results)} | Saved: {saved_count} | Media saved in: {media_folder}")

    skip_stats = Counter(r["skip_reason"] for r in all_results if r["skip_reason"])
    print("\n📊 Skipped Tweet Stats:")
    for reason, count in skip_stats.items():
        print(f"{reason}: {count}")

if __name__ == "__main__":
    main()
