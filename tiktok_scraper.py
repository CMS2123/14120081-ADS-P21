import os
import json
import time
import random
import logging
import asyncio
import nest_asyncio
import pandas as pd
from bs4 import BeautifulSoup
import nodriver as uc
from playwright.async_api import async_playwright
import yt_dlp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Allow nested event loops for environments like Jupyter Notebook
nest_asyncio.apply()

# ------------------------------------------
# Profile Scraper using nodriver + BeautifulSoup
# ------------------------------------------
async def scrape_tiktok_profile(username):
    try:
        print(f"Initiating scrape for TikTok profile: @{username}")
        browser = await uc.start(headless=False)
        print("Browser started successfully")

        page = await browser.get(f"https://www.tiktok.com/@{username}")
        print("TikTok profile page loaded successfully")

        await asyncio.sleep(10)
        print("Waited for 10 seconds to allow content to load")

        html_content = await page.evaluate('document.documentElement.outerHTML')
        print(f"HTML content retrieved (length: {len(html_content)} characters)")

        soup = BeautifulSoup(html_content, 'html.parser')
        print("HTML content parsed with BeautifulSoup")

        profile_info = {
            'username': soup.select_one('h1[data-e2e="user-title"]').text.strip() if soup.select_one('h1[data-e2e="user-title"]') else None,
            'display_name': soup.select_one('h2[data-e2e="user-subtitle"]').text.strip() if soup.select_one('h2[data-e2e="user-subtitle"]') else None,
            'follower_count': soup.select_one('strong[data-e2e="followers-count"]').text.strip() if soup.select_one('strong[data-e2e="followers-count"]') else None,
            'following_count': soup.select_one('strong[data-e2e="following-count"]').text.strip() if soup.select_one('strong[data-e2e="following-count"]') else None,
            'like_count': soup.select_one('strong[data-e2e="likes-count"]').text.strip() if soup.select_one('strong[data-e2e="likes-count"]') else None,
            'bio': soup.select_one('h2[data-e2e="user-bio"]').text.strip() if soup.select_one('h2[data-e2e="user-bio"]') else None
        }

        print("Profile information extracted successfully")
        return profile_info

    except Exception as e:
        print(f"An error occurred while scraping: {str(e)}")
        return None

    finally:
        if 'browser' in locals():
            browser.stop()
        print("Browser closed")

# ------------------------------------------
# Video Scraper using Playwright + yt_dlp
# ------------------------------------------

async def random_sleep(min_seconds=1, max_seconds=3):
    await asyncio.sleep(random.uniform(min_seconds, max_seconds))

async def extract_video_info(page, video_url):
    await page.goto(video_url, wait_until="networkidle")
    await asyncio.sleep(3)

    video_info = await page.evaluate("""
        () => {
            const getTextContent = (selector) => {
                const element = document.querySelector(selector);
                return element ? element.textContent.trim() : 'N/A';
            };

            const getTags = () => {
                const tagElements = document.querySelectorAll('a[data-e2e="search-common-link"]');
                return Array.from(tagElements).map(el => el.textContent.trim());
            };

            return {
                likes: getTextContent('[data-e2e="like-count"]'),
                comments: getTextContent('[data-e2e="comment-count"]'),
                shares: getTextContent('[data-e2e="share-count"]'),
                description: getTextContent('span.css-j2a19r-SpanText'),
                musicTitle: getTextContent('.css-pvx3oa-DivMusicText'),
                date: getTextContent('span[data-e2e="browser-nickname"] span:last-child'),
                tags: getTags()
            };
        }
    """)

    video_info['url'] = video_url
    logging.info(f"Extracted info for {video_url}: {video_info}")
    return video_info

def download_tiktok_video(video_url, save_path):
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),
        'format': 'best',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            logging.info(f"Video successfully downloaded: {filename}")
            return filename
    except Exception as e:
        logging.error(f"Error downloading video: {str(e)}")
        return None

async def scrape_channel_videos(username, max_videos=10):
    video_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        profile_url = f"https://www.tiktok.com/@{username}"
        await page.goto(profile_url, wait_until="networkidle")
        logging.info(f"Scraping TikTok profile: {profile_url}")

        video_urls = set()
        scroll_count = 0

        while len(video_urls) < max_videos:
            video_elements = await page.query_selector_all('div[data-e2e="user-post-item"] a')
            for elem in video_elements:
                href = await elem.get_attribute('href')
                if href and href not in video_urls:
                    video_urls.add(href)
                    logging.info(f"Found video URL: {href}")

                if len(video_urls) >= max_videos:
                    break

            await page.mouse.wheel(0, 5000)
            await random_sleep(1, 2)
            scroll_count += 1
            if scroll_count > 20:
                break

        logging.info(f"Collected {len(video_urls)} video URLs.")

        for idx, video_url in enumerate(video_urls):
            logging.info(f"Processing video {idx + 1}/{len(video_urls)}: {video_url}")

            video_info = await extract_video_info(page, video_url)
            video_data.append(video_info)

            save_path = os.path.join(os.getcwd(), username)
            os.makedirs(save_path, exist_ok=True)
            downloaded_file = download_tiktok_video(video_url, save_path)

            if downloaded_file:
                video_info['downloaded_file'] = downloaded_file

        await browser.close()

    df = pd.DataFrame(video_data)

    csv_filename = f"{username}_tiktok_videos.csv"
    df.to_csv(csv_filename, index=False)
    logging.info(f"Data saved to {csv_filename}")

    excel_filename = f"{username}_tiktok_videos.xlsx"
    df.to_excel(excel_filename, index=False)
    logging.info(f"Data saved to {excel_filename}")

    print(df.head())
    return df

# ------------------------------------------
# Main Function
# ------------------------------------------

async def main():
    # Part 1 - Profile Info
    usernames = ["manchesternews", "secretmanchester", "lifeinmanchester"]
    profile_tasks = [scrape_tiktok_profile(user) for user in usernames]
    profiles = await asyncio.gather(*profile_tasks)

    profile_results = {user: profile for user, profile in zip(usernames, profiles)}

    with open("tiktok_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profile_results, f, indent=2)
    print("✅ Profile data saved to tiktok_profiles.json")

    # Part 2 - Video Info (only one user for demo)
    selected_user = "lifeinmanchester"
    await scrape_channel_videos(selected_user, max_videos=5)

# Entry Point
if __name__ == "__main__":
    asyncio.run(main())
