# allevents.py

import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = 'https://allevents.in/manchester?ref=home-page'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def get_event_links():
    """Extract event links from the homepage."""
    print("🔗 Collecting event links...")
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    cards = soup.find_all('li', class_='event-card event-card-link')
    links = [card.get('data-link') for card in cards if card.get('data-link')]

    print(f"✅ Found {len(links)} event links.")
    return links


def scrape_event_content(link):
    """Scrape event content from individual event page."""
    try:
        response = requests.get(link, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.find('div', id='event-container', class_='eps-container')
        content = container.get_text(separator='\n', strip=True) if container else ""
        return content
    except Exception as e:
        print(f"[Error] Failed to scrape {link}: {e}")
        return ""


def extract_fields(text):
    """Extract structured event fields from text."""
    lines = text.split('\n')

    # Title
    title = next((line.strip() for line in lines if line.isupper() and len(line.split()) >= 3), "")

    # Date
    date_match = re.search(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\d{1,2}\s+\w+\s+\d{4}', text)
    date = date_match.group(0) if date_match else ""

    # Time
    time_match = re.search(r'\d{1,2}:\d{2}\s*(AM|PM)?\s*to\s*\d{1,2}:\d{2}\s*(AM|PM)?', text, re.IGNORECASE)
    time_str = time_match.group(0) if time_match else ""

    # Location
    location_match = re.search(r'(?:Area|Venue):?\s*(.+?)\n', text)
    location = location_match.group(1).strip() if location_match else ""

    # Price
    price_match = re.search(r'GBP\s*\d+', text)
    price = price_match.group(0) if price_match else ""

    # People interested
    interested_match = re.search(r'(\d{1,4}\+?)\s*people\s*are\s*Interested', text, re.IGNORECASE)
    interested = interested_match.group(1) if interested_match else ""

    return pd.Series([title, date, time_str, location, price, interested])


def main():
    # Step 1: Get event links
    event_links = get_event_links()

    # Step 2: Scrape each event
    event_data = []
    for idx, link in enumerate(event_links, 1):
        print(f"📄 Scraping {idx}/{len(event_links)}: {link}")
        content = scrape_event_content(link)
        event_data.append({'link': link, 'content': content})
        time.sleep(1)

    df = pd.DataFrame(event_data)
    df.to_csv('allevents_raw.csv', index=False)
    print("✅ Saved raw event content to 'allevents_raw.csv'")

    # Step 3: Parse structured fields
    df[['title', 'date', 'time', 'location', 'price', 'interested']] = df['content'].apply(extract_fields)
    df.to_csv('allevents_structured.csv', index=False)
    print("✅ Saved structured data to 'allevents_structured.csv'")


if __name__ == "__main__":
    main()
