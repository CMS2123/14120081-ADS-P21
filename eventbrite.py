# eventbrite.py

import time
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# User-Agent headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}

def extract_event_urls():
    """Scrape the Eventbrite page to extract event URLs."""
    url = 'https://www.eventbrite.co.uk/d/united-kingdom--manchester/events/'
    print(f"Requesting: {url}")
    time.sleep(2)
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to get page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=lambda href: href and '/e/' in href)
    urls = []

    for link in links:
        href = link.get('href')
        if href.startswith('https://www.eventbrite.co.uk') and href not in urls:
            urls.append(href)

    print(f"Found {len(urls)} unique event URLs")
    return urls

def extract_layout_text(url):
    """Extract full layout text content from an event URL."""
    print(f"\nScraping: {url}")
    time.sleep(1.5)
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to retrieve page.")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    container = soup.find('div', class_=re.compile(r'Layout-module__layout'))

    if container:
        return container.get_text(separator='\n', strip=True)
    else:
        print("Container not found.")
        return None

def extract_event_fields(text):
    """Parse structured fields from raw layout text."""
    fields = {}
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Title
    for i, line in enumerate(lines):
        if not any(kw.lower() in line.lower() for kw in ['few tickets', 'sales end', 'going fast']) and len(line.split()) >= 3:
            fields['title'] = line
            break

    # Date/Time
    date_line = next((line for line in lines if re.search(r'\b\d{1,2} [A-Za-z]{3,9} \d{4}', line)), None)
    if date_line:
        fields['date_time'] = date_line

    # Location
    for i, line in enumerate(lines):
        if 'location' in line.lower() and i + 1 < len(lines):
            fields['location'] = lines[i+1]
            break

    # Organizer
    for i, line in enumerate(lines):
        if line.lower().startswith('by ') or 'organised by' in line.lower():
            fields['organizer'] = line.replace('By ', '').replace('Organised by', '').strip()
            break

    # Description
    if 'About this event' in lines:
        start_idx = lines.index('About this event') + 1
        end_idx = len(lines)
        for marker in ['Frequently asked questions', 'Tags']:
            if marker in lines:
                end_idx = lines.index(marker)
                break
        description = "\n".join(lines[start_idx:end_idx]).strip()
        if description:
            fields['description'] = description

    # FAQs
    if 'Frequently asked questions' in lines:
        faq_index = lines.index('Frequently asked questions') + 1
        tag_index = lines.index('Tags') if 'Tags' in lines else len(lines)
        faqs = "\n".join(lines[faq_index:tag_index]).strip()
        if faqs:
            fields['faqs'] = faqs

    # Tags
    if 'Tags' in lines:
        tag_index = lines.index('Tags') + 1
        tags = []
        for i in range(tag_index, len(lines)):
            if lines[i].lower().startswith("organised by"):
                break
            tags.append(lines[i])
        fields['tags'] = ", ".join(tags)

    return fields

def main():
    # Step 1: Extract event URLs
    event_urls = extract_event_urls()

    # Step 2: Extract raw layout text from each URL
    results = []
    for url in event_urls:
        content = extract_layout_text(url)
        results.append({
            'url': url,
            'content': content
        })

    # Save raw layout content to CSV
    df_raw = pd.DataFrame(results)
    df_raw.to_csv('event_layout_texts.csv', index=False)
    print("\n✅ Saved raw content to 'event_layout_texts.csv'")

    # Step 3: Parse structured fields from text
    parsed_rows = []
    for _, row in df_raw.iterrows():
        if pd.isna(row['content']):
            continue
        fields = extract_event_fields(row['content'])
        fields['url'] = row['url']
        parsed_rows.append(fields)

    df_parsed = pd.DataFrame(parsed_rows)
    df_parsed.to_csv("structured_event_fields.csv", index=False)
    print("✅ Saved structured event fields to 'structured_event_fields.csv'")

if __name__ == "__main__":
    main()
