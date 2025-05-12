import openai
import pytesseract
from PIL import Image
import os

# Set your OpenAI API key
# openai.api_key = "your-openai-api-key"
openai.api_key = ""

# OCR function
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

# GPT-4 correction function
def correct_text_with_gpt4(ocr_text):
    system_prompt = (
        "You are an assistant that cleans up messy OCR output from event posters. "
        "Fix grammar, spelling, and reconstruct full sentences. If possible, output structured event details "
        "like title, date, time, venue, ticket info, and website."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ocr_text}
        ],
        temperature=0.2
    )

    return response['choices'][0]['message']['content']

# Main pipeline
def process_folder(folder_path):
    results = []

    # Create an output folder if it doesn't exist
    output_folder = os.path.join(folder_path, "corrected_output")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpeg", ".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            print(f"\n🖼️ Processing: {filename}")
            raw_text = extract_text_from_image(image_path)
            corrected = correct_text_with_gpt4(raw_text)

            # Save to a .txt file
            output_filename = os.path.splitext(filename)[0] + "_corrected.txt"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(corrected)

            results.append({
                "filename": filename,
                "corrected_text": corrected
            })

            print("\n✅ --- Corrected Output ---\n")
            print(corrected)
    
    return results


# Run the process
if __name__ == "__main__":
    folder = "images"  # change to your actual image folder
    process_folder(folder)
