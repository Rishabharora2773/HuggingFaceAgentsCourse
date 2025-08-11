from PIL import Image
import requests
from io import BytesIO
import os
from smolagents import CodeAgent, OpenAIServerModel

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
]

images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
    }
    response = requests.get(url,headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

# print(f"Downloaded {len(images)} images")

# print(os.getenv("OPENAI_API_KEY"))

model = OpenAIServerModel(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

agent = CodeAgent(model=model, tools=[], max_steps=20, verbosity_level=2)

response = agent.run(
    """
    Describe the costume and makeup that the comic character in these photos is wearing and return the description.
    Tell me if the guest is The Joker or Wonder Woman.
    """,
    images=images
)

print(response)

# Method 3: Display images using PIL's show method (opens in default image viewer)
def display_images_pil(images):
    """Display images using PIL's show method"""
    for i, image in enumerate(images):
        print(f"Opening image {i+1} in default viewer...")
        image.show()

# Example usage:
# print("\n=== Opening images with PIL show method ===")
# display_images_pil(images)