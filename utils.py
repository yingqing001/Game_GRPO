from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit # To handle line wrapping
import numpy as np
import matplotlib.pyplot as plt

def save_text_to_pdf(text_content, filename="game_log.pdf"):
    """
    Saves a long string of text to a professional-looking PDF file.

    Args:
        text_content (str): The entire text to be saved.
        filename (str): The name of the output PDF file.
    """
    # 1. Create a canvas object (the PDF document)
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter # Get page dimensions (in points)

    # 2. Set the font
    # We use Courier, a standard monospaced font available in all PDFs.
    font_name = "Courier"
    font_size = 9
    c.setFont(font_name, font_size)

    # 3. Create a text object
    text = c.beginText()
    # Set the starting position (1 inch from top-left corner)
    text.setTextOrigin(inch, height - inch)
    text.setFont(font_name, font_size)
    
    # 4. Process and add the text line by line
    # This handles page breaks automatically if the log is very long
    lines = text_content.split('\n')
    for line in lines:
        # Check if the text object has moved past the bottom margin
        if text.getY() < inch:
            # Draw the current text and start a new page
            c.drawText(text)
            c.showPage()
            # Reset font and text object for the new page
            c.setFont(font_name, font_size)
            text = c.beginText()
            text.setTextOrigin(inch, height - inch)
            text.setFont(font_name, font_size)
        
        text.textLine(line)

    # 5. Draw the final text object and save the file
    c.drawText(text)
    c.save()
    print(f"📄 Game log saved successfully to {filename}")


def save_text_to_image(text_content, filename="game_log.png"):
    """
    Saves a long string of text to a single image file.
    
    Args:
        text_content (str): The entire text to be saved.
        filename (str): The name of the output image file.
    """
    # --- Font Selection for Monospaced Look ---
    # We try to find a common monospaced font on the user's system.
    try:
        font = ImageFont.truetype("cour.ttf", size=15) # Windows
    except IOError:
        try:
            font = ImageFont.truetype("Menlo.ttc", size=15) # MacOS
        except IOError:
            # Fallback to a default font if others are not found
            print("Monospaced font not found. Using default PIL font.")
            font = ImageFont.load_default()

    # --- Image Creation ---
    # Create a temporary dummy image to calculate text size
    dummy_img = Image.new('RGB', (0, 0))
    draw = ImageDraw.Draw(dummy_img)

    # Calculate the bounding box of the text
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text_content, font=font)
    
    # Calculate text width and height
    text_width = right - left
    text_height = bottom - top
    
    # Add some padding
    padding = 20
    image_width = text_width + 2 * padding
    image_height = text_height + 2 * padding

    # Create the final image with a white background
    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw the text onto the image
    draw.multiline_text((padding, padding), text_content, font=font, fill='black')
    
    # Save the image
    img.save(filename)
    print(f"Game log saved successfully to {filename}")


def plot_trend(data, x_interval, path, plot_std=True):
    mean_values = np.mean(data, axis=1)
    std_values = np.std(data, axis=1)

    # --- X-axis: Generate the x-coordinates ---
    x_values = x_interval * np.arange(data.shape[0])  # Scale by x_interval for better spacing

    # --- Plotting  ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_values, mean_values, color='red', lw=2, label='Mean')
    if plot_std:
        ax.fill_between(x_values,
                        mean_values - std_values,
                        mean_values + std_values,
                        color='red',
                        alpha=0.2,
                    )

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True)

    # --- Save the plot as a PDF file ---
    # This is the line you need to add.
    # The plot will be saved in the same directory as your script.
    plt.savefig(path, bbox_inches='tight')
    print(f"Trend plot saved successfully to {path}")


if __name__ == "__main__":
    # sample_text = "\n".join([f"This is line {i}" for i in range(1, 101)])
    # save_text_to_image(sample_text, "sample_game_log.png")
    # save_text_to_pdf(sample_text, "sample_game_log.pdf")

    # import os
    # os.environ['HF_HUB_ENABLE_OFFLINE'] = '1'
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # from grpo import rollout
    # from train import load_tokenizer_and_models
    # import torch
    # from transformers import AutoTokenizer, AutoModelForCausalLM


    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # tok = AutoTokenizer.from_pretrained(model_name)
    # if tok.pad_token_id is None:
    #     tok.pad_token_id = tok.eos_token_id
    #     tok.pad_token = tok.eos_token
    # # tok.padding_side = "left"

    # policy = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.bfloat16
    # ).to(device)

    # _, reward, record_history = rollout(
    #     policy_model=policy,
    #     tokenizer=tok,
    #     device=device,
    #     )
    
    # print("Reward:", reward)

    # full_text = "\n".join(record_history)
    # save_text_to_image(full_text, "final_game_log.png")
    # save_text_to_pdf(full_text, "final_game_log.pdf")


    path = "./output/checkpoints/run_test_20251005_171619/eval_rewards.npy"
    data = np.load(path)
    save_path = path.replace(".npy", "_mean.pdf")
    plot_trend(data, x_interval=5, path=save_path, plot_std=False)