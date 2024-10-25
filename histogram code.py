import os  # For working with directories and file paths
import numpy as np  # (Optional) You imported it, but it's not used here
from PIL import Image, ImageOps  # For image handling and processing

# Define a function to apply histogram equalization to an image
def histogram_equalization(img):
    """
    Perform histogram equalization on the luminance channel of an image 
    to enhance contrast while preserving the color information.

    Args:
        img (PIL.Image): The input image in RGB format.

    Returns:
        PIL.Image: The processed image with enhanced contrast.
    """
    # Convert the image to YCbCr color space (where Y is the luminance channel)
    img_yuv = img.convert('YCbCr')

    # Split the Y, Cb, and Cr channels (luminance, chroma-blue, chroma-red)
    y_channel, cb_channel, cr_channel = img_yuv.split()

    # Apply histogram equalization to the Y channel (luminance)
    y_channel_eq = ImageOps.equalize(y_channel)

    # Merge the equalized Y channel with the original Cb and Cr channels
    img_eq_yuv = Image.merge('YCbCr', (y_channel_eq, cb_channel, cr_channel))

    # Convert the YCbCr image back to RGB format
    img_eq = img_eq_yuv.convert('RGB')

    return img_eq  # Return the processed image

# Define the path where the original images are stored
original_images_path = r'C:\Users\XEON\Documents\trainer for VIT\dataset\validation\worms'

# Define the path to save the processed images
processed_images_path = r'D:\projects college\Dataset\Pre-Processed dataset\validation\processed_worms'

# Create the directory to save processed images if it doesn't exist
os.makedirs(processed_images_path, exist_ok=True)

# Process each image in the original images directory
for img_name in os.listdir(original_images_path):
    img_path = os.path.join(original_images_path, img_name)  # Full path to the image

    try:
        # Open the image and ensure it's in RGB mode
        img = Image.open(img_path).convert("RGB")

        # Print image type and size for confirmation
        print(f'Processing {img_name}: Image type: {type(img)}, Image size: {img.size}')

        # Apply histogram equalization to the image
        processed_img = histogram_equalization(img)

        # Save the processed image to the new directory
        processed_img.save(os.path.join(processed_images_path, img_name))

        # Print success message for the processed image
        print(f'Processed {img_name} successfully!')

    except Exception as e:
        # Print error message if processing fails for any image
        print(f"Error processing {img_name}: {e}")

# Print final message with the path where processed images are saved
print(f'Processed images saved to: {processed_images_path}')
