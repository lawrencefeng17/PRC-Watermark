# take part of an image and translate it
from PIL import Image, ImageFilter
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser('Args')
parser.add_argument('-f', '--file', type=str, required=True)
parser.add_argument('-x', '--width', type=int, default=100, help='Width of the section to move')
parser.add_argument('-y', '--height', type=int, default=100, help='Height of the section to move')
parser.add_argument('-r', '--resize', action='store_true', help='Use resize method instead of centering')
parser.add_argument('-b', '--blur', action='store_true', help='Apply Gaussian blur to the image')
parser.add_argument('-s', '--sigma', type=float, default=2.0, help='Sigma/radius for Gaussian blur (higher = more blur)')
parser.add_argument('-n', '--noise', action='store_true', help='Add random Gaussian noise to the image')
parser.add_argument('--noise_level', type=float, default=10.0, help='Standard deviation of the Gaussian noise (higher = more noise)')
parser.add_argument('-o', '--output_dir', type=str, default='transformed', help='Output directory')
args = parser.parse_args()

image = Image.open(args.file)
dir = Path(args.file).parent

def move_top_left_to_center(img, section_width, section_height):
    """
    Move the top-left part of the image to the center.
    
    Args:
        img: PIL Image object
        section_width: Width of the section to move
        section_height: Height of the section to move
        
    Returns:
        PIL Image with the top-left section moved to the center
    """
    # Get image dimensions
    img_width, img_height = img.size
    
    # Ensure section dimensions don't exceed image dimensions
    section_width = min(section_width, img_width)
    section_height = min(section_height, img_height)
    
    # Create a new blank image with the same dimensions and mode
    result = Image.new(img.mode, (img_width, img_height), color=0)
    
    # Calculate the center position for the section
    center_x = (img_width - section_width) // 2
    center_y = (img_height - section_height) // 2
    
    # Crop the top-left section
    section = img.crop((0, 0, section_width, section_height))
    
    # Paste the section at the center of the new image
    result.paste(section, (center_x, center_y))
    
    return result

def crop_and_resize(img, section_width, section_height):
    """
    Crop the top-left part of the image and resize it to the original dimensions.
    
    Args:
        img: PIL Image object
        section_width: Width of the section to crop
        section_height: Height of the section to crop
        
    Returns:
        PIL Image with the top-left section cropped and resized to original dimensions
    """
    # Get image dimensions
    img_width, img_height = img.size
    
    # Ensure section dimensions don't exceed image dimensions
    section_width = min(section_width, img_width)
    section_height = min(section_height, img_height)
    
    # Crop the top-left section
    section = img.crop((0, 0, section_width, section_height))
    
    # Resize the cropped section to the original image dimensions
    resized_section = section.resize((img_width, img_height), Image.LANCZOS)
    
    return resized_section

def gaussian_blur(img, sigma=2.0):
    """
    Apply Gaussian blur to an image while preserving its overall appearance.
    
    Args:
        img: PIL Image object
        sigma: Blur radius/sigma value (higher = more blur)
        
    Returns:
        PIL Image with Gaussian blur applied
    """
    # Create a copy of the image to avoid modifying the original
    blurred_img = img.copy()
    
    # Apply Gaussian blur with the specified radius
    blurred_img = blurred_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    return blurred_img

def add_gaussian_noise(img, std_dev=10.0):
    """
    Add random Gaussian noise to an image.
    
    Args:
        img: PIL Image object
        std_dev: Standard deviation of the Gaussian noise (higher = more noise)
        
    Returns:
        PIL Image with Gaussian noise added
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img).astype(np.float64)
    
    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(0, std_dev, img_array.shape)
    
    # Add noise to the image array
    noisy_img_array = img_array + noise
    
    # Clip the values to be within the valid range [0, 255]
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    noisy_img = Image.fromarray(noisy_img_array)
    
    return noisy_img

# Apply the function to the loaded image
if __name__ == "__main__":
    result_image = image
    
    if args.blur:
        result_image = gaussian_blur(result_image, args.sigma)
        output_prefix = f"blurred_{args.sigma}"
    elif args.noise:
        result_image = add_gaussian_noise(result_image, args.noise_level)
        output_prefix = f"noisy_{args.noise_level}"
    elif args.resize:
        result_image = crop_and_resize(result_image, args.width, args.height)
        output_prefix = "resized"
    else:
        result_image = move_top_left_to_center(result_image, args.width, args.height)
        output_prefix = "centered"
    
    # Save the result with a new filename
    output_filename = f"{args.output_dir}/{output_prefix}_{Path(args.file).name}"
    result_image.save(output_filename)
    print(f"Image saved as {output_filename}")


