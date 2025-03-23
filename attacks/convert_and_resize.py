from PIL import Image
import argparse

parser = argparse.ArgumentParser('Args')
parser.add_argument('-f', '--file', type=str, required=True)
args = parser.parse_args()

image = Image.open(args.file)

# Convert the image to RGB (removing the Alpha channel)
rgb_image = image.convert("RGB")

# Resize the image to 512x512
resized_image = rgb_image.resize((512, 512))

# Save the new image
resized_image.save("output_image.jpg")  # Save it as a JPEG or PNG file

