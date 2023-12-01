import sys
from PIL import Image
import imageio
import os
from skimage.registration import phase_cross_correlation

import numpy as np

def scale_image(image, scale_factor):
    """
    Scale a PIL Image by a given factor.

    Parameters:
    - image (PIL.Image): The input image.
    - scale_factor (float): The scaling factor.

    Returns:
    - scaled_image (PIL.Image): The scaled image.
    """
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image

def crop_images(image_array, crop_size):
    cropped_images = []

    for img in image_array:
        width, height = img.size
        left = crop_size
        top = crop_size
        right = width - crop_size
        bottom = height - crop_size

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Append the cropped image to the list
        cropped_images.append(cropped_img)

    return cropped_images

def align_frames(frames):
    # Convert frames to numpy arrays
    frame_arrays = [np.array(frame) for frame in frames]

    # Use the first frame as the reference for alignment
    reference_frame = frame_arrays[0]

    # Initialize aligned frames list
    aligned_frames = [Image.fromarray(np.uint8(reference_frame))]

    # Align the frames to the reference frame using phase correlation
    for frame in frame_arrays[1:]:
        shift, error, diffphase = phase_cross_correlation(reference_frame.mean(axis=-1), frame.mean(axis=-1))

        # Apply the shift to the original frame
        aligned_frame = np.roll(frame, shift.astype(int), axis=(0, 1))

        # Convert the aligned frame back to PIL Image
        aligned_frame = Image.fromarray(np.uint8(aligned_frame))

        # Append the aligned frame to the list
        aligned_frames.append(aligned_frame)

    return aligned_frames

def slice_and_create_gif(input_path, output_gif_path):
    # Open the image
    image = Image.open(input_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the width of each slice (1/3 of the total width)
    slice_width = width // 3

    # Create a list to store individual frames
    frames = []

    # Slice the image and create frames
    for i in range(3):
        # Calculate the starting and ending coordinates for each slice
        start_x = i * slice_width
        end_x = start_x + slice_width

        # Slice the image
        slice_image = image.crop((start_x, 0, end_x, height))

        # Convert the slice to RGB mode (required for imageio)
        slice_image = slice_image.convert('RGB')

        # Append the slice to the frames list
        frames.append(slice_image)

    frames.append(frames[1])
    # Align the frames
    aligned_frames = align_frames(frames)

    # Crop the frames
    cropped_frames = crop_images(aligned_frames, 200)

    # Scale down frames
    scaled_frames = [scale_image(image, 0.2) for image in cropped_frames]

    # Save the aligned frames as an animated GIF
    imageio.mimsave(output_gif_path, scaled_frames, duration=3, loop=0)  # Adjust the duration as needed

    print(f"Aligned and animated GIF saved at {output_gif_path}")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py input_image1 [input_image2 ...]")
        sys.exit(1)

    # Process each input filename
    for input_image_path in sys.argv[1:]:
        # Make sure the input image file exists
        if not os.path.exists(input_image_path):
            print(f"Error: The input image file '{input_image_path}' does not exist.")
            continue

        output_filename = os.path.splitext(os.path.basename(input_image_path))[0] + "_wiggle.gif"
        output_gif_path = os.path.join(os.path.dirname(input_image_path), output_filename)

        # Call the function to slice, align, and create the animated GIF
        slice_and_create_gif(input_image_path, output_gif_path)