from PIL import Image
import os
import shutil
import tempfile

Image.MAX_IMAGE_PIXELS = None

def create_gif_3k4k(folder_in: str, output_file: str, target_size:tuple[any, any]=(3000, 4000), duration:int=100) -> None:
    """
    Resize frames to 3k×4k, create a GIF, preserve original files, and clean up all temp files.
    """
    files = sorted([f for f in os.listdir(folder_in) if f.endswith((".png", ".jpg"))])

    # Temporary folder for resized frames
    tmp_dir = tempfile.mkdtemp()
    try:
        # Resize frames
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Resizing {f}...")
            img = Image.open(os.path.join(folder_in, f)).convert("RGBA")
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(tmp_dir, f))

        # Load resized frames
        resized_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith((".png", ".jpg"))])
        frames_resized = [Image.open(os.path.join(tmp_dir, f)) for f in resized_files]

        # Save GIF
        print("Saving final GIF...") 
        frames_resized[0].save(
            output_file,
            save_all=True,
            append_images=frames_resized[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved as {output_file}")

    finally:
        # Clean up temporary folder
        shutil.rmtree(tmp_dir)
        print("Temporary files cleaned up. Original files remain unchanged.")

# Example usage:
create_gif_3k4k("plots/frames", "output.gif")
