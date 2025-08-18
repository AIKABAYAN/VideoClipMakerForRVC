# test_video_builder.py
import os
from core.video_builder import build_video_multithread



def main():
    """
    Test script for the video builder.
    Finds assets in the 'test' directory and runs the build process.
    """
    TEST_DIR = "test"
    
    # Create the test directory if it doesn't exist
    os.makedirs(TEST_DIR, exist_ok=True)

    test_mp3 = None

    # Find the first mp3 file in the test directory
    for f in os.listdir(TEST_DIR):
        if f.lower().endswith(".mp3"):
            test_mp3 = os.path.join(TEST_DIR, f)
            break

    # Check if any images exist in the test directory
    has_images = any(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in os.listdir(TEST_DIR))

    if not test_mp3 or not has_images:
        print(f"⚠️  Please put at least one .mp3 file and some .jpg/.png images into the '{TEST_DIR}' folder to run the test.")
        return

    print("🚀 Running test build with assets from /test ...\n")

    # The song name is derived from the mp3 filename, without the extension.
    song_name = os.path.splitext(os.path.basename(test_mp3))[0]
    
    # *** MODIFIED LINE: Set the output folder to your desired path. ***
    output_folder = r"C:\Users\SIKABAYAN\Desktop\result2"

    # All parameters are now passed in a single 'settings' dictionary.
    settings = {
        "mp3_file": test_mp3,
        "song_name": song_name,
        "artist_name": "RVC by Sharkoded",
        "output_folder": output_folder,
        
        # The builder gets images from this folder.
        "bg_folder": TEST_DIR,
        
        # Video settings
        "resolution": (640, 360),
        "bg_mode": "Blur",
        "blur_level": 4,
        
        # We'll test the faster "Hybrid Turbo" mode.
        "fast_mode": True,
        
        # Visualizer options
        "include_visualizer": True,
        "visualizer_height": "25%", # Pass as string with '%'
        
        # Export options (we only need one for the test)
        "export_youtube": True,
        "export_shorts": True,
        
        # Animations are not needed for this basic test.
        "animations": None,
    }

    # Call the build function with the settings dictionary.
    build_video_multithread(settings)

    output_path = os.path.join(output_folder, f"{song_name}.mp4")
    print(f"\n✅ Test complete. Check {output_path} and console logs.")

if __name__ == "__main__":
    main()