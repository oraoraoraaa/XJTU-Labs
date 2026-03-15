from PIL import Image
import os


def main():
    print(f"{"-" * 80}")
    print("SCALE AND MOVE PICTURES")
    print(f"{"-" * 80}")

    save_dir = os.path.join("results", "scale_and_move")
    os.makedirs(save_dir, exist_ok=True)

    try:
        cwd = os.getcwd()

        for r in range(0, 5):
            img = Image.open(os.path.join("pics", f"{r+1}.jpg"))
            w, h = img.size
            scaled_img = img.resize((w * 3, h * 3))

            result = Image.new("RGB", scaled_img.size)
            result.paste(scaled_img, (-2, 0))

            result.save(os.path.join(save_dir, f"{r+1}.jpg"))

    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Figure saved to {save_dir}.")
        print("Execution finished.\n")


if __name__ == "__main__":
    main()
