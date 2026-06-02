from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt


def main():
    print(f"{"-" * 80}")
    print("READ PICTURES AND DISPLAY")
    print(f"{"-" * 80}")

    save_dir = os.path.join("results", "read_and_display")
    os.makedirs(save_dir, exist_ok=True)

    try:
        for r in range(0, 5):
            pil_img = Image.open(os.path.join("pics", f"{r+1}.jpg"))

            cv_img = cv2.imread(os.path.join("pics", f"{r+1}.jpg"))
            if cv_img is None:
                print(f"Error: cv2 failed to open file {r+1}.jpg")
                exit(1)

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title("Pillow")
            plt.imshow(pil_img)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("OpenCV")
            plt.imshow(cv_img)
            plt.axis("off")

            plt.savefig(os.path.join(save_dir, f"{r+1}.jpg"))

    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Figure saved to {save_dir}.")
        print("Execution finished.\n")


if __name__ == "__main__":
    main()
