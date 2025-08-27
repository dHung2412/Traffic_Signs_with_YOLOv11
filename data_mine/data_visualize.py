from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict

# Constants
ROOT_DIR = Path(r"D:\Project\Traffic_signs\data_mine\images")
ANALYZE_DIR = Path(r"D:\Project\Traffic_signs\analyze")  # Chuyển thành Path object
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FOLDER_NAMES = {
    "0": "W.224",
    "2": "P.102",
    "5": "W.207",
    "8": "I.434a",
    "9": "R.303",
    "10": "P.130",
    "27": "W.225",
    "32": "P.123b",
    "46": "P.131a",
    "48": "W.210"
}

def count_images(folder_path: Path) -> int:
    return sum(1 for file in folder_path.iterdir() 
               if file.is_file() and file.suffix.lower() in IMG_EXTENSIONS)

def get_image_counts(root_dir: Path, folder_mapping: Dict[str, str]) -> Dict[str, int]:
    counts = {}
    for old_name, new_name in folder_mapping.items():
        folder_path = root_dir / old_name
        try:
            counts[new_name] = count_images(folder_path)
        except FileNotFoundError:
            print(f"[!] Directory {old_name} does not exist!")
    return counts

def plot_image_counts(counts: Dict[str, int], output_dir: Path) -> None:
    try:
        output_dir.mkdir(exist_ok=True)
    except PermissionError:
        print(f"[!] Error: No permission to create or access directory {output_dir}")
        return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.keys(), counts.values(), color="skyblue")
    plt.title("Số lượng ảnh theo từng loại biển báo", fontsize=14, fontweight="bold")
    plt.xlabel("Loại biển báo", fontsize=12)
    plt.ylabel("Số ảnh", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    
    output_path = output_dir / "image_counts.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[+] Biểu đồ đã được lưu tại: {output_path}")
    except Exception as e:
        print(f"[!] Error saving chart: {e}")
    
    plt.show()

def main():
    counts = get_image_counts(ROOT_DIR, FOLDER_NAMES)
    
    print("📊 Số lượng ảnh:")
    for label, count in counts.items():
        print(f"- {label}: {count} ảnh")
    
    plot_image_counts(counts, ANALYZE_DIR)

if __name__ == "__main__":
    main()