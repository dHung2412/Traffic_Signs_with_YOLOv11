import os

folder = r"D:\Project\Traffic_signs\data_mine\labels\48"
folder_name = os.path.basename(folder)

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                parts[0] = folder_name   # thay bằng "2"
                new_lines.append(" ".join(parts))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
