import os

# Danh sách các thư mục cần đếm
folders = [r"D:\Project\Traffic_signs\data_mine\labels\0",
           r"D:\Project\Traffic_signs\data_mine\labels\2",
           r"D:\Project\Traffic_signs\data_mine\labels\5",
           r"D:\Project\Traffic_signs\data_mine\labels\8",
           r"D:\Project\Traffic_signs\data_mine\labels\9",
           r"D:\Project\Traffic_signs\data_mine\labels\10",
           r"D:\Project\Traffic_signs\data_mine\labels\27",
           r"D:\Project\Traffic_signs\data_mine\labels\32",
           r"D:\Project\Traffic_signs\data_mine\labels\46",
           r"D:\Project\Traffic_signs\data_mine\labels\48",           
           ]

total_txt_files = 0

for folder in folders:
    # Lấy danh sách tất cả các file trong thư mục
    files = os.listdir(folder)
    # Lọc ra những file có đuôi .txt
    txt_files = [f for f in files if f.endswith('.txt')]
    count = len(txt_files)
    print(f"Thư mục {folder} có {count} file .txt")
    total_txt_files += count

print(f"Tổng số file .txt: {total_txt_files}")
