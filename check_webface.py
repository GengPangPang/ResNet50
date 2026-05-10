import os

root = "/home/rum/.cache/kagglehub/datasets/yakhyokhuja/webface-112x112/versions/1/webface_112x112"

exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

ids = [
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d))
]

total_images = 0
min_count = 10**9
max_count = 0
empty_ids = []

for identity in ids:
    folder = os.path.join(root, identity)
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

    count = len(images)
    total_images += count
    min_count = min(min_count, count)
    max_count = max(max_count, count)

    if count == 0:
        empty_ids.append(identity)

print("identities:", len(ids))
print("images:", total_images)
print("min images per id:", min_count)
print("max images per id:", max_count)
print("empty folders:", len(empty_ids))

if empty_ids:
    print("first empty ids:", empty_ids[:10])