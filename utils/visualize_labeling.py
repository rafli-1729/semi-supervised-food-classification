def visualize_labeling(
    label_source,
    base_image_folder: Path = None,
    classes_limit=15,
    examples_per_class=16
):
    class_names = sorted(list(label_source.keys()))[:classes_limit]
    counts = {c: len(label_source[c]) for c in class_names}

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(class_names)), [counts[c] for c in class_names])
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.title('Jumlah gambar per kelas')
    plt.tight_layout()
    plt.show()

    for cname in class_names:
        if isinstance(label_source, dict):
            img_files = label_source[cname][-examples_per_class:]
            if base_image_folder is None:
                raise ValueError("base_image_folder harus diset jika label_source berupa dict")
            img_paths = [base_image_folder / f for f in img_files]
        else:
            img_paths = list((label_source / cname).iterdir())[-examples_per_class:]

        if len(img_paths) == 0:
            continue

        n = len(img_paths)
        cols = min(8, n)
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(cols * 2, rows * 2.2))
        for i, p in enumerate(img_paths):
            try:
                img = Image.open(p)
                ax = plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(str(p)[-10:])
            except Exception as e:
                print(f"[warning] gagal buka gambar {p}: {e}")
        plt.suptitle(cname)
        plt.show()

visualize_labeling(
    label_dict, 
    base_image_folder=TRAIN_DIR, 
    classes_limit=15, 
    examples_per_class = 8
)
print("Done.")
