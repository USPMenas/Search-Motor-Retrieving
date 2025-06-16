from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from interface_gui import Ui_Dialog
import sys
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui

import os
import warnings
import cv2
import numpy as np
import yaml
from pathlib import Path
import faiss
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import csv
from transformers import CLIPProcessor, CLIPModel
import torch


warnings.filterwarnings("ignore", category=DeprecationWarning)


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

clip_index = faiss.read_index("index_store/index_clip_text.faiss")
clip_df_ids = pd.read_parquet("index_store/ids_clip_text.parquet")


# Our group of images
query_requests = {
    "R1": "3_4_poissons_eagleray_3310",
    "R2": "3_5_poissons_hammerhead_3495",
    "R3": "3_3_poissons_tigershark_3244",
    "R4": "1_2_chiens_boxer_1146",
    "R5": "1_4_chiens_goldenretriever_1423",
    "R6": "1_5_chiens_Rottweiler_1578",
    "R7": "4_3_singes_squirrelmonkey_4082",
    "R8": "4_2_singes_gorilla_4004",
    "R9": "4_1_singes_chimpanzee_3772"
}


def preprocess(img, target_size=256):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    normalized = padded.astype(np.float32) / 255.0
    return normalized


# Define target vector lengths
TARGET_LENGTHS = {
    "hist": 512,
    "orb": 500 * 32,
    "sift": 500 * 128
}

# Carrega o target_size ao iniciar
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
target_size = config["preprocessing"]["target_size"]


def fix_vector(vec, target_len):
    flat = vec.flatten()
    if flat.shape[0] >= target_len:
        return flat[:target_len]
    else:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:flat.shape[0]] = flat
        return padded


def extract_color_histogram(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def extract_orb(img):
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def extract_sift(img):
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


DESCRIPTORS = {
    "hist": {
        "extractor": extract_color_histogram,
        "index_path": "index_store/index_hist.faiss",
        "ids_path": "index_store/ids_hist.parquet",
        "metric": "l2"
    },
    "orb": {
        "extractor": extract_orb,
        "index_path": "index_store/index_orb.faiss",
        "ids_path": "index_store/ids_orb.parquet",
        "metric": "l2"
    },
    "sift": {
        "extractor": extract_sift,
        "index_path": "index_store/index_sift.parquet",
        "ids_path": "index_store/ids_sift.parquet",
        "metric": "l2"
    }
}


def compute_avg_precision_from_results(results, expected_class):
    hits = 0
    precisions = []
    for i, r in enumerate(results):
        if r["classe"] == expected_class:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0.0


def compute_r_precision(results, expected_class, R):
    relevant = [r for r in results if r["classe"] == expected_class]
    R_estimated = len(relevant)
    return sum([1 for r in results[:R_estimated] if r["classe"] == expected_class]) / max(R_estimated, 1)


def extract_class_from_image_id(image_id):
    # Extracts the class from a filename like 3_5_poissons_hammerhead_3495
    return image_id.split("_")[2]


def get_total_relevant_images(class_name, df_ids):
    return sum(df_ids["image_id"].apply(lambda x: extract_class_from_image_id(x) == class_name))


def compute_average_precision(relevance_list, total_relevant):
    hits = 0
    precisions = []
    for i, rel in enumerate(relevance_list):
        if rel:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.sum(precisions) / total_relevant if total_relevant else 0.0

def evaluate_vit_query(image_id, index, df_ids, feature_dir, top_ks=[50, 100]):
        class_name = extract_class_from_image_id(image_id)

        # Load ViT feature for the query image
        query_vector = None
        for root, _, files in os.walk(feature_dir):
            for f in files:
                if f.startswith(image_id) and f.endswith(".npy"):
                    query_vector = np.load(os.path.join(root, f)).astype(
                        "float32").reshape(1, -1)
                    break
            if query_vector is not None:
                break

        if query_vector is None:
            print(f"[!] Feature not found for {image_id}")
            return None

        result = {"Indice requête": image_id}
        total_relevant = get_total_relevant_images(class_name, df_ids)
        result["TopMax"] = total_relevant

        max_k = max(top_ks)
        distances, indices = index.search(query_vector, max_k + 10)

        retrieved_ids = [df_ids.iloc[i]["image_id"] for i in indices[0]]
        prefix = "_".join(image_id.split("_")[:4])
        retrieved_ids = [
            rid for rid in retrieved_ids if not rid.startswith(prefix)]

        for k in top_ks:
            rel_k_ids = retrieved_ids[:k]
            relevance_k = [1 if extract_class_from_image_id(
                rid) == class_name else 0 for rid in rel_k_ids]

            precision = sum(relevance_k) / k
            recall = sum(relevance_k) / total_relevant
            ap = compute_average_precision(relevance_k, total_relevant)

            result[f"P (Top{k})"] = round(precision, 3)
            result[f"R (Top{k})"] = round(recall, 3)
            result[f"AP (Top{k})"] = round(ap, 3)

        result["MaP (Top50)"] = result["AP (Top50)"]
        result["MaP (Top100)"] = result["AP (Top100)"]

        return result


class MainApp(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.current_image_path = None

        self.ui.btn_load_image.clicked.connect(self.load_image)
        self.ui.btn_preprocess.clicked.connect(self.preprocess_image)
        self.ui.btn_extract_features.clicked.connect(self.extract_features)
        self.ui.btn_build_index.clicked.connect(self.build_index)
        self.ui.btn_query_image.clicked.connect(self.query_image)
        self.ui.btn_show_topk.clicked.connect(self.show_topk_results)
        self.ui.btn_metrics.clicked.connect(self.evaluate_metrics)
        self.ui.btn_vit_evaluation.clicked.connect(self.run_vit_evaluation)
        self.ui.btn_text_search.clicked.connect(self.search_textual_query)
        self.ui.btn_text_visualize.clicked.connect(self.show_text_search_results)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path).scaled(self.ui.image_label.width(
            ), self.ui.image_label.height(), QtCore.Qt.KeepAspectRatio)
            self.ui.image_label.setPixmap(pixmap)
            self.ui.log_output.append(
                f"✅ Loaded image: {os.path.basename(file_path)}")

    def preprocess_image(self):
        if not self.current_image_path:
            self.ui.log_output.append("⚠️ No image loaded.")
            return

        img = cv2.imread(self.current_image_path)
        if img is None:
            self.ui.log_output.append("❌ Error to load the images.")
            return

        # --- PREPROCESSAMENTO ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        normalized = padded.astype(np.float32) / 255.0

        # Armazena para uso futuro
        self.preprocessed_image = normalized

        # Exibe a imagem pré-processada no QLabel (convertida para QImage)
        qimg = QtGui.QImage((normalized * 255).astype(np.uint8),
                            target_size, target_size, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(self.ui.image_label.width(
        ), self.ui.image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.ui.image_label.setPixmap(pixmap)

        self.ui.log_output.append("✅ Pre processing concluded.")

    def extract_features(self):
        if not hasattr(self, "preprocessed_image"):
            self.ui.log_output.append(
                "⚠️ Please preprocess an image before extracting features.")
            return

        # back to 0-255 for OpenCV
        img_rgb = (self.preprocessed_image * 255).astype(np.uint8)

        # --- Extract descriptors ---
        try:
            hist = extract_color_histogram(img_rgb)
            orb_desc = extract_orb(img_rgb)
            sift_desc = extract_sift(img_rgb)
        except Exception as e:
            self.ui.log_output.append(f"❌ Feature extraction failed: {e}")
            return

        # --- Save to files ---
        output_dir = "features_temp"
        os.makedirs(output_dir, exist_ok=True)

        base = Path(self.current_image_path).stem

        np.save(os.path.join(output_dir, f"{base}_hist.npy"), hist)
        np.save(os.path.join(output_dir, f"{base}_orb.npy"),
                orb_desc if orb_desc is not None else np.zeros((0, 32)))
        np.save(os.path.join(output_dir, f"{base}_sift.npy"),
                sift_desc if sift_desc is not None else np.zeros((0, 128)))

        # --- Log results ---
        self.ui.log_output.append(
            f"✅ Features extracted and saved for: {base}")
        self.ui.log_output.append(f"  • Histogram shape: {hist.shape}")
        self.ui.log_output.append(
            f"  • ORB shape: {None if orb_desc is None else orb_desc.shape}")
        self.ui.log_output.append(
            f"  • SIFT shape: {None if sift_desc is None else sift_desc.shape}")

    def build_index(self):
        features_root = "features_temp"
        index_output = "index_temp"

        os.makedirs(index_output, exist_ok=True)

        descriptor_types = ["hist", "orb", "sift"]
        vectors = {desc: [] for desc in descriptor_types}
        ids = {desc: [] for desc in descriptor_types}

        # Loop through folders
        for file in os.listdir(features_root):
            if not file.endswith(".npy"):
                continue

            for desc in descriptor_types:
                if file.endswith(f"_{desc}.npy"):
                    try:
                        vec = np.load(os.path.join(features_root, file))
                        if vec is None or len(vec.shape) == 0:
                            continue

                        vec = fix_vector(vec, TARGET_LENGTHS[desc])
                        vectors[desc].append(vec.astype(np.float32))

                        ids[desc].append({
                            "path": file,
                            "vector_dim": vec.shape[0],
                            "classe": file.split("_")[0]
                        })

                    except Exception as e:
                        self.ui.log_output.append(
                            f"❌ Error loading {file}: {e}")
                    break

        # Build FAISS index for each descriptor
        for desc in descriptor_types:
            vec_list = vectors[desc]
            if not vec_list:
                self.ui.log_output.append(
                    f"⚠️ No vectors for descriptor: {desc}")
                continue

            X = np.vstack(vec_list).astype(np.float32)
            d = X.shape[1]

            index = faiss.IndexFlatL2(d)
            index.add(X)

            faiss.write_index(index, f"{index_output}/index_{desc}.faiss")
            df_ids = pd.DataFrame(ids[desc])
            df_ids.to_parquet(f"{index_output}/ids_{desc}.parquet")

            self.ui.log_output.append(
                f"✅ Index saved for '{desc}': {len(vec_list)} vectors")

    def query_image(self):
        if not hasattr(self, "preprocessed_image"):
            self.ui.log_output.append("⚠️ No preprocessed image available.")
            return

        # weights (can be customized)
        ensemble_config = {"hist": 0.5, "orb": 0.5}
        top_k = 5

        img = (self.preprocessed_image * 255).astype(np.uint8)

        combined_scores = {}
        combined_meta = {}

        for desc, weight in ensemble_config.items():
            if desc not in DESCRIPTORS:
                continue

            try:
                vec = DESCRIPTORS[desc]["extractor"](img)
                if vec is None or vec.size == 0:
                    continue

                vec = fix_vector(vec, TARGET_LENGTHS[desc]).reshape(
                    1, -1).astype(np.float32)

                index = faiss.read_index(DESCRIPTORS[desc]["index_path"])
                ids_df = pd.read_parquet(DESCRIPTORS[desc]["ids_path"])

                distances, indices = index.search(vec, top_k * 5)
                for i, idx in enumerate(indices[0]):
                    if idx == -1:
                        continue
                    item = ids_df.iloc[idx]
                    key = item["path"]
                    cls = item["classe"]
                    score = distances[0][i]

                    if key not in combined_scores:
                        combined_scores[key] = 0
                        combined_meta[key] = {"classe": cls}
                    combined_scores[key] += weight * score

            except Exception as e:
                self.ui.log_output.append(f"❌ Failed descriptor {desc}: {e}")
                continue

        # Aggregate and sort
        results = [
            {"image_id": key,
                "score": combined_scores[key], "classe": combined_meta[key]["classe"]}
            for key in combined_scores
        ]
        results = sorted(results, key=lambda x: x["score"])[:top_k]

        self.query_results = results
        self.ui.log_output.append("✅ Query completed. Top results:")
        for r in results:
            self.ui.log_output.append(
                f"  • {r['image_id']} | class: {r['classe']} | score: {r['score']:.3f}")

    def show_topk_results(self):
        if not hasattr(self, "query_results") or not self.query_results:
            self.ui.log_output.append("⚠️ No query results to display.")
            return

        query_img = cv2.imread(self.current_image_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        results = self.query_results
        n = len(results)

        plt.figure(figsize=(15, 3))

        # Show query image
        plt.subplot(1, n + 1, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")

        for i, result in enumerate(results):
            file_id = result["image_id"]
            parts = Path(file_id).name.split("_")

            classe = result["classe"]
            image_name = "_".join(parts[:-1]) + ".jpg"

            # Search recursively for the correct image file
            search_root = Path("MIR_DATASETS_B/MIR_DATASETS_B") / classe
            found_paths = list(search_root.rglob(image_name))

            if not found_paths:
                self.ui.log_output.append(
                    f"[!] Image not found: (tried fuzzy match) {classe}/**/{image_name}")
                continue

            img = cv2.imread(str(found_paths[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(1, n + 1, i + 2)
            plt.imshow(img)
            plt.title(f"{classe}\nScore: {result['score']:.1f}")
            plt.axis("off")

        plt.suptitle("Top-k Similar Images", fontsize=16)
        plt.tight_layout()
        plt.show()

    def evaluate_metrics(self):
        test_path = "test_queries.csv"
        if not os.path.exists(test_path):
            self.ui.log_output.append("❌ test_queries.csv not found.")
            return

        df = pd.read_csv(test_path)
        all_results = []

        for idx, row in df.iterrows():
            query_path = row["query_image_path"]
            expected_class = row["expected_class"]

            if not Path(query_path).exists():
                self.ui.log_output.append(f"[!] File not found: {query_path}")
                continue

            # Load + preprocess
            img = cv2.imread(query_path)
            if img is None:
                self.ui.log_output.append(f"[!] Could not load: {query_path}")
                continue

            preprocessed = preprocess(img, target_size=256)
            self.preprocessed_image = preprocessed
            self.current_image_path = query_path

            self.query_image()

            results = self.query_results if hasattr(
                self, "query_results") else []
            ap = compute_avg_precision_from_results(results, expected_class)
            rp = compute_r_precision(results, expected_class, R=5)

            all_results.append({
                "query": Path(query_path).name,
                "expected_class": expected_class,
                "average_precision": ap,
                "r_precision": rp
            })

            self.ui.log_output.append(
                f"✅ {Path(query_path).name} | AP: {ap:.3f} | R-Prec: {rp:.3f}")

        # Save to CSV
        pd.DataFrame(all_results).to_csv("evaluation_results.csv", index=False)
        self.ui.log_output.append("📄 Saved evaluation_results.csv")

    def run_vit_evaluation(self):
        import pandas as pd

        self.ui.log_output.append("🚀 Starting ViT evaluation...")

        # Index and metadata
        index = faiss.read_index("index_store/index_vit.faiss")
        df_ids = pd.read_parquet("index_store/ids_vit.parquet")
        feature_dir = "feature_output_vit"

        # Query requests — você pode adaptar isso para vir de CSV
        query_requests = {
            "R1": "3_4_poissons_eagleray_3310",
            "R2": "3_5_poissons_hammerhead_3495",
            "R3": "3_3_poissons_tigershark_3244",
            "R4": "1_2_chiens_boxer_1146",
            "R5": "1_4_chiens_goldenretriever_1423",
            "R6": "1_5_chiens_Rottweiler_1578",
            "R7": "4_3_singes_squirrelmonkey_4082",
            "R8": "4_2_singes_gorilla_4004",
            "R9": "4_1_singes_chimpanzee_3772"
        }

        results = []
        for req_name, image_id in query_requests.items():
            row = evaluate_vit_query(image_id, index, df_ids, feature_dir)
            if row:
                row["Indice requête"] = req_name
                results.append(row)
                self.ui.log_output.append(
                    f"✅ {req_name} | AP@50: {row['AP (Top50)']} | TopMax: {row['TopMax']}")
            else:
                self.ui.log_output.append(f"[!] Falha na query {req_name}")

        df_table2 = pd.DataFrame(results)[[
            "Indice requête", "R (Top50)", "R (Top100)", "P (Top50)", "P (Top100)",
            "AP (Top50)", "AP (Top100)", "MaP (Top50)", "MaP (Top100)", "TopMax"
        ]]

        df_table2.to_csv("vit_eval_results.csv", index=False)
        self.ui.log_output.append("📄 Result saved in vit_eval_results.csv")

    def search_textual_query(self):
        query = self.ui.input_text_query.text().strip()
        if not query:
            self.ui.log_output.append("⚠️ No query was made.")
            return

        self.ui.log_output.append(f"🔍 Searching for: '{query}'...")

        # Gera embedding do texto
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs).numpy().astype("float32")

        distances, indices = clip_index.search(text_embedding, 5)
        results = [clip_df_ids.iloc[i]["image_id"] for i in indices[0]]

        for i, (img_id, dist) in enumerate(zip(results, distances[0])):
            self.ui.log_output.append(f"• {img_id} | distance: {dist:.4f}")
        self.text_search_results = results


    def show_text_search_results(self):
        if not hasattr(self, "text_search_results"):
            self.ui.log_output.append("⚠️ No text results to show.")
            return

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        fig = plt.figure(figsize=(12, 3))
        spec = gridspec.GridSpec(1, len(self.text_search_results))

        for i, img_id in enumerate(self.text_search_results):
            parts = img_id.split("_")
            classe = parts[2]
            image_name = img_id + ".jpg"

            # Busca recursiva dentro da pasta da classe para localizar a imagem real
            search_root = Path("MIR_DATASETS_B/MIR_DATASETS_B") / classe
            found_paths = list(search_root.rglob(image_name))

            if not found_paths:
                self.ui.log_output.append(f"[!] Imagem não encontrada: {classe}/**/{image_name}")
                continue

            img = cv2.imread(str(found_paths[0]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax = fig.add_subplot(spec[i])
            ax.imshow(img_rgb)
            ax.set_title(img_id, fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
