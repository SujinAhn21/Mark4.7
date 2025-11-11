# extraction/extract_soft_labels.py

"""
[Deprecated: train/val/test 전체에서 소프트라벨을 생성하던 방식]
# all rows -> soft_labels
"""
import os, sys, pickle, csv, torch, torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # extraction/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
VILD_DIR = os.path.join(PROJECT_ROOT, "vild")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR, MODEL_DIR):
    if p not in sys.path: sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser_teacher import AudioParser

def _in_split(path: str, allowed={"train","val"}) -> bool:
    p = path.replace("\\", "/")
    return any(f"/data/{s}/" in p for s in allowed)

class TeacherPredictor:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.encoder = SimpleAudioEncoder(config).to(device)
        self.classifier = ViLDTextHead(config).to(device)

        enc_candidates = [
            os.path.join(MODEL_DIR,    f"best_teacher_encoder_{config.mark_version}.pth"),
            os.path.join(PROJECT_ROOT, f"best_teacher_encoder_{config.mark_version}.pth"),
            os.path.join(BASE_DIR,     f"best_teacher_encoder_{config.mark_version}.pth"),
        ]
        cls_candidates = [
            os.path.join(MODEL_DIR,    f"best_teacher_classifier_{config.mark_version}.pth"),
            os.path.join(PROJECT_ROOT, f"best_teacher_classifier_{config.mark_version}.pth"),
            os.path.join(BASE_DIR,     f"best_teacher_classifier_{config.mark_version}.pth"),
        ]
        enc = next((p for p in enc_candidates if os.path.exists(p)), None)
        cls = next((p for p in cls_candidates if os.path.exists(p)), None)
        if enc is None or cls is None:
            raise FileNotFoundError("Teacher weights not found:\n" +
                                    "\n".join(enc_candidates+cls_candidates))
        self.encoder.load_state_dict(torch.load(enc, map_location=device))
        self.classifier.load_state_dict(torch.load(cls, map_location=device))

        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        prompts = [f"A sound of {c.replace('_', ' ')} in the room"
                   for c in config.get_classes_for_text_prompts()]
        self.text_emb = torch.tensor(text_model.encode(prompts), dtype=torch.float).to(device)

        self.encoder.eval(); self.classifier.eval()

    @torch.no_grad()
    def predict(self, mel_segments):
        if not mel_segments: return []
        batch = torch.stack(mel_segments, dim=0).to(self.device)
        region = self.encoder(batch)
        logits = self.classifier(region, self.text_emb)
        return F.softmax(logits, dim=1).cpu().tolist()

def extract_soft_labels(mark_version="mark4.7"):
    config = AudioViLDConfig(mark_version=mark_version)
    device = torch.device(config.device)

    csv_candidates = [
        os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv"),
        os.path.join(BASE_DIR,     f"dataset_index_{mark_version}.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print("[ERROR] CSV not found:\n  - " + "\n  - ".join(csv_candidates)); return 1

    teacher = TeacherPredictor(config, device)
    parser = AudioParser(config, segment_mode=True)

    out = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in tqdm(list(csv.DictReader(f)), desc=f"Soft labels ({mark_version})"):
            path = row['path']
            if not _in_split(path, {"train","val"}):  # test 제외
                continue
            segs = parser.load_and_segment(path)
            if not segs:
                print(f"[Warn] No segments: {path}"); continue
            out.append({"path": path, "soft_labels": teacher.predict(segs)})

    # 저장: 루트/ extraction/ model/ 미러
    name = f"soft_labels_{mark_version}.pkl"
    for d in (PROJECT_ROOT, BASE_DIR, MODEL_DIR):
        target = os.path.join(d, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'wb') as f: pickle.dump(out, f)
        print(f"[INFO] Saved: {target}")

    print(f"[완료] soft_labels(train+val) 저장: {len(out)} files")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mark_version', type=str, default="mark4.7")
    args = ap.parse_args()
    raise SystemExit(extract_soft_labels(args.mark_version))
    