# model/student_train_distillation.py

"""
[Deprecated: 라벨 전체를 모아 train_test_split(90/10)으로 다시 무작위 분할]
# train_indices, val_indices = train_test_split(..., test_size=0.1, random_state=seed_value)
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle, matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse, functools
print = functools.partial(print, flush=True)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))           # model
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VILD_DIR     = os.path.join(PROJECT_ROOT, "vild")
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")
for p in (PROJECT_ROOT, VILD_DIR, UTILS_DIR):
    if p not in sys.path: sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0,
                 path_encoder='encoder.pth', path_head='head.pth'):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.counter, self.best_score = 0, None
        self.early_stop, self.val_loss_min = False, float('inf')
        self.path_encoder, self.path_head = path_encoder, path_head
    def __call__(self, val_loss, encoder, head):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self._save(val_loss, encoder, head)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'[EarlyStopping] counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self._save(val_loss, encoder, head); self.counter = 0
    def _save(self, val_loss, encoder, head):
        if self.verbose:
            print(f'[EarlyStopping] Val loss {self.val_loss_min:.6f} -> {val_loss:.6f}. Saving...')
        torch.save(encoder.state_dict(), self.path_encoder)
        torch.save(head.state_dict(), self.path_head)
        self.val_loss_min = val_loss

class DistillationLoss(nn.Module):
    def __init__(self, T, alpha, ignore_index=-1):
        super().__init__()
        self.T, self.alpha = T, alpha
        self.hard = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_logits, soft_labels, hard_labels):
        valid = hard_labels != -1
        if not valid.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        loss_hard = self.hard(student_logits[valid], hard_labels[valid])
        s = F.log_softmax(student_logits[valid]/self.T, dim=1)
        t = F.softmax(soft_labels[valid]/self.T, dim=1)
        loss_soft = self.soft(s, t) * (self.T**2)
        return self.alpha*loss_soft + (1-self.alpha)*loss_hard

def load_labels(mark_version):
    """
    PROJECT_ROOT/{extraction, .}/hard|soft_labels_{ver}.pkl 에서 로드
    """
    cand_h = [
        os.path.join(PROJECT_ROOT, "extraction", f"hard_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, f"hard_labels_{mark_version}.pkl"),
    ]
    cand_s = [
        os.path.join(PROJECT_ROOT, "extraction", f"soft_labels_{mark_version}.pkl"),
        os.path.join(PROJECT_ROOT, f"soft_labels_{mark_version}.pkl"),
    ]
    hp = next((p for p in cand_h if os.path.exists(p)), None)
    sp = next((p for p in cand_s if os.path.exists(p)), None)
    if hp is None or sp is None:
        raise FileNotFoundError("hard/soft labels not found. Run extraction first.")
    with open(hp, "rb") as f: hard = pickle.load(f)
    with open(sp, "rb") as f: soft = pickle.load(f)
    smap = {e['path']: e['soft_labels'] for e in soft}
    samples = []
    for e in hard:
        path = e['path']
        if path in smap:
            h = torch.tensor(e['hard_labels'], dtype=torch.long)
            s = torch.tensor(smap[path], dtype=torch.float)
            if len(h) != len(s):
                print(f"[Warn] length mismatch: {path}"); continue
            samples.append((path, h, s))
    return samples

def _in_split(path: str, split: str) -> bool:
    p = path.replace("\\", "/")
    return f"/data/{split}/" in p

def collate_fn(batch, parser: AudioParser):
    mel_list, hard_list, soft_list = [], [], []
    for path, h, s in batch:
        segs = parser.load_and_segment(path)
        if not segs: continue
        k = min(len(segs), len(h))
        if k == 0: continue
        mel = torch.stack(segs[:k])
        mel_list.append(mel)
        hard_list.append(h[:k])
        soft_list.append(s[:k])
    if not mel_list:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    max_k = max(m.shape[0] for m in mel_list)
    num_c = soft_list[0].shape[1] if soft_list else 0
    for i in range(len(mel_list)):
        cur = mel_list[i].shape[0]
        if cur < max_k:
            mel_list[i]  = torch.cat([mel_list[i],  torch.zeros((max_k-cur,1,64,101))], dim=0)
            hard_list[i] = torch.cat([hard_list[i], torch.full((max_k-cur,), -1, dtype=torch.long)], dim=0)
            soft_list[i] = torch.cat([soft_list[i], torch.zeros((max_k-cur, num_c))], dim=0)
    return torch.stack(mel_list), torch.stack(hard_list), torch.stack(soft_list)

def train_student_with_distillation(seed_value=42, mark_version="mark4.7"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    device = torch.device(config.device)
    parser = AudioParser(config)

    samples = load_labels(mark_version)

    # 파일 경로 기반으로 학습/검증 엄격 분리
    train_data = [s for s in samples if _in_split(s[0], "train")]
    val_data   = [s for s in samples if _in_split(s[0], "val")]

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, parser))
    val_loader   = DataLoader(val_data,   batch_size=config.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, parser))

    encoder = SimpleAudioEncoder(config).to(device)
    head    = ViLDHead(config.embedding_dim, len(config.classes)).to(device)
    model   = nn.Sequential(encoder, nn.Flatten(start_dim=1), head).to(device)

    T, alpha = 4.0, 0.7
    crit = DistillationLoss(T=T, alpha=alpha, ignore_index=-1)
    opt  = optim.Adam(model.parameters(), lr=config.learning_rate)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    enc_path = f"distilled_student_encoder_{config.mark_version}.pth"
    head_path= f"distilled_student_head_{config.mark_version}.pth"
    stopper  = EarlyStopping(patience=10, verbose=True, path_encoder=enc_path, path_head=head_path)

    tr_hist, vl_hist = [], []
    print(f"[INFO] Student KD training for {mark_version} on {device}")

    for ep in range(config.num_epochs):
        model.train(); total=0.0
        for mb, hb, sb in tqdm(train_loader, desc=f"[Train {ep+1}]"):
            if mb.numel()==0: continue
            B,K,C,H,W = mb.shape
            mb = mb.view(B*K, C, H, W).to(device)
            hb = hb.view(B*K).to(device)
            sb = sb.view(B*K, -1).to(device)

            logits = model(mb)
            loss = crit(logits, sb, hb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        tr = total/max(1,len(train_loader)); tr_hist.append(tr)

        model.eval(); total=0.0
        with torch.no_grad():
            for mb, hb, sb in val_loader:
                if mb.numel()==0: continue
                B,K,C,H,W = mb.shape
                mb = mb.view(B*K, C, H, W).to(device)
                hb = hb.view(B*K).to(device)
                sb = sb.view(B*K, -1).to(device)
                logits = model(mb)
                loss = crit(logits, sb, hb)
                total += loss.item()
        vl = total/max(1,len(val_loader)); vl_hist.append(vl)
        print(f"\n[Epoch {ep+1}] Train {tr:.6f} | Val {vl:.6f}")

        stopper(vl, encoder, head)
        if stopper.early_stop:
            print("[INFO] Early stopping."); break

        prev = opt.param_groups[0]['lr']; sched.step(vl)
        new  = opt.param_groups[0]['lr']
        if new < prev:
            print(f"[LR] {prev:.6g} -> {new:.6g} (val={vl:.6f})")

    plots = os.path.join(PROJECT_ROOT, "plots"); os.makedirs(plots, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(tr_hist, label='Train'); plt.plot(vl_hist, label='Val')
    plt.title(f'Distilled Student Loss ({mark_version})'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    out = os.path.join(plots, f"loss_curve_distilled_student_{mark_version}.png")
    plt.savefig(out)
    print(f"[INFO] Saved loss curve: {out}")
    print(f"[INFO] Best saved: {enc_path}, {head_path} (val loss {stopper.val_loss_min:.6f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mark_version', type=str, default="mark4.7")
    args = ap.parse_args()
    train_student_with_distillation(mark_version=args.mark_version)
    