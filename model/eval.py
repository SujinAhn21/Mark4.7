# model/eval.py  

"""
[Deprecated: 평가용 샘플을 per_class_max=30으로 샘플링하던 로직]
# sampled_files = []; per_class_max = 30; class_counter = defaultdict(int); ...
"""
import os, sys, csv, glob, torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, pandas as pd, argparse
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, accuracy_score,
                             roc_auc_score, roc_curve, auc)
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
VILD_DIR = os.path.join(PROJECT_ROOT, 'vild')
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path: sys.path.append(p)

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder
from vild_head import ViLDHead
from vild_parser_student import AudioParser
from seed_utils import set_seed

def _find_dataset_index(mark_version):
    for p in [os.path.join(PROJECT_ROOT, f"dataset_index_{mark_version}.csv"),
              os.path.join(BASE_DIR,     f"dataset_index_{mark_version}.csv")]:
        if os.path.exists(p): return p
    raise FileNotFoundError("dataset_index CSV not found.")

def _find_student_weights(mark_version):
    enc_primary = [os.path.join(BASE_DIR, f"distilled_student_encoder_{mark_version}.pth"),
                   os.path.join(BASE_DIR, f"best_student_encoder_{mark_version}.pth")]
    head_primary = [os.path.join(BASE_DIR, f"distilled_student_head_{mark_version}.pth"),
                    os.path.join(BASE_DIR, f"best_student_head_{mark_version}.pth")]
    CWD = os.getcwd()
    extra_roots = [PROJECT_ROOT, CWD]
    enc_extra, head_extra = [], []
    for r in extra_roots:
        enc_extra += [os.path.join(r, f"distilled_student_encoder_{mark_version}.pth"),
                      os.path.join(r, f"best_student_encoder_{mark_version}.pth")]
        head_extra += [os.path.join(r, f"distilled_student_head_{mark_version}.pth"),
                       os.path.join(r, f"best_student_head_{mark_version}.pth")]
    glob_roots = [PROJECT_ROOT, "/content"]
    enc_glob, head_glob = [], []
    for r in glob_roots:
        enc_glob += glob.glob(os.path.join(r, f"**/distilled_student_encoder_{mark_version}.pth"), recursive=True)
        enc_glob += glob.glob(os.path.join(r, f"**/best_student_encoder_{mark_version}.pth"), recursive=True)
        head_glob += glob.glob(os.path.join(r, f"**/distilled_student_head_{mark_version}.pth"), recursive=True)
        head_glob += glob.glob(os.path.join(r, f"**/best_student_head_{mark_version}.pth"), recursive=True)
    enc_candidates = enc_primary + enc_extra + enc_glob
    head_candidates= head_primary+ head_extra+ head_glob
    enc = next((p for p in enc_candidates if os.path.exists(p)), None)
    hed = next((p for p in head_candidates if os.path.exists(p)), None)
    return enc, hed, enc_candidates, head_candidates

def evaluate(audio_label_list, seed_value=42, mark_version="mark4.7"):
    set_seed(seed_value)
    config = AudioViLDConfig(mark_version=mark_version)
    parser = AudioParser(config)
    device = config.device

    cls = config.classes; ncls = len(cls)
    idx2 = {i: l for i,l in enumerate(cls)}
    lab2 = {l: i for i,l in enumerate(cls)}

    enc = SimpleAudioEncoder(config).to(device)
    head= ViLDHead(config.embedding_dim, ncls).to(device)

    enc_path, head_path, ec, hc = _find_student_weights(mark_version)
    if not enc_path or not head_path:
        print(f"[ERROR] 모델 파일을 찾지 못했습니다.\n  - enc: {ec}\n  - head: {hc}")
        return
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))
    enc.eval(); head.eval()

    y_true, y_pred, y_prob, paths = [], [], [], []
    for path, tlabel in audio_label_list:
        if tlabel not in lab2: continue
        tidx = lab2[tlabel]
        segs = parser.load_and_segment(path)
        if not segs:
            print(f"[INFO] Skip (no valid segments): {os.path.basename(path)}"); continue
        total = torch.zeros(ncls, device=device); valid=0
        with torch.no_grad():
            for seg in segs:
                if seg is None or seg.ndim not in (3,4): continue
                if seg.ndim == 3: seg = seg.unsqueeze(0)
                seg = seg.to(device)
                feat = enc(seg)
                logits = head(feat.flatten(start_dim=1))
                prob = torch.softmax(logits, dim=-1).squeeze(0)
                total += prob; valid += 1
        if valid==0: continue
        avg = total/valid; pred = int(torch.argmax(avg).item())
        y_true.append(tidx); y_pred.append(pred); y_prob.append(avg.cpu().numpy()); paths.append(path)

    if not y_true:
        print("[WARN] 평가 가능한 예측 없음."); return

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    plot_dir = os.path.join(PROJECT_ROOT, "plots"); os.makedirs(plot_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(ncls)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cls)
    disp.plot(cmap=plt.cm.Blues); plt.title(f"Confusion Matrix ({mark_version})")
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{mark_version}.png")); plt.close()
    print("[INFO] Confusion matrix 저장 완료.")

    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                      labels=list(range(ncls)), zero_division=0)
    if ncls==2:
        rocA = roc_auc_score(y_true, y_prob[:,1])
    else:
        rocA = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    print("\n" + "="*30)
    print(f"      성능 평가 결과 ({mark_version})")
    print("="*30)
    print(f"  - Accuracy: {acc:.4f}")
    if isinstance(rocA, float): print(f"  - ROC AUC: {rocA:.4f}")
    print("\n클래스별 성능:")
    for i in range(ncls):
        print(f"  - {cls[i]} | P:{pre[i]:.4f} R:{rec[i]:.4f} F1:{f1[i]:.4f}")
    print("="*30 + "\n")

    data = {'Precision': list(pre)+[None], 'Recall': list(rec)+[None], 'F1-Score': list(f1)+[None]}
    df = pd.DataFrame(data, index=cls+['Overall'])
    df.loc['Overall','Accuracy']=acc; df.loc['Overall','ROC AUC']=rocA if isinstance(rocA,float) else None
    plt.figure(figsize=(8,4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="viridis", cbar=False, linewidths=.5)
    plt.title(f'Performance Metrics ({mark_version})'); plt.xticks(fontsize=12); plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f'performance_metrics_table_{mark_version}.png')); plt.close()

    plt.figure(figsize=(7,6))
    if ncls==2:
        fpr,tpr,_ = roc_curve(y_true, y_prob[:,1]); plt.plot(fpr,tpr, label=f'AUC={rocA:.4f}')
    else:
        for i in range(ncls):
            fpr,tpr,_ = roc_curve(y_true==i, y_prob[:,i]); A = auc(fpr,tpr)
            plt.plot(fpr,tpr, label=f'{cls[i]} AUC={A:.4f}')
    plt.plot([0,1],[0,1],'k--'); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({mark_version})'); plt.legend(loc="lower right")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, f'roc_curve_{mark_version}.png')); plt.close()

    csv_path = os.path.join(plot_dir, f'performance_summary_{mark_version}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write(f"# Performance Summary for {mark_version}\n\n")
        pd.DataFrame({'Metric':['Accuracy','ROC AUC' if ncls==2 else 'ROC AUC (Macro)'],
                      'Score':[acc, rocA if isinstance(rocA,float) else 'N/A']}).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({'Class':cls,'Precision':pre,'Recall':rec,'F1-Score':f1}).to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장: {csv_path}")

    pred_results = os.path.join(plot_dir, f'prediction_details_{mark_version}.csv')
    with open(pred_results, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Filename','True Label','Predicted Label'] + [f'Prob_{n}' for n in cls])
        for i in range(len(paths)):
            w.writerow([os.path.basename(paths[i]), idx2[y_true[i]], idx2[y_pred[i]]] + list(y_prob[i]))
    print(f"[INFO] 상세 예측 결과 CSV 저장: {pred_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, default="mark4.7")
    args = parser.parse_args()

    config = AudioViLDConfig(mark_version=args.mark_version)
    csv_path = _find_dataset_index(args.mark_version)
    pre_parser = AudioParser(config)

    # test 전량 사용, 샘플링 제거
    data = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p, l = row['path'], row['label']
            if l in config.classes and ("/data/test/" in p.replace("\\","/")):
                data.append((p, l))

    print(f"[INFO] test 전량 후보: {len(data)}개. 유효성 검사 후 평가 시작.")
    valid = []
    for path, label in data:
        segs = pre_parser.load_and_segment(path)
        if segs: valid.append((path, label))
        else: print(f"[WARN] 무효 파일 제외: {os.path.basename(path)}")

    print(f"[INFO] 유효 test 샘플: {len(valid)}개")
    if not valid:
        print("[ERROR] 평가할 유효 test 샘플이 없습니다.")
    else:
        evaluate(valid, seed_value=42, mark_version=args.mark_version)
    