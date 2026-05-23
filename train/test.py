import os, sys, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_config import OUTPUT_MODEL_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(OUTPUT_MODEL_DIR, local_files_only=True)
qe = AutoModel.from_pretrained(os.path.join(OUTPUT_MODEL_DIR, "query_encoder"), local_files_only=True).to(DEVICE).eval()
re = AutoModel.from_pretrained(os.path.join(OUTPUT_MODEL_DIR, "response_encoder"), local_files_only=True).to(DEVICE).eval()

qa_q, qa_a, test_q = "今天天气怎么样？", "今天天气很好，适合出门散步。", "外面天气如何？"
print(f"问句: {test_q}\n测试问句: {qa_q}\n测试答句: {qa_a}")

with torch.no_grad():
    t = tok([test_q], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    q = tok([qa_q], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    a = tok([qa_a], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    v_t = F.normalize(qe(**t).last_hidden_state[:, 0].float(), p=2, dim=1)[0].cpu()
    v_q = F.normalize(qe(**q).last_hidden_state[:, 0].float(), p=2, dim=1)[0].cpu()
    v_a = F.normalize(re(**a).last_hidden_state[:, 0].float(), p=2, dim=1)[0].cpu()
print(f"双塔 — 与问句相似度: {torch.dot(v_t, v_q).item():.4f}  |  与答句相似度: {torch.dot(v_t, v_a).item():.4f}")