"""
ae_pipeline_simple.py

顺序做三件事，但只输出最终一个结果 CSV：
1) 用 Azure GPT-4o 从 notes 里抽 AE
2) （可选）用 baseline 过滤掉 baseline 以内的 AE
3) 用你微调好的 MedCPT 模型映射到 CTCAE v5.0

最终输出一个表，列大致为：
MRN, Onset Date, Date Resolved, CTCAE, Grade,
Attr to Disease, AE Immune related?, Serious Y/N,
CTCAE_Mapped_Top1, Similarity_Top1, Final_CTCAE_Term
"""

import time
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import AzureOpenAI
from incremental_update import update_patient_history


# ================= 0. Azure GPT 客户端 =================
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://bionlp-ge.openai.azure.com/",
    api_key=""
)


# ================= 1. GPT 提取 AE 的 prompt =================
base_prompt = """
You are a clinical research assistant helping to extract adverse events (AEs) from clinical notes.
Note:
<text>
{text}
</text>

For each AE, extract the following fields **in JSON array format** (one object per AE):

- MRN (from the note)
- Onset Date: If a specific start date is mentioned, extract it directly; otherwise, use the clinic note date as the start date or estimate an onset date according to the notes. "Onset Date" MUST NEVER be "Unknown" or "unknown". For any AE, if no explicit onset date is mentioned, ALWAYS set "Onset Date" exactly to {doc_date} (or date (estimated)), never to "Unknown".
- Date Resolved: If a specific end date or resolution (“…has resolved”) is mentioned, extract it; for events like “weight loss → gain weight,” use the clinic note date as the end date. If the AE is described as ongoing, set end date to “ongoing.” If not mentioned, set end date to “unknown.”
- AE term (mapped to CTCAE terminology)
- Grade (must be 1 to 5) If grade is not explicitly stated, estimate it based on context (Grade 1 for mild, Grade 2 if moderate/intervention needed, Grade 3 if Severe pain; Grade 4 if Life-threatening).
- Attribution to Disease? One of [Unrelated, Unlikely, Possible, Probable, and Definite]
- Immune-related AE? (Yes/No): Mark “Yes” if the AE is immune-related (irAE) based on the following definition.
Definition of immune-related adverse events (irAEs):irAEs are adverse events relevant to immunotherapy, such as colitis, thyroiditis, hypophysitis, adrenalitis, myositis, myocarditis, encephalitis, pneumonitis, hepatitis, immunotherapy-induced diabetes mellitus, vitiligo, and similar conditions. If the AE is immune-mediated or commonly recognized as an irAE, mark “Yes”; otherwise, mark “No”.
- serious AE? (Yes/No) Mark “Yes” if the AE is considered serious (e.g., life-threatening, hospitalization, or significant disability); otherwise, “No.”

**Important**:

Use the note date (if known) to anchor temporal reasoning.

Do not ignore symptoms that are briefly mentioned, appear together with other events, or are described with mild tone. Even minor or vague symptoms should be extracted.

If multiple symptoms are listed in one sentence, treat them as distinct AEs, and extract each separately.

Also extract imaging-based AEs that are not explicitly labeled as diagnoses but can be inferred from radiology findings such as CT scans.

Note any symptoms or minor symptoms that are mentioned as resolved. it should still be extracted and recorded, with end date set to resolution date if known, or clinic note date otherwise.

If no adverse events are present, return an empty JSON array: []
Do NOT include any explanation. Only return the JSON array.

Return a JSON array. Each AE MUST be a JSON object with EXACTLY the following keys
(using the same spelling and capitalization):

[
  {{
    "MRN": "...",
    "Onset Date": "...",
    "Date Resolved": "...",
    "AE Term": "...",
    "Grade": "...",
    "Attribution to Disease": "...",
    "Immune-related AE": "Yes" or "No",
    "Serious AE": "Yes" or "No"
  }}
]

Use these keys EXACTLY as written. 
Do NOT add question marks, extra spaces, or any additional keys.
Do NOT change capitalization.

Patient info:
MRN: {mrn}
Document Date: {doc_date}
Document Name: {doc_name}
"""


# ============= 函数 1：GPT，从 notes CSV -> AE DataFrame（仅在内存） =============
def gpt_extract_ae(note_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(note_csv_path)[43:44]

    structured_results = []

    for i, row in df.iterrows():
        prompt = base_prompt.format(
            text=row["Document Text"],
            mrn=row["mrn"],
            doc_date=row["Document Date"],
            doc_name=row["Document Name"],
        )

        print(f"\n=== GPT Step | row {i} | MRN: {row['mrn']} ===")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 如果你的 deployment 名不同，这里改名字
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content.strip()

            # 解析 JSON 数组
            json_start = reply.find("[")
            json_end = reply.rfind("]") + 1
            json_text = reply[json_start:json_end]

            if not (json_text.startswith("[") and json_text.endswith("]")):
                print(f"⚠️ Row {i}: Not a valid JSON array. Skipping.")
                continue

            ae_list = json.loads(json_text)
            if not isinstance(ae_list, list) or len(ae_list) == 0:
                print(f"ℹ️ Row {i}: No AE extracted.")
                continue

            for ae in ae_list:
                ae["Document Date"] = row["Document Date"]
                ae["Document Name"] = row["Document Name"]
                structured_results.append(ae)

        except Exception as e:
            print(f"❌ Error on row {i}: {e}")
            continue

        time.sleep(1)

    ae_df = pd.DataFrame(structured_results)

    if ae_df.empty:
        print("⚠️ GPT 没有抽到任何 AE。")
        return ae_df

    # 标准化几列，后面 filter / mapping 要用
    ae_df["MRN"] = ae_df["MRN"].astype(str).str.strip()
    ae_df["CTCAE"] = ae_df["AE Term"].astype(str).str.strip().str.lower()
    ae_df["Grade"] = pd.to_numeric(ae_df["Grade"], errors="coerce")

    return ae_df


# ============= 函数 2：baseline filter（可选） =============
def filter_with_baseline(ae_df: pd.DataFrame, baseline_file: str | None) -> pd.DataFrame:
    """如果 baseline_file 是 None 或 ""，则直接返回 ae_df，不做任何过滤。"""
    if ae_df.empty:
        return ae_df

    if baseline_file is None or baseline_file == "":
        print("ℹ️ 未提供 baseline 文件，跳过 baseline 过滤。")
        return ae_df

    baseline_df = pd.read_excel(baseline_file)
    baseline_df.columns = baseline_df.columns.str.strip()

    # 关键列名（按你之前的脚本）
    subject_col = "Patient"
    ae_term_col = "Adverse Event Term (v5.0)"
    baseline_grade_col = "Grade"  # baseline AE grade 列
    ae_grade_col = "Grade"        # 我们 AE 表里的 Grade 列

    # baseline 标准化
    baseline_df[subject_col] = baseline_df[subject_col].astype(str).str.strip()
    baseline_df[ae_term_col] = (
        baseline_df[ae_term_col].astype(str).str.strip().str.lower()
    )
    baseline_df[baseline_grade_col] = (
        baseline_df[baseline_grade_col]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    merged = ae_df.merge(
        baseline_df[[subject_col, ae_term_col, baseline_grade_col]],
        how="left",
        left_on=["MRN", "CTCAE"],
        right_on=[subject_col, ae_term_col],
        suffixes=("", "_baseline"),
    )

    ae_grade = merged[ae_grade_col]
    baseline_grade = merged["Grade_baseline"].fillna(-1)

    keep_mask = ae_grade > baseline_grade
    filtered_df = merged[keep_mask]

    filtered_df = filtered_df[ae_df.columns]
    print(f"✅ baseline filter：从 {len(ae_df)} 条 AE 保留 {len(filtered_df)} 条")
    return filtered_df


# ============= 函数 3：MedCPT 映射 CTCAE（ae_df -> final_df） =============
def map_to_ctcae_medcpt(
    ae_df: pd.DataFrame,
    ctcae_dict_csv: str,
    medcpt_model_dir: str,
) -> pd.DataFrame:
    if ae_df.empty:
        print("⚠️ 没有 AE 可映射。")
        return ae_df

    # 读取 CTCAE 词表
    ctcae_df = pd.read_csv(ctcae_dict_csv)
    ctcae_df.columns = ctcae_df.columns.str.strip()
    ctcae_terms = (
        ctcae_df["CTCAE Term"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    # 加载 MedCPT
    print("⏳ 加载 MedCPT 模型...")
    tokenizer = AutoTokenizer.from_pretrained(medcpt_model_dir)
    model = AutoModel.from_pretrained(medcpt_model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 编码 CTCAE 词表（保存在 CPU，可以节省显存）
    def encode_list(texts):
        embs = []
        for t in texts:
            t = str(t)
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                norm_emb = F.normalize(cls_emb, p=2, dim=1)
                # 保存在 CPU，节省 GPU 显存
                embs.append(norm_emb[0].cpu())
        return torch.stack(embs)

    print("⏳ 编码 CTCAE 术语...")
    ctcae_embeddings_cpu = encode_list(ctcae_terms)

    # AE → top-3 CTCAE
    print("⏳ 匹配 AE → Top-3 CTCAE ...")
    top_k = 3
    topk_rows = []

    for ctcae_free in ae_df["CTCAE"]:
        inputs = tokenizer(
            ctcae_free,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            ae_emb = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)

        # ⭐⭐⭐ 关键修复：把 CPU embeddings 临时移动到 GPU
        ctcae_embeddings = ctcae_embeddings_cpu.to(device)

        sim = torch.mm(ae_emb, ctcae_embeddings.T).squeeze()
        topk_scores, topk_indices = torch.topk(sim, k=top_k)

        row = {}
        for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores), start=1):
            row[f"CTCAE_Mapped_Top{rank}"] = ctcae_terms[idx].title()
            row[f"Similarity_Top{rank}"] = float(score)
        topk_rows.append(row)

    topk_df = pd.DataFrame(topk_rows)
    df = pd.concat([ae_df.reset_index(drop=True), topk_df.reset_index(drop=True)], axis=1)

    # 精确匹配 + Final_CTCAE_Term
    ctcae_set = set(ctcae_terms)

    def exact_match(term):
        if isinstance(term, str) and term.lower() in ctcae_set:
            return term
        return None

    df["CTCAE_Mapped_Exact"] = df["CTCAE_Mapped_Top1"].apply(exact_match)
    df["CTCAE_Mapped_By"] = df["CTCAE_Mapped_Exact"].apply(
        lambda x: "exact" if x is not None else "semantic"
    )
    df["Final_CTCAE_Term"] = df["CTCAE_Mapped_Exact"].combine_first(
        df["CTCAE_Mapped_Top1"]
    )

    # 重命名列
    df = df.rename(
        columns={
            "Attribution to Disease": "Attr to Disease",
            "Immune-related AE": "AE Immune related?",
            "Serious AE": "Serious Y/N",
        }
    )

    # 去重
    df = df.drop_duplicates(
        subset=["MRN", "Onset Date", "CTCAE", "Grade", "Final_CTCAE_Term"]
    )

    final_cols = [
        "MRN",
        "Onset Date",
        "Date Resolved",
        "CTCAE",
        "Grade",
        "Attr to Disease",
        "AE Immune related?",
        "Serious Y/N",
        "CTCAE_Mapped_Top1",
        "Similarity_Top1",
        "Final_CTCAE_Term",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = ""

    return df[final_cols]


# ============= 4. 整个 pipeline：只输出最终 step3 CSV =============
def run_pipeline(
    note_csv_path: str,
    baseline_file: str | None,
    ctcae_dict_csv: str,
    medcpt_model_dir: str,
    final_output_csv: str,
):
    # Step 1: GPT 抽 AE
    ae_df = gpt_extract_ae(note_csv_path)

    # Step 2: baseline 过滤（可选）
    ae_filtered = filter_with_baseline(ae_df, baseline_file)

    # Step 3: MedCPT 映射
    final_df = map_to_ctcae_medcpt(ae_filtered, ctcae_dict_csv, medcpt_model_dir)

    # 👉 只有这里写出一个最终文件
    final_df.to_csv(final_output_csv, index=False)
    print(f"\n🎉 全部完成！最终结果 CSV：{final_output_csv}")

    HISTORY_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/ae_history"

    merged_df = update_patient_history(
        ae_new_df=final_df,
        history_dir=HISTORY_DIR,
        mrn_col="MRN",
    )

    # 可选：只用于 debug / 前端临时查看（不是必须）
    latest_path = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/latest_merged.csv"
    merged_df.to_csv(latest_path, index=False)
    print(f"✅ latest merged saved -> {latest_path}")

    # ✅ 最关键：return merged_df（这样前端/调用方可以直接拿到“最新 AE list”）
    return merged_df

# ============= 5. 直接跑脚本用 =============
if __name__ == "__main__":

    # 1. notes CSV（你现在用的 progress_note，那一版有 Document Text / mrn / Document Date / Document Name）
    NOTES_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/reversed_Clindoc266614-1_05_progress_note.csv"

    # 2. baseline 文件（有就写路径，没有就写 "" 或 None）
    BASELINE_XLSX = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/18C0056_BL_Subgroup_02.xlsx"   

    # 3. CTCAE 词表 CSV
    CTCAE_DICT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/CTCAE_v5.0.csv"

    # 4. 你 finetune 好的 MedCPT 模型目录
    MEDCPT_MODEL_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/medcpt_ctcae_triplet_epoch10"

    # 5. 最终输出 CSV（只有这一个）
    FINAL_OUTPUT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/pipeline_ae_with_ctcae_note44.csv"

    # ====== 开始跑 ======
    run_pipeline(
        note_csv_path=NOTES_CSV,
        baseline_file=BASELINE_XLSX,
        ctcae_dict_csv=CTCAE_DICT_CSV,
        medcpt_model_dir=MEDCPT_MODEL_DIR,
        final_output_csv=FINAL_OUTPUT_CSV,
    )

