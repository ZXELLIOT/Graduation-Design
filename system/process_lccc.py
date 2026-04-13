import os
import json
import random
import zipfile

# 默认参数
DEFAULT_JSON_FILENAME = "LCCC-base_train.json"
DEFAULT_OUTPUT_TXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "raw_dialogues.txt"
)
DEFAULT_SAMPLE_SIZE = 100000
# 默认在项目根目录查找 LCCC 压缩包
DEFAULT_WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZIP_CANDIDATES = [
    r"C:\Users\13713\个人信息\毕业设计\LCCC-base-split.zip",
    r"C:\Users\13713\个人信息\毕业设计\LCCC-large.zip",
    os.path.join(DEFAULT_WORK_DIR, "LCCC-base-split.zip"),
    os.path.join(DEFAULT_WORK_DIR, "LCCC-large.zip"),
]

JSON_FILENAME = DEFAULT_JSON_FILENAME
OUTPUT_TXT_PATH = DEFAULT_OUTPUT_TXT_PATH
SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE  # 提取 10 万条作为训练及测试语料


def _resolve_zip_path():
    for candidate in DEFAULT_ZIP_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    # 兼容把压缩包放在脚本同级目录的情况
    local_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        for name in ("LCCC-base-split.zip", "LCCC-large.zip")
    ]
    for candidate in local_candidates:
        if os.path.exists(candidate):
            return candidate

    return DEFAULT_ZIP_CANDIDATES[0]


LCCC_ZIP_PATH = _resolve_zip_path()


def _resolve_json_filename(zip_path: str) -> str:
    """自动选择压缩包中的 train JSON 文件。"""
    preferred_names = [
        "LCCC-base_train.json",
        "LCCC-large_train.json",
    ]
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

    for name in preferred_names:
        if name in names:
            return name

    # 回退：自动匹配以 train 结尾的 json 文件
    for name in names:
        lower_name = name.lower()
        if lower_name.endswith(".json") and "train" in lower_name:
            return name

    # 兼容旧默认值
    return JSON_FILENAME

def extract_lccc():
    if not os.path.exists(LCCC_ZIP_PATH):
        print(f"找不到 LCCC 数据集压缩包：{LCCC_ZIP_PATH}")
        return

    json_filename = _resolve_json_filename(LCCC_ZIP_PATH)
    print(f"使用压缩包: {LCCC_ZIP_PATH}")
    print(f"使用 JSON 文件: {json_filename}")

    print("开始从压缩包中加载 LCCC JSON 数据...")
    try:
        # 直接读取 ZIP 压缩包中的大 JSON 文件以节省硬盘空间
        with zipfile.ZipFile(LCCC_ZIP_PATH, 'r') as z:
            with z.open(json_filename) as f:
                data = json.load(f)
    except Exception as e:
        print(f"JSON 解析失败：{e}")
        return

    # LCCC 格式解析：data 里面通常是一个 list，每项是一个单轮或多轮对话的句子 list
    # 例如: [["你在干嘛", "在看电视", "看什么电视"], ["吃饭了吗", "刚吃完"]]
    print(f"成功加载，共有 {len(data)} 个对话 session。正在提取相邻问答对...")
    
    qa_pairs = []
    for dialog_session in data:
        # 只取每个 session 的前两句话作为 Q（提问） 和 A（回答）
        if isinstance(dialog_session, list) and len(dialog_session) >= 2:
            q = dialog_session[0].strip()
            a = dialog_session[1].strip()
            
            # 初步清洗：排除过长或过短的垃圾对话
            if 2 <= len(q) <= 40 and 2 <= len(a) <= 40:
                # 确保里面没有制表符（\t）和换行符
                q = q.replace('\t', ' ').replace('\n', ' ')
                a = a.replace('\t', ' ').replace('\n', ' ')
                qa_pairs.append(f"{q}\t{a}\n")

    print(f"共提取出 {len(qa_pairs)} 个初步符合长度要求的问答对。")
    
    # 抽取 SAMPLE_SIZE 条
    print(f"正在随机抽取 {SAMPLE_SIZE} 条作为系统语料库...")
    if len(qa_pairs) > SAMPLE_SIZE:
        sampled_pairs = random.sample(qa_pairs, SAMPLE_SIZE)
    else:
        sampled_pairs = qa_pairs
    
    # 写入最终 txt 文件
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(sampled_pairs)
        
    print(f"\n已成功提取 {len(sampled_pairs)} 条 LCCC 对话并写入 raw_dialogues.txt")

if __name__ == "__main__":
    extract_lccc()