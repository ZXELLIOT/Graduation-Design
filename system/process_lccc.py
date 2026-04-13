import os
import json
import time

from tqdm.auto import tqdm

# simcse-demo 目录与工作区根目录。
SIMCSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(SIMCSE_DIR)
LCCC_BASE_DIR = os.path.join(WORKSPACE_DIR, "LCCC-base-split")
TRAIN_JSON_PATH = os.path.join(LCCC_BASE_DIR, "LCCC-base_train.json")
TEST_JSON_PATH = os.path.join(LCCC_BASE_DIR, "LCCC-base_test.json")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_TRAIN_TXT_PATH = os.path.join(OUTPUT_DIR, "raw_dialogues_train.txt")
OUTPUT_TEST_TXT_PATH = os.path.join(OUTPUT_DIR, "raw_dialogues_test.txt")


def _extract_and_write(json_path: str, output_path: str, stage_name: str) -> int:
    """从 JSON 全量提取问答对并直接写入 txt。"""
    load_start = time.perf_counter()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    load_elapsed = time.perf_counter() - load_start
    print(f"[{stage_name}] JSON 读取完成，session 数: {len(data)}，耗时: {load_elapsed:.2f} 秒")

    # 提前建目录，避免长时间处理后才因为目录问题失败。
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    extract_start = time.perf_counter()
    pair_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # 给 session 层加进度条，处理进度和速度会实时显示。
        for dialog_session in tqdm(
            data,
            total=len(data),
            desc=f"{stage_name} 抽取进度",
            unit="session",
            dynamic_ncols=True,
        ):
            if not isinstance(dialog_session, list) or len(dialog_session) < 2:
                continue

            # 全量读取：一个 session 内所有相邻轮次都转成 Q/A。
            for idx in range(len(dialog_session) - 1):
                q = str(dialog_session[idx]).strip()
                a = str(dialog_session[idx + 1]).strip()
                if not q or not a:
                    continue

                # 统一替换制表符和换行，避免影响后续按列读取。
                q = q.replace("\t", " ").replace("\n", " ")
                a = a.replace("\t", " ").replace("\n", " ")
                f.write(f"{q}\t{a}\n")
                pair_count += 1

    extract_elapsed = time.perf_counter() - extract_start
    print(f"[{stage_name}] 问答对输出完成，共 {pair_count} 条，耗时: {extract_elapsed:.2f} 秒")
    return pair_count


def extract_lccc():
    """读取固定的两个数据集并输出 train/test 两个 txt。"""
    total_start = time.perf_counter()

    if not os.path.exists(TRAIN_JSON_PATH):
        print(f"找不到训练集文件：{TRAIN_JSON_PATH}")
        return
    if not os.path.exists(TEST_JSON_PATH):
        print(f"找不到测试集文件：{TEST_JSON_PATH}")
        return

    print(f"训练集 JSON: {TRAIN_JSON_PATH}")
    print(f"测试集 JSON: {TEST_JSON_PATH}")

    print("开始处理训练集（全量）...")
    train_count = _extract_and_write(
        json_path=TRAIN_JSON_PATH,
        output_path=OUTPUT_TRAIN_TXT_PATH,
        stage_name="训练集",
    )

    print("开始处理测试集（全量）...")
    test_count = _extract_and_write(
        json_path=TEST_JSON_PATH,
        output_path=OUTPUT_TEST_TXT_PATH,
        stage_name="测试集",
    )

    total_elapsed = time.perf_counter() - total_start

    print(f"\n已输出训练语料: {OUTPUT_TRAIN_TXT_PATH}")
    print(f"已输出测试语料: {OUTPUT_TEST_TXT_PATH}")
    print(f"训练集问答对数量: {train_count}")
    print(f"测试集问答对数量: {test_count}")
    print(f"全部流程总耗时: {total_elapsed:.2f} 秒")

if __name__ == "__main__":
    extract_lccc()