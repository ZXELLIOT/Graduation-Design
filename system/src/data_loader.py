import pandas as pd
from tqdm.auto import tqdm


class DataLoader:
    @staticmethod
    def load_corpus(csv_path, n_samples=None, random_state=42):
        """
        从 CSV 文件读取语料，并返回问句与回答列表。

        参数说明：
        - csv_path: CSV 文件路径。
        - n_samples: 如指定，则从语料中随机抽取该数量样本。
        - random_state: 随机种子，保证抽样可重复。

        返回值：
        - tuple (queries, responses)
        """
        print(f"正在读取语料文件：{csv_path} ...")
        df = pd.read_csv(csv_path)

        if n_samples is not None and len(df) > n_samples:
            print(f"将随机抽取 {n_samples} 条样本以加快处理。")
            df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

        # 逐行构建问答列表，并显示进度
        queries = []
        responses = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="构建语料列表", unit="rows"):
            queries.append(str(row.get('query', '')))
            responses.append(str(row.get('response', '')))

        print(f"已构建 {len(queries)} 条问答样本。")
        return queries, responses