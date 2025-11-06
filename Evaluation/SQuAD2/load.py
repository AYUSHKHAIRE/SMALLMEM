import os
import json
import pandas as pd
import math

class DataLoader:
    def __init__(self, raw_path="data/train-v2.0.json", processed_path="data/processed_contexts.csv"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.data = None

    def load_processed(self):
        """Main entry point – load processed data or build it if missing"""
        if os.path.exists(self.processed_path):
            print(f"✅ Found processed file: {self.processed_path}")
            return pd.read_csv(self.processed_path)
        else:
            print("⚙️ Processed file not found — building from raw JSON...")
            df = self._process_raw()
            df.to_csv(self.processed_path, index=False)
            print(f"✅ Saved processed data to {self.processed_path}")
            return df

    def _process_raw(self):
        """Load JSON and create merged context dataframe"""
        with open(self.raw_path, "r") as f:
            data = json.load(f)

        contexts, questions, answers = [], [], []

        # Extract SQuAD fields
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    if qa.get("answers"):
                        for ans in qa["answers"]:
                            contexts.append(context)
                            questions.append(question)
                            answers.append(ans["text"])
                    else:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(None)

        df = pd.DataFrame({
            "context": contexts,
            "question": questions,
            "answer": answers
        })

        # Merge contexts into ~1000 combined ones
        unique_contexts = df["context"].drop_duplicates().tolist()
        num_contexts = len(unique_contexts)
        target = 1000
        merge_size = math.ceil(num_contexts / target)

        merged_contexts = [
            " ".join(unique_contexts[i:i + merge_size])
            for i in range(0, num_contexts, merge_size)
        ]

        # Map old context → merged context
        context_to_merged = {}
        for i, merged in enumerate(merged_contexts):
            for c in unique_contexts[i * merge_size:(i + 1) * merge_size]:
                context_to_merged[c] = merged

        df["merged_context"] = df["context"].map(context_to_merged)
        # remove context column
        df.drop(columns=["context"], inplace=True)
        print(f"✅ Processed {len(df)} rows into {len(merged_contexts)} merged contexts.")
        return df
