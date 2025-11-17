from Evaluation.engine import Engine
import pandas as pd

eg = Engine()

df = pd.read_csv("Evaluation/SQuAD2/data/processed_contexts.csv")

# Run for first 10 rows
df = eg.evaluate(df[:3], output_file="mini_eval.csv")
df.to_csv("result.csv")
print(df.columns)