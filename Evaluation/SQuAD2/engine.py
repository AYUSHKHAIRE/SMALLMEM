import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from load import DataLoader
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.docker_model import dockerModel
from tqdm import tqdm
from datetime import datetime
import pandas as pd

class Engine:
    def __init__(self):
        self.embeder = Embedder()
        self.chunker = Chunker()
        self.dockermodel = dockerModel(
            model="ai/granite-4.0-h-tiny:7B",
            hostname="localhost",
            port=12434,
            stream=True,
            system_prompt="You are a helpful assistant."
        )
        self.data = DataLoader().load_processed()

    def ask_LLM(self, question, context):
        prompt = f"""
            You are a friendly and intelligent AI assistant.

            You are given:
            - A user question.
            - Optional context extracted from uploaded documents.

            ---

            ### Guidelines:
            1. Answer the following question if you can .
            2. Answer the following question if you dont have context in prompt but you know it .

            **Question:** {question}

            **Context (optional):**
            {context}
            ---

            **Answer:**
        """
        response = self.dockermodel.ask_query(prompt[:18000])
        if hasattr(response, "__iter__") and not isinstance(response, str):
            # It's a generator, collect the streamed text
            answer_text = "".join(chunk for chunk in response)
        else:
            answer_text = response or ""
        return answer_text

    def test_LLM_against_questions_contexts(self, questions_answer_contexts, filename):
        os.makedirs("results", exist_ok=True)
        answer_df = {
            "question": [],
            "answer_original": [],
            "answer_LLM": [],
            "time": [],
            "context": []
        }

        for qc_dict in tqdm(questions_answer_contexts, desc="Generating answers"):
            start_t = datetime.now()
            answer = self.ask_LLM(
                question=qc_dict['q'],
                context=qc_dict['c'],
            )
            end_t = datetime.now()
            seconds = (end_t - start_t).total_seconds()

            answer_df["question"].append(qc_dict['q'])
            answer_df["context"].append(qc_dict['c'])
            answer_df["answer_original"].append(qc_dict['a'])
            answer_df["answer_LLM"].append(answer)
            answer_df["time"].append(seconds)

        df = pd.DataFrame(answer_df)
        df.to_csv(f"results/{filename}.csv", index=False)
        return df
            
# Example
eg = Engine()
input = pd.read_csv('data/processed_contexts.csv')
to_input = []
for q , a , c in tqdm(
     zip(
        input["question"],
        input["answer"],
        input["merged_context"]
    ),
    desc="preparing input "
):
    to_input.append(
        {
            'q':q,
            'a':a,
            'c':''
        }
    )
eg.test_LLM_against_questions_contexts(
    questions_answer_contexts=to_input[:100],
    filename="LLM_default_test"
)