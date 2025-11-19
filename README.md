# SMALLMEM

#Smallmem Evaluation Results
| Observation Category   | Metric / Finding                           | Default                                     | RAG                                 | Index                                                                                       | Hybrid                                 |
| ---------------------- | ------------------------------------------ | ------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------- |
| **Performance**        | Overall Average Final Judge Score          | 5.24 (Highest)                              | 4.24 (Lowest)                       | 4.67                                                                                        | 4.26                                   |
| **Performance**        | Frequency as Best Performer (out of 20 Qs) | 9 questions (Most Frequent)                 | 3 questions                         | 5 questions                                                                                 | 3 questions                            |
| **Performance**        | Specialized Strength                       | Consistently High Performer                 | Low to Moderate                     | Highly effective for specific context-dependent/factual queries (e.g., Q5, Q9, Q17, Q18)    | Low to Moderate                        |
| **Performance**        | Score Differential                         | Benchmark for comparison                    | Lowest average score                | Achieved significantly higher scores than default for key questions (up to 6 points higher) | Second lowest average score            |
| **Latency Efficiency** | General Efficiency Ranking                 | Most Efficient (Lowest Latency/Score ratio) | Generally Least Efficient           | Moderate Efficiency                                                                         | Low Efficiency                         |
| **Latency Efficiency** | Trade-off                                  | Minimal latency cost for high performance   | High latency cost relative to score | Performance gains come with a moderate increase in latency relative to default              | Low score with noticeable latency cost |

- Check [Analysis notebook link here](https://www.kaggle.com/code/ayushkhaire/smallmem-evaluation)

```sh
Smallmem Project Structure
├── app.py
├── config
│   └── logger_config.py
├── eval_assets
│   ├── ideal_answers.json
│   └── results.json
├── eval_results
│   ├── evaluation_prefixed.csv
│   ├── evaluation_prefixed.json
│   ├── evaluation_with_llm_judge.csv
│   └── evaluation_with_llm_judge.json
├── Evaluation
│   ├── engine.py
│   └── metric.py
├── Indexer
│   ├── conccontx.py
│   ├── __init__.py
│   └── tree.py
├── LLM
│   ├── conversation.py
│   ├── docker_model.py
│   ├── gemma_local.py
│   └── __init__.py
├── models.md
├── pre_processing
│   ├── __init__.py
│   ├── pdf.py
│   └── text_processor.py
├── RAG
│   ├── embeding.py
│   ├── __init__.py
│   ├── pre_processor.py
│   └── vector.py
├── README.md
├── requirements.txt
├── run_eval.py
├── scripts
│   ├── start.sh
│   └── stop.sh
├── smallmem-evaluation.ipynb
└── uploads
    └── chapter_1.pdf
```

