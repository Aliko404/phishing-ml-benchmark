import json
import os
import re

input_log = "/home/darlene/Documents/sample-10mb.log"  # Имя вашего исходного .log файла
output_jsonl = "/home/darlene/Documents/phishing-ml-benchmark/data/raw/sample_test_run.jsonl"

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

with open(input_log, "r", encoding="utf-8") as infile, \
     open(output_jsonl, "w", encoding="utf-8") as outfile:
    for line in infile:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Пробуем найти число задержки (мс) в строке лога, если есть
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ms|мс)', line_str)
        latency = float(match.group(1)) if match else 120.0

        log_record = {
            "phase": "benchmark",
            "status_code": 200,
            "latency_ms": latency,
            "message": line_str
        }
        outfile.write(json.dumps(log_record, ensure_ascii=False) + "\n")

print("Файл data/raw/sample_test_run.jsonl успешно создан.")
