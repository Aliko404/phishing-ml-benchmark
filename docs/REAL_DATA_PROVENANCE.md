# Real data provenance and evidence boundary

Input: UCI Phishing Websites (dataset ID 327), 11,055 labeled real website records and 30 numerical attributes. DOI https://doi.org/10.24432/C51W2X. This is not a pre-existing network log.

`python -m src.download_uci` requests the public dataset and creates a local CSV with target labels mapped consistently: -1 -> 1 (phishing); +1 -> 0 (legitimate). The script verifies 11,055 rows and 30 features and writes a SHA256 manifest.

`python -m src.collect_inference` trains two models and records actual timing of `predict()` on held-out UCI instances. Calls with `phase=warmup` are discarded by the analysis. Writes JSONL with dataset hash, timestamp, software version, batch size and local success status.

Do not say JSONL is captured production traffic; it is instrumented experimental telemetry. As of package creation, UCI network could not be accessed from the authoring runtime, so no legitimate real benchmark results can be bundled. Run on Linux to produce authentic evidence and then commit final figures, notebook outputs and a small **real** raw log sample if allowed by lecturer.
