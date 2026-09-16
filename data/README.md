# Data

Raw and prepared datasets stay out of Git. The public repository contains code
and synthetic examples, not copies of court records.

Place unmodified ECtHR-A splits in `data/raw/` as `train.parquet`,
`validation.parquet` and `test.parquet`, or equivalently named JSONL files.
Each row must contain `text` (a string or ordered list of factual paragraphs)
and `labels` (a list of integer IDs from 0 through 9). Do not pass the historical
article-description files whose labels may be strings.

Source: [LexGLUE dataset](https://huggingface.co/datasets/coastalcph/lex_glue).
Record the upstream revision or local acquisition details when obtaining data.
The preparation manifest records byte hashes of the actual files used.

Run from the repository root after installation:

```sh
python -m ecthr_benchmark.data --input-dir data/raw --output-dir data/processed
```

Default validation expects 9,000 training, 1,000 validation and 1,000 test cases.
Use `--allow-small` only for intentionally small development fixtures.
