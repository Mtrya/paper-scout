# Data

Place the two WikiText-103 raw parquet files here:

```bash
curl -L -o data/wt103_validation.parquet \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet"
curl -L -o data/wt103_test.parquet \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet"
```

The files are not committed; they are 0.6-0.7 MB each and regenerate deterministically from the URLs.
