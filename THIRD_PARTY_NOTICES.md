# Attribution and license scope

## Original project contributions

Copyright 2026 Siddhesh Jadhav.

Original code and documentation contributed to this repository are licensed under
the Apache License, Version 2.0. See [LICENSE](LICENSE). This grant covers the
author's contributions; it does not assert ownership of upstream work or replace
third-party license and attribution requirements.

## LexGLUE and the historical archive

The historical experiments adapted ideas and code from
[coastalcph/lex-glue](https://github.com/coastalcph/lex-glue), Hugging Face
Transformers and PEFT. They are retained as a research archive, not represented
as wholly original library implementations. Existing upstream rights and notices
remain applicable to adapted portions.

The new package follows LexGLUE's ten article outputs and derived no-violation
evaluation convention. It does not vendor the original LexGLUE library.
Notebook provenance and publication transformations are recorded in
[the historical catalogue](docs/experiment_catalogue.md) and
[the manifest](notebooks/historical/manifest.json).

Benchmark reference: Ilias Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset
for Legal Language Understanding in English*, ACL, pages 4310–4330.
[Paper](https://aclanthology.org/2022.acl-long.297/).

The bounded pretrained pilot references
[`EleutherAI/gpt-neo-125m`](https://huggingface.co/EleutherAI/gpt-neo-125m)
at the immutable revision stored in its configuration. The repository does not
bundle those weights; the model card identifies the upstream model license as MIT.

## Materials outside this license grant

- **ECtHR-A and other datasets:** not bundled as case records in this repository.
  Obtain them from their providers and follow the applicable dataset terms.
- **Pretrained model weights and tokenizers:** obtained separately from model
  providers; each model's own license and access conditions apply.
- **Software dependencies:** remain licensed by their respective authors.
  Installation does not transfer their ownership to this project.
- **Third-party figures, papers and quotations:** remain subject to their
  respective source terms; citing a work does not relicense it.

Reference papers, other students' dissertations and personal records are not
included in the publication tree. The repository license does not grant rights
to privately held source documents or to data/model artifacts downloaded later.
