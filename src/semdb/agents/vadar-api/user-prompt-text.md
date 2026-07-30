# Implement the proposed helpers for corpus `{{corpus_name}}`

## Proposed signatures
Read: `{{sig_path}}`
## Predefined TEXT API
Read the TEXT PRIMITIVES section of `MODULES_SIGNATURES` in
`{{semdb_dir}}/vadar/predefined.py` (the same file also documents the VISION functions;
a text helper must use only the TEXT ones).

Write a Python module with ALL helper implementations (composing only the predefined TEXT
API / earlier helpers / Python standard library) to: `{{helpers_path}}`. Each is
`def name(text, ...): <body>` where `text` is a plain string. Import only the required
offline functions, for example:
`from vadar.predefined import normalize, contains_phrase, contains_any, contains_all, best_lexical_match, regex_extract, split_values`

No model client, HTTP/network import, endpoint/API-key argument, `semvqa`/`semcaption`/`semextract`, or semantic
judgement API may appear in the generated module.
