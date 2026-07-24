# Implement the proposed helpers for corpus `{{corpus_name}}`

## Proposed signatures
Read: `{{sig_path}}`
## Predefined TEXT API
Read `MODULES_SIGNATURES_TEXT` in `{{semdb_dir}}/vadar/predefined_text.py`.

Write a Python module with ALL helper implementations (composing only the predefined TEXT
API / earlier helpers / Python standard library) to: `{{helpers_path}}`. Each is
`def name(text, ...): <body>` where `text` is a plain string. Import only the required
offline functions, for example:
`from vadar.predefined_text import normalize, contains_phrase, contains_any, contains_all, best_lexical_match, regex_extract, split_values`

No model client, HTTP/network import, endpoint/API-key argument, `semtext`, or semantic
judgement API may appear in the generated module.
