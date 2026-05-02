# Data Directory

This directory holds your FAQ source files and category metadata.

## Structure

```
data/
├── faq_categories.yaml   # category metadata (maps folder name → group/label/description)
└── faq/                  # your raw FAQ text files (optional — or keep in FAQ_data/)
    ├── url_txt_tra_chi/   # Traditional Chinese
    │   ├── domain/
    │   │   ├── register/   *.txt
    │   │   └── transfer/   *.txt
    │   └── ...
    ├── url_txt_sim_chi/   # Simplified Chinese
    └── url_txt_en/        # English
```

## FAQ Text File Format

Each `.txt` file maps to one document chunk in the vector index.

Optional anchor support (for deep-linking to specific sections):

```
ANCHOR:section_id
Your FAQ content here...
```

Or embed the anchor in the filename:  `how-to-renew__anchor_renewal.txt`

## Adding a New Language / Knowledge Base

1. Create a folder under `FAQ_data/` (e.g. `url_txt_jp/`)
2. Add category entries to `faq_categories.yaml`
3. Add a new entry under `knowledge_bases` in `config/config.yaml`
4. Re-run `python scripts/build_index.py`

## Rebuilding Indices

```bash
python scripts/build_index.py
# or for a specific language only:
python scripts/build_index.py --kb traditional_chinese
```
