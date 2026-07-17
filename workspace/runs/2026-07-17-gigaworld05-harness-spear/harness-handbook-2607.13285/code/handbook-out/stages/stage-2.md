# stage-2 Prompt & Template Assembly

Loading prompt/timeout templates, skill frontmatter, instruction construction.


## L3 `Terminus2._get_prompt_template_path` — `terminus_2.py:402-411`
- signature: `def _get_prompt_template_path(self) -> Path:`
- assignment provenance: rule
- reads state: `_parser_name`
- callers: `Terminus2.__init__`

## L3 `Terminus2._get_timeout_template_path` — `terminus_2.py:413-415`
- signature: `def _get_timeout_template_path(self) -> Path:`
- assignment provenance: rule
- callers: `Terminus2.__init__`

## L3 `Terminus2._parse_skill_frontmatter` — `terminus_2.py:418-433`
- signature: `def _parse_skill_frontmatter(content: str) -> dict[str, str] | None:`
- decorators: `staticmethod`
- assignment provenance: rule-ambiguous(stage-2,stage-4)
- callers: `Terminus2._build_skills_section`
