# stage-4 Response Parsing

Turning raw model text into commands + completion flags (JSON/XML parsers).


## L3 `Terminus2._get_parser` — `terminus_2.py:391-400`
- signature: `def _get_parser(self):`
- assignment provenance: rule
- reads state: `_parser_name`
- callers: `Terminus2.__init__`
- calls: `TerminusJSONPlainParser.__init__`, `TerminusXMLPlainParser.__init__`

## L3 `TerminusJSONPlainParser.__init__` — `terminus_json_plain_parser.py:26-27`
- signature: `def __init__(self):`
- assignment provenance: rule
- writes state: `required_fields`
- callers: `Terminus2._get_parser`

## L3 `TerminusJSONPlainParser.parse_response` — `terminus_json_plain_parser.py:29-62`
- signature: `def parse_response(self, response: str) -> ParseResult:`
- assignment provenance: rule
- calls: `TerminusJSONPlainParser._combine_warnings`, `TerminusJSONPlainParser._get_auto_fixes`, `TerminusJSONPlainParser._try_parse_response`

## L3 `TerminusJSONPlainParser._try_parse_response` — `terminus_json_plain_parser.py:64-163`
- signature: `def _try_parse_response(self, response: str) -> ParseResult:`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser.parse_response`
- calls: `TerminusJSONPlainParser._extract_json_content`, `TerminusJSONPlainParser._parse_commands`, `TerminusJSONPlainParser._validate_json_structure`

## L3 `TerminusJSONPlainParser._extract_json_content` — `terminus_json_plain_parser.py:165-212`
- signature: `def _extract_json_content(self, response: str) -> tuple[str, list[str]]:`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser._try_parse_response`

## L3 `TerminusJSONPlainParser._validate_json_structure` — `terminus_json_plain_parser.py:214-249`
- signature: `def _validate_json_structure( self, data: dict[str, Any], json_content: str, warnings: list[str] ) -> str:`
- assignment provenance: rule
- reads state: `required_fields`
- callers: `TerminusJSONPlainParser._try_parse_response`
- calls: `TerminusJSONPlainParser._check_field_order`

## L3 `TerminusJSONPlainParser._parse_commands` — `terminus_json_plain_parser.py:251-303`
- signature: `def _parse_commands( self, commands_data: list[dict[str, Any]], warnings: list[str] ) -> tuple[list[ParsedCommand], str]:`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser._try_parse_response`

## L3 `TerminusJSONPlainParser._get_auto_fixes` — `terminus_json_plain_parser.py:305-313`
- signature: `def _get_auto_fixes(self):`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser.parse_response`

## L3 `TerminusJSONPlainParser._fix_incomplete_json` — `terminus_json_plain_parser.py:315-328`
- signature: `def _fix_incomplete_json(self, response: str, error: str) -> tuple[str, bool]:`
- assignment provenance: rule

## L3 `TerminusJSONPlainParser._fix_mixed_content` — `terminus_json_plain_parser.py:330-343`
- signature: `def _fix_mixed_content(self, response: str, error: str) -> tuple[str, bool]:`
- assignment provenance: rule

## L3 `TerminusJSONPlainParser._combine_warnings` — `terminus_json_plain_parser.py:345-350`
- signature: `def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser.parse_response`

## L3 `TerminusJSONPlainParser._check_field_order` — `terminus_json_plain_parser.py:352-393`
- signature: `def _check_field_order( self, data: dict[str, Any], response: str, warnings: list[str] ) -> None:`
- assignment provenance: rule
- callers: `TerminusJSONPlainParser._validate_json_structure`

## L3 `TerminusXMLPlainParser.__init__` — `terminus_xml_plain_parser.py:25-26`
- signature: `def __init__(self):`
- assignment provenance: rule
- writes state: `required_sections`
- callers: `Terminus2._get_parser`

## L3 `TerminusXMLPlainParser.parse_response` — `terminus_xml_plain_parser.py:28-60`
- signature: `def parse_response(self, response: str) -> ParseResult:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser.salvage_truncated_response`
- calls: `TerminusXMLPlainParser._combine_warnings`, `TerminusXMLPlainParser._get_auto_fixes`, `TerminusXMLPlainParser._try_parse_response`

## L3 `TerminusXMLPlainParser._try_parse_response` — `terminus_xml_plain_parser.py:62-169`
- signature: `def _try_parse_response(self, response: str) -> ParseResult:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser.parse_response`
- calls: `TerminusXMLPlainParser._check_extra_text`, `TerminusXMLPlainParser._check_task_complete`, `TerminusXMLPlainParser._extract_response_content`, `TerminusXMLPlainParser._extract_sections`, `TerminusXMLPlainParser._parse_xml_commands`

## L3 `TerminusXMLPlainParser._get_auto_fixes` — `terminus_xml_plain_parser.py:171-178`
- signature: `def _get_auto_fixes(self):`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser.parse_response`

## L3 `TerminusXMLPlainParser._combine_warnings` — `terminus_xml_plain_parser.py:180-185`
- signature: `def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser.parse_response`

## L3 `TerminusXMLPlainParser._fix_missing_response_tag` — `terminus_xml_plain_parser.py:187-194`
- signature: `def _fix_missing_response_tag(self, response: str, error: str) -> tuple[str, bool]:`
- assignment provenance: rule

## L3 `TerminusXMLPlainParser._check_extra_text` — `terminus_xml_plain_parser.py:196-223`
- signature: `def _check_extra_text(self, response: str, warnings: list[str]) -> None:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._try_parse_response`

## L3 `TerminusXMLPlainParser._extract_response_content` — `terminus_xml_plain_parser.py:225-236`
- signature: `def _extract_response_content(self, response: str) -> str:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._try_parse_response`

## L3 `TerminusXMLPlainParser._extract_sections` — `terminus_xml_plain_parser.py:238-318`
- signature: `def _extract_sections(self, content: str, warnings: list[str]) -> dict[str, Any]:`
- assignment provenance: rule
- reads state: `required_sections`
- callers: `TerminusXMLPlainParser._try_parse_response`
- calls: `TerminusXMLPlainParser._check_section_order`, `TerminusXMLPlainParser._find_top_level_tags`

## L3 `TerminusXMLPlainParser._parse_xml_commands` — `terminus_xml_plain_parser.py:320-391`
- signature: `def _parse_xml_commands( self, xml_content: str, warnings: list[str] ) -> tuple[list[ParsedCommand], str]:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._try_parse_response`
- calls: `TerminusXMLPlainParser._check_attribute_issues`

## L3 `TerminusXMLPlainParser._find_top_level_tags` — `terminus_xml_plain_parser.py:393-440`
- signature: `def _find_top_level_tags(self, content: str) -> list[str]:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._extract_sections`

## L3 `TerminusXMLPlainParser._check_section_order` — `terminus_xml_plain_parser.py:442-480`
- signature: `def _check_section_order(self, content: str, warnings: list[str]) -> None:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._extract_sections`

## L3 `TerminusXMLPlainParser._check_attribute_issues` — `terminus_xml_plain_parser.py:482-512`
- signature: `def _check_attribute_issues( self, attributes_str: str, command_num: int, warnings: list[str] ) -> None:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._parse_xml_commands`

## L3 `TerminusXMLPlainParser._check_task_complete` — `terminus_xml_plain_parser.py:514-526`
- signature: `def _check_task_complete(self, response_content: str) -> bool:`
- assignment provenance: rule
- callers: `TerminusXMLPlainParser._try_parse_response`

## L3 `TerminusXMLPlainParser.salvage_truncated_response` — `terminus_xml_plain_parser.py:528-580`
- signature: `def salvage_truncated_response( self, truncated_response: str ) -> tuple[str | None, bool]:`
- assignment provenance: rule
- calls: `TerminusXMLPlainParser.parse_response`
