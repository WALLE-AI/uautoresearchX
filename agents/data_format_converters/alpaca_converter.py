"""转换目标格式：Alpaca JSON（`{"instruction","input","output"}`列表）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.data_format_converters._llm_common import record_to_turns, turns_to_alpaca
from agents.data_format_converters.base_converter import BaseDataFormatConverter, apply_field_mapping
from agents.planning.schemas import FieldMapping


class AlpacaConverter(BaseDataFormatConverter):
    def convert(
        self,
        records: list[dict[str, Any]],
        mapping_rules: list[FieldMapping],
        dst_path: Path,
    ) -> Path:
        output: list[dict[str, Any]] = []
        for record in records:
            mapped = apply_field_mapping(record, mapping_rules)
            turns = record_to_turns(mapped)
            output.append(turns_to_alpaca(turns))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return dst_path
