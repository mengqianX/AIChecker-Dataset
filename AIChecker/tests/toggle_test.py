    

import json
from pathlib import Path
from aichecker.checkers import check_toggle
from aichecker.utils import _encode


def main():
    file="./sample_payload.json"
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    debug_dir = Path("./debug/crops")
    result = check_toggle(data, debug_dir=debug_dir)
    print(json.dumps(result, default=_encode, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
