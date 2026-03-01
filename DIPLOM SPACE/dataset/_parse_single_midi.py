#!/usr/bin/env python3
"""
Вспомогательный скрипт для парсинга одного MIDI файла.
Используется из preprocess_midi.py с timeout через subprocess.
"""
import sys
import json
import warnings
import music21 as m21

warnings.filterwarnings("ignore")
def parse_midi_file(file_path):
    """Parses a MIDI file and returns the number of parts."""
    try:
        score = m21.converter.parse(file_path, forceSource=True)
        parts_count = len(score.parts)
        return {"success": True, "parts_count": parts_count}
    except Exception as e:
        error_message = str(e)
        return {"success": False, "error": error_message[:100]}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No filepath provided"}))
        sys.exit(1)
    
    filepath = sys.argv[1]
    result = parse_midi_file(filepath)
    print(json.dumps(result))
    sys.exit(0 if result["success"] else 1)
