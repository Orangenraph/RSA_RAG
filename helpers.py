import json
import re
from datetime import datetime

def format_time(seconds):
    """Format seconds into time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        sec = seconds % 60
        return f"{int(minutes)} minutes {int(sec)} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        sec = seconds % 60
        return f"{int(hours)} hours {int(minutes)} minutes {int(sec)} seconds"



def extract_rules(response_text):
    """Extrahiert Regeln im Format 'if <condition> then <do>' aus dem Text, wobei nur der Inhalt NACH dem </think>-Tag berücksichtigt wird.
    """
    # Finde den Inhalt nach dem </think>-Tag
    start_index = response_text.find('</think>')
    if start_index == -1:
        relevant_part = response_text  # Falls kein Tag vorhanden
    else:
        relevant_part = response_text[start_index + len('</think>'):]

    # Regex für Regeln mit if-then
    pattern = r'(if\s+.*?\s+then\s+.*?)(?=\n\s*if\s|\n\s*\n|$)'

    # Finde alle Übereinstimmungen im relevanten Teil
    matches = re.findall(pattern, relevant_part, re.IGNORECASE | re.DOTALL)

    # Bereinige die Regeln (entferne überflüssige Leerzeichen)
    rules = [rule.strip() for rule in matches]

    return rules


# Beispiel für die Verwendung in deiner save_response_to_json Funktion
def save_response_to_json(query_text, response_text, sources, similarity_score=None):
    """
    Speichert die RAG-Antwort in einer JSON-Datei.

    Args:
        query_text (str): Die ursprüngliche Abfrage
        response_text (str): Die vollständige Antwort des LLM
        sources (list): Liste der Quellen-IDs
        similarity_score (float, optional): Der beste Ähnlichkeitswert
        filename (str, optional): Dateiname für die JSON-Ausgabe, Standard ist "law.json"

    Returns:
        str: Pfad zur gespeicherten JSON-Datei
    """
    # Extrahiere nummerierte Regeln
    rules = extract_rules(response_text)

    # Wandle in Dict um für bessere JSON-Struktur
    rules_dict = {str(i + 1): rule for i, rule in enumerate(rules)}

    # Create JSON structure
    output_data = {
        "query": query_text,
        "timestamp": datetime.now().isoformat(),
        "similarity_score": similarity_score,
        "sources": sources,
        "rules_numbered": rules_dict,
    }

    # Save to JSON file (overwriting previous file)
    filename = f"{query_text}_laws.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResponse saved to {filename}")

    return filename