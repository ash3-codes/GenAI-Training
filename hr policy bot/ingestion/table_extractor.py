def convert_table_to_markdown(table):
    if not table:
        return None

    header = table[0]
    rows = table[1:]

    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"

    for row in rows:
        md += "| " + " | ".join([cell if cell else "" for cell in row]) + " |\n"

    return md