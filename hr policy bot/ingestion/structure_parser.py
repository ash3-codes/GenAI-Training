import re



class StructureParser:

    ALPHA_HEADING_PATTERN = re.compile(r"^[A-Z]\)")
    COLON_HEADING_PATTERN = re.compile(r".+:\s*$")
    ALL_CAPS_PATTERN = re.compile(r"^[A-Z\s]{5,}$")

    def parse_page(self, text):

        lines = text.split("\n")

        structured_blocks = []

        current_chapter = None
        current_heading = None
        current_subheading = None
        current_paragraph = []

        for line in lines:
            line = line.strip()

            if not line:
                # End of paragraph
                if current_paragraph:
                    structured_blocks.append({
                        "chapter": current_chapter,
                        "heading": current_heading,
                        "subheading": current_subheading,
                        "content": " ".join(current_paragraph)
                    })
                    current_paragraph = []
                continue

            # ALL CAPS = chapter
            if self.ALL_CAPS_PATTERN.match(line) and len(line.split()) <= 8:
                current_chapter = line
                current_heading = None
                current_subheading = None
                continue

            # Resignation
            if self.ALPHA_HEADING_PATTERN.match(line):
                if current_paragraph:
                    structured_blocks.append({
                        "chapter": current_chapter,
                        "heading": current_heading,
                        "subheading": current_subheading,
                        "content": " ".join(current_paragraph)
                    })
                    current_paragraph = []

                current_heading = line
                current_subheading = None
                continue

            # Subheading
            if self.COLON_HEADING_PATTERN.match(line):
                current_subheading = line
                continue

            # Otherwise accumulate paragraph
            current_paragraph.append(line)

        # Flush last paragraph
        if current_paragraph:
            structured_blocks.append({
                "chapter": current_chapter,
                "heading": current_heading,
                "subheading": current_subheading,
                "content": " ".join(current_paragraph)
            })

        return structured_blocks