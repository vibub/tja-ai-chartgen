from pydantic import BaseModel

REQUIRED_HEADERS = ["TITLE:", "BPM:", "WAVE:", "OFFSET:", "COURSE:", "LEVEL:"]
ALLOWED_CHART_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz, \t")


class ValidationIssue(BaseModel):
    level: str
    message: str
    line: int | None = None


def validate_tja_text(text: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    lines = text.splitlines()

    for header in REQUIRED_HEADERS:
        if not any(line.startswith(header) for line in lines):
            issues.append(ValidationIssue(level="error", message=f"Missing required header: {header}"))

    start_lines = [index for index, line in enumerate(lines, start=1) if line.strip() == "#START"]
    end_lines = [index for index, line in enumerate(lines, start=1) if line.strip() == "#END"]

    if not start_lines:
        issues.append(ValidationIssue(level="error", message="Missing #START"))
    if not end_lines:
        issues.append(ValidationIssue(level="error", message="Missing #END"))

    if start_lines and end_lines:
        start_line = start_lines[0]
        end_line = end_lines[0]
        if start_line >= end_line:
            issues.append(
                ValidationIssue(
                    level="error",
                    message="#START must appear before #END",
                    line=start_line,
                )
            )
        else:
            chart_lines = lines[start_line:end_line - 1]
            for line_number, line in enumerate(chart_lines, start=start_line + 1):
                stripped = line.strip()
                if not stripped:
                    continue
                invalid_chars = sorted(set(stripped) - ALLOWED_CHART_CHARS)
                if invalid_chars:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            message=f"Invalid chart characters: {''.join(invalid_chars)}",
                            line=line_number,
                        )
                    )
                if not stripped.endswith(","):
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            message="Chart bar line should end with comma",
                            line=line_number,
                        )
                    )

    return issues
