# Exporting the Report to PDF (Suggested)

This project stores the report as Markdown:

* `paper/report.md`
* `paper/appendix.md`

## Option A: Pandoc (Recommended)

If you have Pandoc installed, you can export a combined PDF:

```bash
pandoc paper/report.md paper/appendix.md ^
  --toc --number-sections ^
  -V geometry:margin=1in ^
  -V fontsize=12pt ^
  -V linestretch=1.3 ^
  -o paper/TemporalGuardRAG_Report.pdf
```

Notes:
* Use `--pdf-engine=xelatex` if you have LaTeX installed and want better font handling.
* Mermaid diagrams may require a Pandoc plugin or pre-rendering to images, depending on your setup.

## Option B: Word / Google Docs

1. Open `paper/report.md` and `paper/appendix.md`.
2. Convert to DOCX:

```bash
pandoc paper/report.md paper/appendix.md -o paper/TemporalGuardRAG_Report.docx
```

Then open the DOCX in Word/Docs, adjust spacing, and export to PDF.

