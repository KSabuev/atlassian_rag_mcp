"""Tests for attachment text extraction (confluence/attachments/) and AttachmentsMode."""

from __future__ import annotations

import base64
import io
import zlib
from unittest.mock import AsyncMock, MagicMock

import pytest

import atlassian_rag_mcp.confluence.page_content.attachments_mode as attachments_mode
from atlassian_rag_mcp.confluence.attachments import (
    bpmn,
    classify,
    doc,
    drawio,
    eml,
    extract_text,
    html,
    pdf,
    pptx,
    rtf,
    svg,
    text,
    xls,
    xlsx,
)
from atlassian_rag_mcp.confluence.attachments import (
    docx as docx_mod,
)
from atlassian_rag_mcp.confluence.attachments.errors import (
    AttachmentError,
    ConversionError,
    UnsupportedFormat,
)
from atlassian_rag_mcp.confluence.models import PageContent
from atlassian_rag_mcp.confluence.page_content import execute_page_content


def _make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello docx")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "a"
    table.cell(0, 1).text = "b"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["x", "y"])
    ws.append([1, 2])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_pptx() -> bytes:
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Slide title"
    slide.placeholders[1].text = "Body text"
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_pdf(body: str = "Hello PDF") -> bytes:
    content = f"BT /F1 24 Tf 100 700 Td ({body}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return bytes(out)


DRAWIO_PLAIN = (
    b'<mxfile host="x"><diagram name="s">'
    b"<mxGraphModel><root>"
    b'<mxCell id="0"/><mxCell id="1" parent="0"/>'
    b'<mxCell id="2" value="Node A" vertex="1" parent="1"/>'
    b'<mxCell id="3" value="Node B" vertex="1" parent="1"/>'
    b'<mxCell id="4" value="edge label" edge="1" source="2" target="3" parent="1"/>'
    b"</root></mxGraphModel></diagram></mxfile>"
)

BPMN_SAMPLE = (
    b'<?xml version="1.0"?>'
    b'<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" id="d">'
    b'<process id="p" name="Process1">'
    b'<startEvent id="se" name="Start"/>'
    b'<task id="t1" name="Do work"/>'
    b'<textAnnotation id="ta"><text>Note text</text></textAnnotation>'
    b"</process></definitions>"
)

SVG_SAMPLE = (
    b'<svg xmlns="http://www.w3.org/2000/svg">'
    b"<title>Chart</title>"
    b'<text x="10">Hello <tspan>SVG</tspan></text>'
    b"</svg>"
)





@pytest.mark.parametrize(
    "name,media,expected",
    [
        ("movie.mp4", None, "skipped:video"),
        ("movie.MP4", None, "skipped:video"),
        ("clip.webm", "video/webm", "skipped:video"),
        ("song.mp3", None, "skipped:audio"),
        ("voice.m4a", "audio/mp4", "skipped:audio"),
        ("pic.png", "image/png", "skipped:image"),
        ("pic.jpg", None, "skipped:image"),
        ("stuff.zip", None, "skipped:archive"),
        ("stuff.rar", None, "skipped:archive"),
        ("inst.msi", None, "skipped:binary"),
        ("cert.cer", None, "skipped:binary"),
        ("doc.docx", None, None),
        ("doc.doc", None, None),
        ("sheet.xlsx", None, None),
        ("sheet.xls", None, None),
        ("show.pptx", None, None),
        ("file.pdf", None, None),
        ("note.txt", None, None),
        ("diagram.drawio", None, None),
        ("diagram.bpmn", None, None),
        ("vector.svg", "image/svg+xml", None),
        ("noext", None, None),
        ("noext", "application/octet-stream", None),
        ("weird.unknownext", None, None),
    ],
)
def test_classify(name, media, expected):
    assert classify(name, media) == expected





def test_text_utf8_and_cp1251():
    assert text.convert("привет мир".encode("utf-8")) == "привет мир"
    assert text.convert("привет".encode("cp1251")) == "привет"
    assert text.convert("abc".encode("latin-1")) == "abc"


def test_docx():
    result = docx_mod.convert(_make_docx())
    assert "Hello docx" in result
    assert "a | b" in result


def test_xlsx():
    result = xlsx.convert(_make_xlsx())
    assert "Sheet: Data" in result
    assert "x | y" in result
    assert "1 | 2" in result


def test_xls():
    import xlwt

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("S1")
    sheet.write(0, 0, "c1")
    sheet.write(0, 1, "c2")
    sheet.write(1, 0, "v1")
    sheet.write(1, 1, "v2")
    buf = io.BytesIO()
    workbook.save(buf)
    result = xls.convert(buf.getvalue())
    assert "Sheet: S1" in result
    assert "c1 | c2" in result
    assert "v1 | v2" in result


def test_pptx():
    result = pptx.convert(_make_pptx())
    assert "Slide 1" in result
    assert "Slide title" in result
    assert "Body text" in result


def test_pdf():
    result = pdf.convert(_make_pdf())
    assert "page 1" in result
    assert "Hello PDF" in result


def test_rtf_ascii():
    result = rtf.convert(b"{\\rtf1\\ansi Hello RTF}")
    assert "Hello RTF" in result


def test_rtf_raw_cp1251():
    body = "Комментарий к закону".encode("cp1251")
    data = b"{\\rtf1\\ansicpg1251{" + body + b"}"
    result = rtf.convert(data)
    assert "Комментарий к закону" in result


def test_html():
    result = html.convert(b"<html><body><script>bad()</script><p>Para</p></body></html>")
    assert "Para" in result
    assert "bad()" not in result


def test_eml():
    eml_bytes = b"Subject: Test subj\r\nFrom: a@b.c\r\nTo: d@e.f\r\n\r\nPlain body"
    result = eml.convert(eml_bytes)
    assert "Subject: Test subj" in result
    assert "From: a@b.c" in result
    assert "Plain body" in result


def test_drawio_plain():
    result = drawio.convert(DRAWIO_PLAIN)
    assert "Node A" in result
    assert "Node B" in result
    assert "edge label" in result
    assert "2 nodes" in result


def test_drawio_compressed():
    inner = (
        b"<mxGraphModel><root>"
        b'<mxCell id="0"/><mxCell id="1" parent="0"/>'
        b'<mxCell id="2" value="Comp A" vertex="1" parent="1"/>'
        b"</root></mxGraphModel>"
    )
    b64 = base64.b64encode(zlib.compress(inner, 9)[2:]).decode()
    data = f'<mxfile host="x"><diagram name="s">{b64}</diagram></mxfile>'.encode()
    result = drawio.convert(data)
    assert "Comp A" in result


def test_drawio_compressed_percent_encoded():
    import urllib.parse

    inner = (
        b"<mxGraphModel dx=\"2044\" dy=\"786\" grid=\"1\"><root>"
        b'<mxCell id="0"/><mxCell id="1" parent="0"/>'
        b'<mxCell id="2" value="PercentNode" vertex="1" parent="1"/>'
        b"</root></mxGraphModel>"
    )
    percent = urllib.parse.quote(inner.decode()).encode()
    b64 = base64.b64encode(zlib.compress(percent, 9)[2:]).decode()
    data = f'<mxfile host="x"><diagram name="s">{b64}</diagram></mxfile>'.encode()
    result = drawio.convert(data)
    assert "PercentNode" in result


def test_drawio_invalid():
    with pytest.raises(ConversionError):
        drawio.convert(b"not xml at all")


def test_bpmn():
    result = bpmn.convert(BPMN_SAMPLE)
    assert "Process1" in result
    assert "Do work" in result
    assert "Note text" in result


def test_svg():
    result = svg.convert(SVG_SAMPLE)
    assert "Chart" in result
    assert "Hello SVG" in result





def test_xls_is_actually_xlsx():
    xlsx_bytes = _make_xlsx()
    result = xls.convert(xlsx_bytes)
    assert "Sheet: Data" in result


def test_xlsx_is_actually_ole_xls():
    import xlwt

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("L")
    sheet.write(0, 0, "legacy")
    buf = io.BytesIO()
    workbook.save(buf)
    result = xlsx.convert(buf.getvalue())
    assert "Sheet: L" in result
    assert "legacy" in result


def test_doc_is_actually_rtf():
    body = "Привет из RTF".encode("cp1251")
    data = b"{\\rtf1\\ansicpg1251{" + body + b"}"
    result = doc.convert(data)
    assert "Привет из RTF" in result


def test_docx_fallback_on_broken_table():
    import zipfile

    document_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body>"
        b'<w:p><w:r><w:t>Hello fallback</w:t></w:r></w:p>'
        b"<w:tbl><w:tr>"
        b'<w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc>'
        b'<w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc>'
        b"</w:tr></w:tbl>"
        b"</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    result = docx_mod.convert(buf.getvalue())
    assert "Hello fallback" in result
    assert "a | b" in result


def test_svg_accepts_filename():
    result = svg.convert(SVG_SAMPLE, "chart.svg")
    assert "Chart" in result


def test_xlsx_raw_xml_fallback():
    import zipfile

    workbook = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        b'<sheets><sheet name="S1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type=".../worksheet" Target="worksheets/sheet1.xml"/>'
        b"</Relationships>"
    )
    sst = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2">'
        b"<si><t>hello</t></si><si><t>world</t></si></sst>"
    )
    sheet = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData>"
        b'<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        b'<row r="2"><c r="A2" t="inlineStr"><is><t>inline</t></is></c></row>'
        b"</sheetData></worksheet>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/sharedStrings.xml", sst)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
    result = xlsx._via_raw_xml(buf.getvalue())
    assert "Sheet: S1" in result
    assert "hello | world" in result
    assert "inline" in result


def test_extract_by_extension():
    assert "Hello docx" in extract_text("a.docx", _make_docx())
    assert "Hello PDF" in extract_text("a.pdf", _make_pdf())
    assert "привет" in extract_text("a.txt", "привет".encode("cp1251"))
    assert "Node A" in extract_text("a.drawio", DRAWIO_PLAIN)


def test_extract_no_extension_magic():
    assert "Node A" in extract_text("noext", DRAWIO_PLAIN)
    assert "Hello PDF" in extract_text("noext", _make_pdf())
    assert "Hello docx" in extract_text("noext", _make_docx())
    assert "text" in extract_text("noext", b"plain text content")


def test_extract_unknown_binary_raises():
    with pytest.raises(UnsupportedFormat):
        extract_text("blob.bin", b"\x00\x01\x02\x03" * 100)


def test_extract_bad_docx_raises_attachment_error():
    with pytest.raises(AttachmentError):
        extract_text("broken.docx", b"not a real docx")





def _svc(attachments: list[dict]) -> MagicMock:
    s = MagicMock()
    s.search_deps.confluence_url = "https://c.example.com/"
    s.confluence_client.get_page = AsyncMock(return_value=PageContent(
        page_id="123", page_title="P", version=1,
    ))
    s.confluence_client.list_attachments = AsyncMock(return_value=attachments)
    return s


ATTACHMENTS = [
    {"title": "report.txt", "media_type": "text/plain", "size": 10,
     "version_number": 1, "created": "2024-01-01",
     "download_url": "https://c.example.com/dl/report.txt"},
    {"title": "movie.mp4", "media_type": "video/mp4", "size": 100,
     "version_number": 1, "created": "2024-01-01",
     "download_url": "https://c.example.com/dl/movie.mp4"},
    {"title": "BIG.PDF", "media_type": "application/pdf", "size": 99 * 1024 * 1024,
     "version_number": 1, "created": "2024-01-01",
     "download_url": "https://c.example.com/dl/big.pdf"},
]


async def test_attachments_mode_ok_skip_too_large(monkeypatch):
    dl = AsyncMock(return_value=b"hello text")
    monkeypatch.setattr(attachments_mode, "download_attachment", dl)
    svc = _svc(ATTACHMENTS)
    result = await execute_page_content("attachments", "123", svc)
    assert result.error is None
    by_title = {f.title: f for f in result.attachment_files}
    assert by_title["report.txt"].status == "ok"
    assert by_title["report.txt"].text.startswith("Файл: report.txt")
    assert "hello text" in by_title["report.txt"].text
    assert by_title["movie.mp4"].status == "skipped:video"
    assert by_title["movie.mp4"].text is None
    assert by_title["BIG.PDF"].status == "skipped:too_large"
    dl.assert_awaited_once_with("https://c.example.com/dl/report.txt", 20 * 1024 * 1024)


async def test_attachments_mode_titles_filter_and_not_found(monkeypatch):
    dl = AsyncMock(return_value=b"hello text")
    monkeypatch.setattr(attachments_mode, "download_attachment", dl)
    svc = _svc(ATTACHMENTS)
    result = await execute_page_content(
        "attachments", "123", svc, titles=["REPORT.txt", "missing.docx"],
    )
    by_title = {f.title: f for f in result.attachment_files}
    assert by_title["report.txt"].status == "ok"
    assert by_title["missing.docx"].status == "not_found"
    assert "movie.mp4" not in by_title
    dl.assert_awaited_once()


async def test_attachments_mode_truncation(monkeypatch):
    dl = AsyncMock(return_value=b"x" * 25000)
    monkeypatch.setattr(attachments_mode, "download_attachment", dl)
    svc = _svc(ATTACHMENTS)
    result = await execute_page_content("attachments", "123", svc, titles=["report.txt"])
    f = result.attachment_files[0]
    assert f.status == "ok"
    assert f.truncated is True
    assert len(f.text) == 20000


async def test_attachments_mode_conversion_error(monkeypatch):
    dl = AsyncMock(return_value=b"garbage")
    monkeypatch.setattr(attachments_mode, "download_attachment", dl)
    svc = _svc([
        {"title": "broken.docx", "media_type": "application/vnd.openxmlformats",
         "size": 7, "version_number": 1, "created": "2024-01-01",
         "download_url": "https://c.example.com/dl/broken.docx"},
    ])
    result = await execute_page_content("attachments", "123", svc)
    assert result.attachment_files[0].status.startswith("error:")
    assert result.attachment_files[0].text is None


async def test_attachments_mode_no_download_url(monkeypatch):
    dl = AsyncMock(return_value=b"hello")
    monkeypatch.setattr(attachments_mode, "download_attachment", dl)
    svc = _svc([
        {"title": "no_url.txt", "media_type": "text/plain", "size": 10,
         "version_number": 1, "created": "2024-01-01", "download_url": None},
    ])
    result = await execute_page_content("attachments", "123", svc)
    assert result.attachment_files[0].status == "error: no download url"


async def test_files_mode_titles_filter():
    svc = _svc(ATTACHMENTS)
    result = await execute_page_content(
        "files", "123", svc, titles=["MOVIE.mp4"],
    )
    assert [f.title for f in result.files] == ["movie.mp4"]
    result_all = await execute_page_content("files", "123", svc)
    assert len(result_all.files) == 3





def test_registry_all_converters_registered():
    from atlassian_rag_mcp.confluence.attachments import registry

    names = [c.name for c in registry.all_converters()]
    expected = {
        "text", "pdf", "rtf", "docx", "xlsx", "xls", "pptx",
        "html", "eml", "doc", "drawio", "bpmn", "svg",
    }
    assert set(names) == expected
    assert len(names) == len(set(names)), "converter names must be unique"


def test_registry_duplicate_name_raises():
    from atlassian_rag_mcp.confluence.attachments import registry

    @registry.converter(name="zzdup", exts=("zzdup1",))
    def convert1(data: bytes, filename: str = "") -> str:
        return "1"

    try:
        with pytest.raises(ValueError, match="duplicate converter name"):
            @registry.converter(name="zzdup", exts=("zzdup2",))
            def convert2(data: bytes, filename: str = "") -> str:
                return "2"
    finally:
        registry._REGISTRY[:] = [c for c in registry._REGISTRY if c.name != "zzdup"]


def test_registry_by_ext():
    from atlassian_rag_mcp.confluence.attachments import registry

    assert registry.by_ext("pdf").name == "pdf"
    assert registry.by_ext("txt").name == "text"
    assert registry.by_ext("html").name == "html"
    assert registry.by_ext("docx").name == "docx"
    assert registry.by_ext("zzz") is None


def test_registry_with_magic():
    from atlassian_rag_mcp.confluence.attachments import registry

    magic_names = {c.name for c in registry.with_magic()}
    assert magic_names == {"pdf", "rtf", "docx", "xlsx", "pptx", "drawio", "bpmn", "svg"}


def test_add_converter_registers_and_dispatches():
    """The 'drop a file' contract: @converter self-registers and dispatches."""
    from atlassian_rag_mcp.confluence.attachments import registry

    @registry.converter(name="zztest", exts=("zztest",))
    def convert(data: bytes, filename: str = "") -> str:
        return "zztest:" + data.decode()

    try:
        assert registry.by_ext("zztest").name == "zztest"
        assert extract_text("a.zztest", b"hi") == "zztest:hi"
        assert extract_text("a.unknownext", b"plain") == "plain"
    finally:
        registry._REGISTRY[:] = [c for c in registry._REGISTRY if c.name != "zztest"]


def test_dispatch_content_magic_without_ext():
    """Files with no/foreign extension are routed by content signature."""
    assert "rtf body" in extract_text("blob.bin", b"{\\rtf1 rtf body}")
    out = extract_text("blob.bin", b'<mxGraphModel><mxCell value="N1"/></mxGraphModel>')
    assert "N1" in out
    assert extract_text("blob.bin", b'<?xml version="1.0"?><foo>bar</foo>') == (
        '<?xml version="1.0"?><foo>bar</foo>'
    )
