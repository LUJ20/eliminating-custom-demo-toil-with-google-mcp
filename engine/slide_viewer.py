"""Interactive Google Slides / PPTX Presentation Player for Streamlit UI.
Parses native PowerPoint shapes, text, tables, and presenter notes directly from the .pptx
file and renders an interactive presentation player with slide navigation, notes, and
direct links to Google Slides.
"""
import os
import json
import html as _html
import base64 as _b64
from typing import Optional

try:
    import pptx
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    pptx = None


def render_presentation_player(deck_path: str, gslides_url: Optional[str] = None, height: int = 520) -> str:
    """Builds an interactive Google Slides HTML player string from a .pptx file."""
    if not os.path.exists(deck_path) or pptx is None:
        return "<p style='color:#5F6368;'>Presentation deck preview not available.</p>"

    try:
        prs = pptx.Presentation(deck_path)
    except Exception as e:
        return f"<p style='color:#EA4335;'>Failed to load presentation: {_html.escape(str(e))}</p>"

    sw = float(prs.slide_width or 12192000)
    sh = float(prs.slide_height or 6858000)
    slides_html = []
    notes_list = []
    total_slides = len(prs.slides)

    for s_idx, slide in enumerate(prs.slides):
        shape_divs = []
        slide_title = f"Slide {s_idx + 1}"
        first_title_found = False

        for shp in slide.shapes:
            l_pct = round(max(0.0, float(getattr(shp, "left", 0) or 0) / sw * 100.0), 2)
            t_pct = round(max(0.0, float(getattr(shp, "top", 0) or 0) / sh * 100.0), 2)
            w_pct = round(max(0.5, float(getattr(shp, "width", 0) or 0) / sw * 100.0), 2)
            h_pct = round(max(0.5, float(getattr(shp, "height", 0) or 0) / sh * 100.0), 2)

            # Table shapes
            if getattr(shp, "has_table", False):
                try:
                    tbl = shp.table
                    tr_list = []
                    for r_i, row in enumerate(tbl.rows):
                        td_list = []
                        for c_i, cell in enumerate(row.cells):
                            c_txt = (cell.text or "").strip()
                            c_bg = "#1A73E8" if r_i == 0 else ("#F8F9FA" if r_i % 2 == 1 else "#FFFFFF")
                            c_fg = "#FFFFFF" if r_i == 0 else "#202124"
                            c_fw = "700" if (r_i == 0 or c_i == 0) else "400"
                            try:
                                if cell.fill and cell.fill.type == 1 and cell.fill.fore_color and cell.fill.fore_color.rgb:
                                    c_bg = f"#{cell.fill.fore_color.rgb}"
                                    if c_bg.upper() in ("#1A73E8", "#174EA6", "#202124", "#34A853", "#EA4335"):
                                        c_fg = "#FFFFFF"
                            except Exception:
                                pass
                            td_list.append(
                                f'<td style="background:{c_bg};color:{c_fg};font-weight:{c_fw};'
                                f'padding:4px 8px;border:1px solid #DADCE0;font-size:0.95cqw;line-height:1.25;vertical-align:top;">'
                                f'{_html.escape(c_txt)}</td>'
                            )
                        tr_list.append(f"<tr>{''.join(td_list)}</tr>")
                    shape_divs.append(
                        f'<div class="gs-shape" style="position:absolute;left:{l_pct}%;top:{t_pct}%;width:{w_pct}%;height:{h_pct}%;overflow:hidden;z-index:2;">'
                        f'<table style="width:100%;height:100%;border-collapse:collapse;table-layout:fixed;">{"".join(tr_list)}</table>'
                        f'</div>'
                    )
                    continue
                except Exception:
                    pass

            bg_hex = "transparent"
            try:
                if shp.fill and shp.fill.type == 1 and shp.fill.fore_color and shp.fill.fore_color.rgb:
                    bg_hex = f"#{shp.fill.fore_color.rgb}"
            except Exception:
                pass

            border_css = "none"
            try:
                if shp.line and shp.line.color and shp.line.color.rgb:
                    border_css = f"1px solid #{shp.line.color.rgb}"
            except Exception:
                pass

            radius_css = "5px"
            try:
                if getattr(shp, "auto_shape_type", None) == MSO_SHAPE.OVAL:
                    radius_css = "50%"
                elif getattr(shp, "auto_shape_type", None) == MSO_SHAPE.RIGHT_ARROW:
                    # Render as styled arrow symbol if arrow
                    shape_divs.append(
                        f'<div style="position:absolute;left:{l_pct}%;top:{t_pct}%;width:{w_pct}%;height:{h_pct}%;'
                        f'display:flex;align-items:center;justify-content:center;color:#1A73E8;font-size:1.6cqw;font-weight:bold;z-index:2;">'
                        f'➜</div>'
                    )
                    continue
            except Exception:
                pass

            paras_html = []
            if getattr(shp, "has_text_frame", False):
                for p_obj in shp.text_frame.paragraphs:
                    raw_t = (p_obj.text or "").strip()
                    if not raw_t:
                        continue
                    if not first_title_found and len(raw_t) > 3:
                        slide_title = raw_t[:45]
                        first_title_found = True

                    pt_val = 10.0
                    is_bold = False
                    col_hex = "#202124"
                    try:
                        if p_obj.font and p_obj.font.size:
                            pt_val = float(p_obj.font.size.pt)
                        elif p_obj.runs and p_obj.runs[0].font.size:
                            pt_val = float(p_obj.runs[0].font.size.pt)
                    except Exception:
                        pass
                    try:
                        if p_obj.font and p_obj.font.bold:
                            is_bold = True
                        elif p_obj.runs and p_obj.runs[0].font.bold:
                            is_bold = True
                    except Exception:
                        pass
                    try:
                        if p_obj.font and p_obj.font.color and p_obj.font.color.rgb:
                            col_hex = f"#{p_obj.font.color.rgb}"
                        elif p_obj.runs and p_obj.runs[0].font.color and p_obj.runs[0].font.color.rgb:
                            col_hex = f"#{p_obj.runs[0].font.color.rgb}"
                    except Exception:
                        pass

                    cqw_size = round(max(0.85, min(2.6, (pt_val / 960.0) * 110.0)), 2)
                    fw_val = "700" if is_bold else "400"
                    paras_html.append(
                        f'<div style="font-size:{cqw_size}cqw;font-weight:{fw_val};color:{col_hex};line-height:1.28;margin-bottom:0.25cqw;word-break:break-word;">'
                        f'{_html.escape(raw_t)}</div>'
                    )

            pad_css = "0.5cqw 0.75cqw" if paras_html else "0"
            z_idx = "2" if paras_html else "1"
            shape_divs.append(
                f'<div class="gs-shape" style="position:absolute;left:{l_pct}%;top:{t_pct}%;width:{w_pct}%;height:{h_pct}%;'
                f'background:{bg_hex};border:{border_css};border-radius:{radius_css};padding:{pad_css};'
                f'box-sizing:border-box;overflow:hidden;z-index:{z_idx};">'
                f'{"".join(paras_html)}</div>'
            )

        notes_txt = ""
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_txt = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:
            pass
        notes_list.append(notes_txt if notes_txt else f"Slide {s_idx + 1}: {slide_title}")

        disp_style = "block" if s_idx == 0 else "none"
        slide_inner_html = "".join(shape_divs)
        slides_html.append(
            f'<div class="gslide-frame" id="gslide_{s_idx}" style="display:{disp_style};container-type:inline-size;position:relative;width:97%;height:410px;background:#FFFFFF;overflow:hidden;border-radius:4px;box-shadow:0 3px 12px rgba(0,0,0,0.25);">'
            f'{slide_inner_html}'
            f'</div>'
        )

    options_html = "".join(f'<option value="{i}">Slide {i+1} of {total_slides}</option>' for i in range(total_slides))
    # Notes come from model-written text: keep them inert inside the <script> block.
    notes_json = json.dumps(notes_list).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    slides_link = ""
    if (gslides_url or "").startswith("https://docs.google.com/"):  # only a deck that was really published
        slides_link = (f'<a href="{_html.escape(gslides_url, quote=True)}" target="_blank" rel="noopener noreferrer" '
                       'style="text-decoration:none;color:#1A73E8;font-size:13px;font-weight:600;display:flex;'
                       'align-items:center;gap:6px;background:#FFF;border:1px solid #DADCE0;padding:5px 14px;'
                       'border-radius:6px;"><span style="display:inline-block;width:12px;height:12px;'
                       'background:#FBBC04;border-radius:2px;"></span><span>Open in Google Slides &#8599;</span></a>')

    player_html = f"""
    <div style="font-family:'Google Sans',Roboto,sans-serif;width:100%;border:1px solid #DADCE0;border-radius:8px;overflow:hidden;background:#1F1F1F;box-shadow:0 2px 8px rgba(0,0,0,0.12);">
      <div style="position:relative;background:#121212;height:435px;display:flex;align-items:center;justify-content:center;padding:10px 8px;">
        <div style="position:absolute;top:12px;right:16px;z-index:20;">
          <button onclick="toggleNotes()" style="background:rgba(32,33,36,0.9);border:1px solid #747775;color:#FFF;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;display:flex;align-items:center;gap:6px;">
            <span>📄</span> <span>Speaker Notes</span>
          </button>
        </div>
        <div id="gs_container" style="display:flex;align-items:center;justify-content:center;width:100%;height:100%;">
          {"".join(slides_html)}
        </div>
        <div id="gs_notes" style="display:none;position:absolute;bottom:12px;left:18px;right:18px;background:rgba(32,33,36,0.96);color:#E8EAED;padding:12px 16px;border-radius:6px;font-size:12px;line-height:1.45;z-index:25;border-left:4px solid #FBBC04;max-height:140px;overflow-y:auto;white-space:pre-line;">
          {_html.escape(notes_list[0] if notes_list else "")}
        </div>
      </div>
      <div style="background:#F1F3F4;height:48px;padding:0 16px;display:flex;align-items:center;justify-content:space-between;border-top:1px solid #DADCE0;">
        <div style="display:flex;align-items:center;gap:10px;">
          <button onclick="stepSlide(-1)" style="background:#FFF;border:1px solid #DADCE0;padding:4px 12px;border-radius:999px;cursor:pointer;font-weight:bold;font-size:13px;" title="Previous slide">&#10094;</button>
          <select id="gs_select" onchange="showSlide(parseInt(this.value))" style="background:#FFF;border:1px solid #DADCE0;padding:4px 12px;border-radius:999px;cursor:pointer;font-size:13px;font-weight:600;color:#202124;">
            {options_html}
          </select>
          <button onclick="stepSlide(1)" style="background:#FFF;border:1px solid #DADCE0;padding:4px 12px;border-radius:999px;cursor:pointer;font-weight:bold;font-size:13px;" title="Next slide">&#10095;</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          {slides_link}
        </div>
      </div>
    </div>
    <script>
      var curSlide = 0;
      var totalSlides = {total_slides};
      var slideNotes = {notes_json};
      var notesVisible = false;

      function showSlide(idx) {{
        if (idx < 0) idx = totalSlides - 1;
        if (idx >= totalSlides) idx = 0;
        curSlide = idx;
        for (var i = 0; i < totalSlides; i++) {{
          var el = document.getElementById('gslide_' + i);
          if (el) el.style.display = (i === curSlide) ? 'block' : 'none';
        }}
        var sel = document.getElementById('gs_select');
        if (sel) sel.value = String(curSlide);
        var nb = document.getElementById('gs_notes');
        if (nb) nb.innerText = slideNotes[curSlide] || '';
      }}

      function stepSlide(delta) {{
        showSlide(curSlide + delta);
      }}

      function toggleNotes() {{
        notesVisible = !notesVisible;
        var nb = document.getElementById('gs_notes');
        if (nb) nb.style.display = notesVisible ? 'block' : 'none';
      }}
    </script>
    """
    return player_html
