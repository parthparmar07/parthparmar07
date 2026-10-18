"""Builds the animated profile SVGs in assets/. Run: python scripts/build_assets.py
Needs network once (Google Fonts + Simple Icons); output SVGs are fully self-contained."""
import base64, io, re, urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
PORTRAIT = ROOT / "scripts" / "parth.png"  # transparent B&W portrait from the portfolio

# Portfolio tokens (Portfolio-cool/src/index.css, dark theme) + emerald "open to work" accent
BG, BG2, BORDER, FG, FG2, ACC = "#050505", "#0A0A0A", "#27272a", "#ffffff", "#b4b4bd", "#10b981"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()


def font_b64(family, weight):
    css = get(f"https://fonts.googleapis.com/css2?family={family}:wght@{weight}&display=swap").decode()
    # last @font-face block is the latin subset
    url = re.findall(r"url\((https://[^)]+\.woff2)\)", css)[-1]
    return base64.b64encode(get(url)).decode()


FONTS = {
    "Outfit": [(900, font_b64("Outfit", 900)), (600, font_b64("Outfit", 600))],
    "JetBrains Mono": [(500, font_b64("JetBrains+Mono", 500))],
}


def font_css(prefix):
    # namespaced family names so multiple SVGs never collide
    out = []
    for fam, ws in FONTS.items():
        for w, b in ws:
            out.append(f"@font-face{{font-family:'{prefix}-{fam}';font-weight:{w};"
                       f"src:url(data:font/woff2;base64,{b}) format('woff2');}}")
    return "".join(out)


def portrait(size):
    im = Image.open(PORTRAIT).convert("LA").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


SI_VER = "13.21.0"


def icon(slug):
    svg = get(f"https://cdn.jsdelivr.net/npm/simple-icons@{SI_VER}/icons/{slug}.svg").decode()
    return re.search(r' d="([^"]+)"', svg).group(1)


def svg(name, w, h, css, body, prefix):
    base = f"""
  .{prefix} .d{{font-family:'{prefix}-Outfit',sans-serif;font-weight:900}}
  .{prefix} .s{{font-family:'{prefix}-Outfit',sans-serif;font-weight:600}}
  .{prefix} .m{{font-family:'{prefix}-JetBrains Mono',monospace;font-weight:500}}
  @media (prefers-reduced-motion: reduce){{.{prefix} *{{animation:none!important}}}}"""
    doc = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}" class="{prefix}" role="img" aria-label="{name}">
<style>{font_css(prefix)}{base}{css}</style>
<defs>
  <pattern id="{prefix}-dots" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="{FG}" opacity=".07"/></pattern>
  <linearGradient id="{prefix}-hair" x1="0" x2="1"><stop offset="0" stop-color="{FG}" stop-opacity="0"/><stop offset=".5" stop-color="{FG}" stop-opacity=".35"/><stop offset="1" stop-color="{FG}" stop-opacity="0"/></linearGradient>
  <radialGradient id="{prefix}-glow" cx=".5" cy="0" r=".9"><stop offset="0" stop-color="{FG}" stop-opacity=".07"/><stop offset="1" stop-color="{FG}" stop-opacity="0"/></radialGradient>
</defs>
<rect width="{w}" height="{h}" rx="24" fill="{BG}"/>
<rect width="{w}" height="{h}" rx="24" fill="url(#{prefix}-dots)"/>
<rect width="{w}" height="{h}" rx="24" fill="url(#{prefix}-glow)"/>
<rect x=".5" y=".5" width="{w-1}" height="{h-1}" rx="23.5" fill="none" stroke="{BORDER}"/>
<rect x="80" y="0" width="{w-160}" height="1" fill="url(#{prefix}-hair)"/>
{body}
</svg>"""
    (OUT / name).write_text(doc, encoding="utf-8")
    print(f"{name}: {len(doc)//1024} KB")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- hero
def hero():
    p = "hr"
    roles = ["AI Systems Engineer", "LLM Platform Developer", "Voice AI Builder", "Agentic RAG Architect"]
    role_css = "".join(f".{p} .r{i}{{animation:{p}role 12s {i*3}s infinite both}}" for i in range(4))
    css = f"""
  .{p} .rise{{animation:{p}rise 1s cubic-bezier(.2,.8,.2,1) both}}
  .{p} .fade{{animation:{p}fade .9s ease-out both}}
  .{p} .pic{{animation:{p}pic 1.4s ease-out both}}
  .{p} .r1,.{p} .r2,.{p} .r3{{opacity:0}}
  {role_css}
  .{p} .blink{{animation:{p}blink 1s steps(1) infinite}}
  .{p} .pulse{{animation:{p}pulse 2s ease-in-out infinite;transform-origin:center;transform-box:fill-box}}
  @keyframes {p}rise{{from{{transform:translateY(120px)}}}}
  @keyframes {p}fade{{from{{opacity:0;transform:translateY(12px)}}}}
  @keyframes {p}pic{{from{{opacity:0;transform:scale(1.06)}}}}
  @keyframes {p}role{{0%{{opacity:0;transform:translateY(14px)}}3%,22%{{opacity:1;transform:none}}25%,100%{{opacity:0;transform:translateY(-14px)}}}}
  @keyframes {p}blink{{50%{{opacity:0}}}}
  @keyframes {p}pulse{{50%{{transform:scale(2.2);opacity:0}}}}"""
    # first role is the static fallback, so it gets a non-fading variant when animation is off
    roles_svg = "".join(f'<text class="s r{i}" x="64" y="330" font-size="30" fill="{FG}">{esc(r)}</text>' for i, r in enumerate(roles))
    greet = "> hello, world. i'm"
    body = f"""
<image class="pic" href="{portrait(440)}" x="740" y="20" width="440" height="440"/>
<rect x="700" y="300" width="500" height="160" fill="url(#{p}-fadeb)"/>
<defs>
  <linearGradient id="{p}-fadeb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BG}" stop-opacity="0"/><stop offset="1" stop-color="{BG}"/></linearGradient>
  <clipPath id="{p}-type"><rect x="64" y="44" height="30" width="360"><animate attributeName="width" values="0;0;360" keyTimes="0;.25;1" dur="1.6s"/></rect></clipPath>
  <clipPath id="{p}-n1"><rect x="0" y="80" width="760" height="112"/></clipPath>
  <clipPath id="{p}-n2"><rect x="0" y="182" width="760" height="112"/></clipPath>
  <clipPath id="{p}-roles"><rect x="60" y="296" width="640" height="46"/></clipPath>
</defs>
<g clip-path="url(#{p}-type)"><text class="m" x="64" y="66" font-size="20" fill="{FG2}">{esc(greet)}</text></g>
<rect class="blink" x="300" y="48" width="11" height="22" fill="{ACC}"/>
<g clip-path="url(#{p}-n1)"><text class="d rise" x="58" y="182" font-size="124" letter-spacing="-5" fill="{FG}">PARTH</text></g>
<g clip-path="url(#{p}-n2)"><text class="d rise" style="animation-delay:.15s" x="58" y="284" font-size="124" letter-spacing="-5" fill="{FG}">PARMAR</text></g>
<g clip-path="url(#{p}-roles)">{roles_svg}</g>
<text class="s fade" style="animation-delay:.6s" x="64" y="372" font-size="18" fill="{FG2}">Production-grade LLM platforms, agentic AI &amp; real-time voice systems.</text>
<rect x="64" y="396" width="620" height="1" fill="{BORDER}"/>
<g class="fade m" style="animation-delay:.9s" font-size="14" fill="{FG2}">
  <circle class="pulse" cx="72" cy="423" r="5" fill="{ACC}"/><circle cx="72" cy="423" r="5" fill="{ACC}"/>
  <text x="86" y="428" fill="{ACC}" letter-spacing="2">OPEN TO WORK</text>
  <text x="230" y="428">India</text><text x="290" y="428" fill="{BORDER}">/</text>
  <text x="310" y="428">Tech Lead @ Vocacity</text><text x="492" y="428" fill="{BORDER}">/</text>
  <text x="512" y="428">parthparmarr.netlify.app</text>
</g>"""
    svg("hero.svg", 1200, 460, css, body, p)


# ---------------------------------------------------------------- about
def about():
    p = "ab"
    caps = [("01", "Voice AI pipelines", "ASR → LLM → memory → TTS, under 500ms"),
            ("02", "Agentic RAG systems", "Retrieval, tools &amp; multi-agent orchestration"),
            ("03", "LLM platforms", "FastAPI services, evals, routing, cost control"),
            ("04", "Full-stack product", "React / Next.js front ends that ship")]
    slides = [("NOW", "Mar 2025 → Present", "Vocacity", "Tech Team Lead · Voice AI",
               ["10,000+ automated calls / month", "&lt;500ms voice latency", "Team of 5 engineers"]),
              ("NOW", "Feb 2026 → Present", "DevHub", "Founder's Office · Growth",
               ["Community scaled to 1,600+ members", "Mumbai Tech Week – AI Festival 2026", "Buildverse technical lead"]),
              ("PREVIOUSLY", "Apr 2026 → Jul 2026", "Ventura Securities", "AI/ML Intern · Financial Systems",
               ["60–70% less manual analysis time", "Financial ML models on market data", "Automated reporting pipelines"])]
    cap_svg = "".join(f"""<g class="fade" style="animation-delay:{.15*i:.2f}s">
  <text class="m" x="64" y="{150+i*64}" font-size="13" fill="{ACC}">{n}</text>
  <text class="s" x="104" y="{150+i*64}" font-size="22" fill="{FG}">{t}</text>
  <text class="m" x="104" y="{174+i*64}" font-size="13" fill="{FG2}">{d}</text></g>""" for i, (n, t, d) in enumerate(caps))
    sl_svg, bars = "", ""
    for i, (tag, when, org, role, pts) in enumerate(slides):
        items = "".join(f'<text class="m" x="680" y="{258+j*30}" font-size="15" fill="{FG2}"><tspan fill="{ACC}">→ </tspan>{t}</text>' for j, t in enumerate(pts))
        sl_svg += f"""<g class="sl sl{i}">
  <text class="m" x="680" y="142" font-size="12" letter-spacing="3" fill="{ACC if tag == 'NOW' else FG2}">{tag} · {when}</text>
  <text class="d" x="676" y="196" font-size="48" letter-spacing="-1.5" fill="{FG}">{org}</text>
  <text class="s" x="680" y="226" font-size="18" fill="{FG2}">{role}</text>{items}</g>"""
        x = 680 + i * 160
        bars += f'<rect x="{x}" y="372" width="148" height="3" rx="1.5" fill="{BORDER}"/><rect class="bar b{i}" x="{x}" y="372" width="148" height="3" rx="1.5" fill="{FG}"/>'
    css = f"""
  .{p} .fade{{animation:{p}fade .8s ease-out both}}
  .{p} .sl1,.{p} .sl2{{opacity:0}}
  .{p} .sl{{animation:{p}slide 12s infinite both}}
  .{p} .sl1{{animation-delay:4s}} .{p} .sl2{{animation-delay:8s}}
  .{p} .bar{{transform-box:fill-box;transform-origin:left;animation:{p}bar 12s linear infinite both}}
  .{p} .b1{{animation-delay:4s;transform:scaleX(0)}} .{p} .b2{{animation-delay:8s;transform:scaleX(0)}}
  @keyframes {p}fade{{from{{opacity:0;transform:translateX(-12px)}}}}
  @keyframes {p}slide{{0%{{opacity:0;transform:translateY(10px)}}3%,30%{{opacity:1;transform:none}}33.3%,100%{{opacity:0;transform:translateY(-10px)}}}}
  @keyframes {p}bar{{0%{{transform:scaleX(0)}}33.3%{{transform:scaleX(1)}}33.4%,100%{{transform:scaleX(0)}}}}"""
    body = f"""
<text class="m" x="64" y="80" font-size="13" letter-spacing="3" fill="{FG2}">01 / WHAT I BUILD</text>
{cap_svg}
<rect x="624" y="40" width="536" height="360" rx="18" fill="{BG2}" stroke="{BORDER}"/>
<text class="m" x="680" y="88" font-size="13" letter-spacing="3" fill="{FG2}">02 / WHERE I'M BUILDING</text>
{sl_svg}{bars}"""
    svg("about-life.svg", 1200, 420, css, body, p)


# ---------------------------------------------------------------- stack
STACK = {
    "Languages": ["Python", "TypeScript", "JavaScript", "C++"],
    "AI / LLM": ["Claude", "Gemini", "Groq", "LangChain", "LangGraph", "Ollama", "Hugging Face", "PyTorch"],
    "Backend & Data": ["FastAPI", "Node.js", "PostgreSQL", "MongoDB", "Supabase", "Qdrant", "ChromaDB"],
    "Frontend": ["Next.js", "React", "Tailwind", "Three.js", "D3.js"],
    "Voice & Ops": ["LiveKit", "Deepgram", "Docker", "Git", "Vercel", "Cloudflare"],
}
ORBITS = [  # (rx, ry, tilt, seconds, simple-icons slugs)
    (110, 46, -18, 14, ["python", "typescript", "react"]),
    (200, 84, 12, 22, ["fastapi", "langchain", "anthropic", "nextdotjs", "postgresql"]),
    (262, 116, -8, 30, ["docker", "pytorch", "huggingface", "googlegemini", "ollama", "supabase", "git"]),
]


def stack():
    p = "st"
    cx, cy = 320, 310
    orb = ""
    for k, (rx, ry, tilt, dur, slugs) in enumerate(ORBITS):
        path = f"M{cx-rx},{cy} a{rx},{ry} 0 1,0 {2*rx},0 a{rx},{ry} 0 1,0 {-2*rx},0"
        orb += f'<g transform="rotate({tilt} {cx} {cy})"><path id="{p}-o{k}" d="{path}" fill="none" stroke="{BORDER}" stroke-dasharray="3 6"/>'
        for i, s in enumerate(slugs):
            d = icon(s)
            # counter-rotate so icons stay upright on a tilted orbit; begin offsets spread them out
            orb += f"""<g><animateMotion dur="{dur}s" repeatCount="indefinite" begin="-{dur*i/len(slugs):.2f}s"><mpath href="#{p}-o{k}"/></animateMotion>
  <g transform="rotate({-tilt})"><circle r="22" fill="{BG2}" stroke="{BORDER}"/><path transform="translate(-11 -11) scale(.9167)" d="{d}" fill="{FG}"/></g></g>"""
        orb += "</g>"
    chips, y = "", 104
    for group, items in STACK.items():
        chips += f'<text class="m" x="680" y="{y}" font-size="12" letter-spacing="3" fill="{FG2}">{esc(group.upper())}</text>'
        x, y = 680, y + 14
        for it in items:
            w = 18 + len(it) * 8.4
            if x + w > 1150:
                x, y = 680, y + 38
            chips += f'<rect x="{x}" y="{y}" width="{w:.0f}" height="28" rx="14" fill="{BG2}" stroke="{BORDER}"/><text class="m" x="{x+w/2:.0f}" y="{y+19}" font-size="13" text-anchor="middle" fill="{FG}">{it}</text>'
            x += w + 8
        y += 56
    css = f"""
  .{p} .core{{animation:{p}pulse 3s ease-in-out infinite;transform-box:fill-box;transform-origin:center}}
  @keyframes {p}pulse{{50%{{transform:scale(1.15);opacity:.6}}}}"""
    body = f"""
<text class="m" x="64" y="80" font-size="13" letter-spacing="3" fill="{FG2}">03 / STACK</text>
{orb}
<circle class="core" cx="{cx}" cy="{cy}" r="34" fill="{ACC}" opacity=".18"/>
<circle cx="{cx}" cy="{cy}" r="26" fill="{FG}"/><text class="d" x="{cx}" y="{cy+8}" font-size="22" text-anchor="middle" fill="{BG}">PP</text>
<line x1="640" y1="70" x2="640" y2="510" stroke="{BORDER}"/>
{chips}"""
    svg("stack.svg", 1200, 560, css, body, p)


# ---------------------------------------------------------------- id + dashboard
def id_dashboard(gh):
    p = "id"
    bars = "".join(f'<rect x="{126+i*7}" y="476" width="{(1, 3, 2, 1, 4, 2)[i % 6]}" height="34" fill="{FG}"/>' for i, _ in enumerate(range(14)))
    metrics = [("10,000+", "automated calls / month", "Vocacity · 2026"),
               ("&lt;500ms", "voice pipeline latency", "Vocacity · 2026"),
               ("3", "major production releases", "Vocacity · 2026"),
               ("60–70%", "manual analysis time cut", "Ventura · Jul 2026"),
               (str(gh["public_repos"]), "public repositories", f"GitHub · {gh['date']}"),
               ("10+", "awards &amp; selections", "2024 – 2026")]
    m_svg = ""
    for i, (v, l, src) in enumerate(metrics):
        x, y = 520 + (i % 2) * 320, 120 + (i // 2) * 130
        m_svg += f"""<g class="fade" style="animation-delay:{.8+i*.12:.2f}s">
  <rect x="{x}" y="{y}" width="296" height="110" rx="16" fill="{BG2}" stroke="{BORDER}"/>
  <text class="d" x="{x+24}" y="{y+52}" font-size="38" letter-spacing="-1" fill="{FG}">{v}</text>
  <text class="s" x="{x+24}" y="{y+78}" font-size="15" fill="{FG2}">{l}</text>
  <text class="m" x="{x+24}" y="{y+98}" font-size="11" fill="{ACC}">{src}</text></g>"""
    css = f"""
  .{p} .swing{{transform-origin:240px 0;animation:{p}drop 2.6s cubic-bezier(.3,.7,.3,1) both,{p}swing 5s 2.6s ease-in-out infinite}}
  .{p} .fade{{animation:{p}fade .7s ease-out both}}
  .{p} .foil{{animation:{p}foil 5s 1.5s ease-in-out infinite}}
  @keyframes {p}drop{{0%{{transform:translateY(-560px) rotate(0)}}35%{{transform:translateY(0) rotate(9deg)}}55%{{transform:rotate(-6deg)}}72%{{transform:rotate(3.5deg)}}86%{{transform:rotate(-2.4deg)}}100%{{transform:rotate(1.7deg)}}}}
  @keyframes {p}swing{{0%,100%{{transform:rotate(1.7deg)}}50%{{transform:rotate(-1.7deg)}}}}
  @keyframes {p}fade{{from{{opacity:0;transform:translateY(10px)}}}}
  @keyframes {p}foil{{0%{{transform:translateX(-300px)}}40%,100%{{transform:translateX(420px)}}}}"""
    body = f"""
<defs>
  <clipPath id="{p}-card"><rect x="100" y="150" width="280" height="384" rx="20"/></clipPath>
  <linearGradient id="{p}-sheen" x1="0" x2="1"><stop offset="0" stop-color="{FG}" stop-opacity="0"/><stop offset=".5" stop-color="{FG}" stop-opacity=".14"/><stop offset="1" stop-color="{FG}" stop-opacity="0"/></linearGradient>
  <linearGradient id="{p}-metal" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#e4e4e7"/><stop offset="1" stop-color="#71717a"/></linearGradient>
</defs>
<g class="swing">
  <path d="M212 0 L228 118 M268 0 L252 118" stroke="{FG}" stroke-width="16" opacity=".9"/>
  <path d="M212 0 L228 118 M268 0 L252 118" stroke="{BG}" stroke-width="16" stroke-dasharray="2 10" opacity=".5"/>
  <rect x="216" y="112" width="48" height="22" rx="6" fill="url(#{p}-metal)"/>
  <rect x="232" y="130" width="16" height="26" rx="4" fill="none" stroke="url(#{p}-metal)" stroke-width="4"/>
  <rect x="100" y="150" width="280" height="384" rx="20" fill="{BG2}" stroke="{BORDER}"/>
  <g clip-path="url(#{p}-card)">
    <rect x="100" y="150" width="280" height="384" fill="url(#{p}-dots)"/>
    <rect x="216" y="164" width="48" height="8" rx="4" fill="{BG}" stroke="{BORDER}"/>
    <image href="{portrait(220)}" x="130" y="182" width="220" height="220"/>
    <rect x="100" y="320" width="280" height="90" fill="url(#{p}-fadec)"/>
    <text class="d" x="126" y="420" font-size="29" letter-spacing="-1" fill="{FG}">PARTH PARMAR</text>
    <text class="m" x="126" y="444" font-size="12" fill="{FG2}">AI SYSTEMS ENGINEER</text>
    <circle cx="131" cy="461" r="4" fill="{ACC}"/><text class="m" x="142" y="465" font-size="11" fill="{ACC}">ACCESS: OPEN TO WORK</text>
    {bars}
    <text class="m" x="330" y="510" font-size="10" text-anchor="end" fill="{FG2}">@parthparmar07</text>
    <rect class="foil" x="100" y="150" width="140" height="384" fill="url(#{p}-sheen)" transform="skewX(-12)"/>
  </g>
  <defs><linearGradient id="{p}-fadec" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BG2}" stop-opacity="0"/><stop offset=".7" stop-color="{BG2}"/></linearGradient></defs>
</g>
<text class="m" x="520" y="80" font-size="13" letter-spacing="3" fill="{FG2}">04 / ID &amp; DASHBOARD</text>
{m_svg}"""
    svg("id-dashboard.svg", 1200, 560, css, body, p)


# ---------------------------------------------------------------- achievements
GOLD = "#fbbf24"  # portfolio amber-400 trophy accent
TROPHY = "M7 3h10v5a5 5 0 0 1-10 0zM7 5H3v1a4 4 0 0 0 4 4M17 5h4v1a4 4 0 0 1-4 4M12 13v4M8 21h8l-1-4H9z"


def achievements():
    p = "ac"
    stats = [("4×", "hackathon wins"), ("2", "startups → Pvt Ltd"),
             ("1,600+", "DevHub community"), ("300+", "gwava.dev users"), ("#8", "GSSoC global rank")]
    tiles = ""
    for i, (v, l) in enumerate(stats):
        x = 64 + i * 218
        tiles += f"""<g class="up" style="animation-delay:{.1+i*.1:.1f}s">
  <rect x="{x}" y="108" width="202" height="122" rx="16" fill="{BG2}" stroke="{BORDER}"/>
  <text class="d" x="{x+22}" y="170" font-size="46" letter-spacing="-1.5" fill="{FG}">{v}</text>
  <text class="m" x="{x+22}" y="204" font-size="12" fill="{FG2}">{esc(l)}</text></g>"""
    wall = [("Code 4 Compassion", "", GOLD, "Featured in Free Press Journal &amp; The Hans India"),
            ("HEAT Hackathon 1.0", "", GOLD, "Predictive food-decision agent with Swiggy MCP"),
            ("NIT Goa Blockchain", "", GOLD, "National blockchain hackathon"),
            ("BNB Chain Hackathon", "", GOLD, "Blockchain track")]
    rows = ""
    for i, (t, r, c, d) in enumerate(wall):
        y = 296 + i * 50
        rows += f"""<g class="side" style="animation-delay:{.6+i*.12:.2f}s">
  <circle cx="{622}" cy="{y}" r="16" fill="none" stroke="{c}" stroke-opacity=".45"/>
  <path transform="translate(613 {y-9}) scale(.75)" d="{TROPHY}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>
  <text class="s" x="652" y="{y-2}" font-size="17" fill="{FG}">{t}</text>
  <text class="m" x="652" y="{y+16}" font-size="11" fill="{FG2}">{d}</text>
</g>"""
    chips_l = ["GSSoC · 8th globally", "Google Student Ambassador", "OS Connect Global Mentor '26",
               "ISDC 2025 · Top 25", "Mumbai Hacks · Top 100", "2 research papers"]
    chips, x, y = "", 440, 524
    for i, c in enumerate(chips_l):
        w = 22 + len(c) * 7.8
        if x + w > 1136:
            x, y = 440, y + 40
        chips += f'<g class="up" style="animation-delay:{1.1+i*.08:.2f}s"><rect x="{x}" y="{y}" width="{w:.0f}" height="30" rx="15" fill="{BG2}" stroke="{BORDER}"/><text class="m" x="{x+w/2:.0f}" y="{y+20}" font-size="12" text-anchor="middle" fill="{FG}">{esc(c)}</text></g>'
        x += w + 8
    css = f"""
  .{p} .up{{animation:{p}up .7s ease-out both}}
  .{p} .side{{animation:{p}side .7s ease-out both}}
  .{p} .sweep{{animation:{p}sweep 4s 1s ease-in-out infinite}}
  .{p} .pulse{{animation:{p}pulse 2s ease-in-out infinite;transform-box:fill-box;transform-origin:center}}
  .{p} .float{{animation:{p}float 4s ease-in-out infinite}}
  @keyframes {p}up{{from{{opacity:0;transform:translateY(14px)}}}}
  @keyframes {p}side{{from{{opacity:0;transform:translateX(14px)}}}}
  @keyframes {p}sweep{{0%{{transform:translateX(-260px)}}45%,100%{{transform:translateX(620px)}}}}
  @keyframes {p}pulse{{50%{{transform:scale(2.4);opacity:0}}}}
  @keyframes {p}float{{50%{{transform:translateY(-6px)}}}}"""
    body = f"""
<defs>
  <clipPath id="{p}-award"><rect x="64" y="254" width="496" height="234" rx="18"/></clipPath>
  <linearGradient id="{p}-gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{GOLD}" stop-opacity=".16"/><stop offset=".6" stop-color="{GOLD}" stop-opacity="0"/></linearGradient>
  <linearGradient id="{p}-sheen" x1="0" x2="1"><stop offset="0" stop-color="{FG}" stop-opacity="0"/><stop offset=".5" stop-color="{FG}" stop-opacity=".09"/><stop offset="1" stop-color="{FG}" stop-opacity="0"/></linearGradient>
</defs>
<text class="m" x="64" y="76" font-size="13" letter-spacing="3" fill="{FG2}">05 / ACHIEVEMENTS</text>
{tiles}
<g class="up" style="animation-delay:.5s">
  <rect x="64" y="254" width="496" height="234" rx="18" fill="{BG2}" stroke="{GOLD}" stroke-opacity=".35"/>
  <g clip-path="url(#{p}-award)">
    <rect x="64" y="254" width="496" height="234" fill="url(#{p}-gold)"/>
    <rect class="sweep" x="64" y="254" width="180" height="234" fill="url(#{p}-sheen)" transform="skewX(-14)"/>
  </g>
  <g class="float"><path transform="translate(452 286) scale(3.4)" d="{TROPHY}" fill="none" stroke="{GOLD}" stroke-width=".9" stroke-linejoin="round"/></g>
  <text class="m" x="96" y="304" font-size="12" letter-spacing="3" fill="{GOLD}">AWARD · 2025–26</text>
  <text class="d" x="94" y="364" font-size="46" letter-spacing="-1.5" fill="{FG}">Young</text>
  <text class="d" x="94" y="412" font-size="46" letter-spacing="-1.5" fill="{FG}">Entrepreneur</text>
  <text class="m" x="96" y="456" font-size="13" fill="{FG2}">Two startups scaled from bootstrap → Pvt Ltd</text>
</g>
<rect x="584" y="254" width="552" height="234" rx="18" fill="{BG2}" stroke="{BORDER}"/>
{rows}
<g class="up" style="animation-delay:1s">
  <rect x="64" y="512" width="352" height="96" rx="16" fill="{BG2}" stroke="{ACC}" stroke-opacity=".35"/>
  <circle class="pulse" cx="92" cy="541" r="5" fill="{ACC}"/><circle cx="92" cy="541" r="5" fill="{ACC}"/>
  <text class="m" x="106" y="545" font-size="11" letter-spacing="2.5" fill="{ACC}">CURRENTLY BUILDING</text>
  <text class="d" x="86" y="588" font-size="32" letter-spacing="-1" fill="{FG}">gwava.dev</text>
  <text class="s" x="392" y="588" font-size="17" text-anchor="end" fill="{FG2}">300+ users</text>
</g>
{chips}"""
    svg("achievements.svg", 1200, 640, css, body, p)


# ---------------------------------------------------------------- connect
SOCIALS = [("github", "GitHub", "@parthparmar07"),
           ("linkedin", "LinkedIn", "in/parthparmar04"),
           ("x", "X / Twitter", "@parthparmar07"),
           ("gmail", "Email", "parthparmar4105@gmail.com"),
           ("cal", "Book a call", "cal.com/parthparmar"),
           ("netlify", "Portfolio", "parthparmarr.netlify.app")]


CAL = "M3 4h18v17H3zM5 9v10h14V9zM7 1h2v3H7zM15 1h2v3h-2zM7 11h4v4H7z"  # evenodd calendar glyph


def connect():
    p = "cn"
    cards = ""
    for i, (slug, name, handle) in enumerate(SOCIALS):
        x, y = 470 + (i % 2) * 350, 70 + (i // 2) * 96
        try:
            d = CAL if slug == "cal" else icon(slug)
        except Exception:  # brand removed from Simple Icons: fall back to a plain dot
            d = "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16z"
        cards += f"""<g class="fade" style="animation-delay:{.2+i*.1:.1f}s">
  <rect x="{x}" y="{y}" width="330" height="76" rx="16" fill="{BG2}" stroke="{BORDER}"/>
  <circle cx="{x+40}" cy="{y+38}" r="22" fill="{BG}" stroke="{BORDER}"/><path transform="translate({x+29} {y+27}) scale(.9167)" d="{d}" fill="{FG}" fill-rule="evenodd"/>
  <text class="s" x="{x+76}" y="{y+34}" font-size="18" fill="{FG}">{name}</text>
  <text class="m" x="{x+76}" y="{y+56}" font-size="12" fill="{FG2}">{handle}</text>
  <text class="m nudge" x="{x+300}" y="{y+44}" font-size="18" text-anchor="end" fill="{ACC}" style="animation-delay:{i*.2:.1f}s">→</text></g>"""
    css = f"""
  .{p} .fade{{animation:{p}fade .7s ease-out both}}
  .{p} .nudge{{animation:{p}nudge 1.6s ease-in-out infinite}}
  .{p} .rise{{animation:{p}rise 1s cubic-bezier(.2,.8,.2,1) both}}
  @keyframes {p}fade{{from{{opacity:0;transform:translateX(16px)}}}}
  @keyframes {p}nudge{{50%{{transform:translateX(6px)}}}}
  @keyframes {p}rise{{from{{transform:translateY(90px)}}}}"""
    body = f"""
<image href="{portrait(360)}" x="40" y="40" width="360" height="360"/>
<rect x="0" y="300" width="440" height="100" fill="url(#{p}-fadeb)"/>
<defs>
  <linearGradient id="{p}-fadeb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{BG}" stop-opacity="0"/><stop offset="1" stop-color="{BG}"/></linearGradient>
  <clipPath id="{p}-t"><rect x="0" y="300" width="460" height="80"/></clipPath>
</defs>
<g clip-path="url(#{p}-t)"><text class="d rise" x="56" y="372" font-size="68" letter-spacing="-3" fill="{FG}">LET'S BUILD.</text></g>
<text class="m" x="470" y="48" font-size="13" letter-spacing="3" fill="{FG2}">06 / CONNECT</text>
{cards}"""
    svg("connect.svg", 1200, 400, css, body, p)


if __name__ == "__main__":
    import datetime, json
    u = json.loads(get("https://api.github.com/users/parthparmar07"))
    gh = {"public_repos": u["public_repos"], "followers": u["followers"],
          "date": datetime.date.today().strftime("%b %Y")}
    hero(); about(); stack(); id_dashboard(gh); achievements(); connect()
