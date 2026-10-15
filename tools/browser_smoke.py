"""Rendered browser verification; requires a separately running local app."""
import os
from pathlib import Path
import sys
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from lab.catalog import REGISTRY

url = os.getenv("LAB_URL", "http://127.0.0.1:8505")
output = Path("test-results")
output.mkdir(exist_ok=True)


def stable(page):
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') === 'notRunning'")


def select_tab(page, label):
    page.get_by_role("tab", name=label, exact=True).click()
    stable(page)
    return page.get_by_role("tabpanel")


def show_explorer_parameters(page):
    sidebar = page.locator('[data-testid="stSidebar"]')
    expander = sidebar.locator("details").filter(has=page.get_by_text("Parámetros del explorador", exact=True))
    if expander.get_attribute("open") is None:
        expander.locator("summary").click()
    return sidebar


def select_family(page, kind):
    page.locator('[data-testid="stSidebar"]').get_by_text(kind, exact=True).click()
    allowed = [spec.name for spec in REGISTRY.values()
               if (spec.kind == "continuous") == (kind == "Continuas")]
    expect(page.get_by_role("heading", name=re.compile("^(" + "|".join(re.escape(name) for name in allowed) + ")$"))).to_be_visible()
    stable(page)


with sync_playwright() as p:
    executable = os.getenv("LAB_BROWSER_EXECUTABLE")
    browser = p.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
    page = browser.new_page(viewport={"width":1280,"height":1000})
    page.goto(url)
    page.get_by_role("heading", name="Distribuciones Lab", exact=True).last.wait_for()
    assert page.get_by_role("tab", name="Caso práctico", exact=True).get_attribute("aria-selected") == "true"
    guided = page.get_by_role("tabpanel")
    expect(guided.get_by_role("heading", name="3. Resolvámoslo paso a paso", exact=True)).to_be_visible()
    expect(guided.locator(".js-plotly-plot").first).to_be_visible()
    page.screenshot(path=str(output / "normal-guided-case.png"))
    panel = select_tab(page, "Explorar")
    sidebar = show_explorer_parameters(page)
    panel.locator(".js-plotly-plot").first.wait_for()
    page.screenshot(path=str(output / "normal-light.png"))
    print("title",page.title())
    print("formulas", page.locator(".katex").count())
    print("charts", page.locator(".js-plotly-plot").count())
    print("formula_errors",page.locator(".katex-error").count())
    print("exceptions",page.locator('[data-testid="stException"]').count())
    assert page.locator(".katex").count() > 0
    assert panel.locator(".js-plotly-plot").count() == 2
    assert page.locator(".katex-error").count() == 0
    assert page.locator('[data-testid="stException"]').count() == 0
    print("app_attributes",page.locator('[data-testid="stApp"]').evaluate("e=>Object.fromEntries([...e.attributes].map(a=>[a.name,a.value]))"))
    panel.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
    page.screenshot(path=str(output / "normal-light-charts.png"))
    mean_input = sidebar.get_by_role("spinbutton").first
    mean_input.fill("1")
    mean_input.press("Enter")
    panel.locator('[data-testid="stMetricValue"]').first.get_by_text("0,15865525",exact=True).wait_for()
    mode = panel.locator(".js-plotly-plot").first.evaluate("e=>{const t=e._fullData.find(t=>t.name==='Densidad f(x)');const y=Array.from(t.y);return t.x[y.indexOf(Math.max(...y))];}")
    assert abs(mode-1) < 0.03, mode
    page.locator('[data-testid="stSidebar"]').get_by_role("button",name="Restablecer valores predeterminados",exact=True).click()
    panel.locator('[data-testid="stMetricValue"]').first.get_by_text("0,5",exact=True).wait_for()
    print("browser controls update numeric probability and plotted density; reset OK")
    for kind in ([] if os.getenv("LAB_QUICK") else ["Continuas", "Discretas"]):
        select_family(page, kind)
        for id, spec in REGISTRY.items():
            if (spec.kind == "continuous") != (kind == "Continuas"):
                continue
            selector = page.locator('[data-testid="stSidebar"]').get_by_role("combobox")
            selector.click()
            page.get_by_role("option",name=spec.name,exact=True).click()
            page.get_by_role("heading",name=spec.name,exact=True).wait_for()
            panel = select_tab(page, "Explorar")
            assert page.locator(".katex-error").count() == 0, id
            assert page.locator('[data-testid="stException"]').count() == 0, id
            assert panel.locator(".js-plotly-plot").count() == (3 if id == "multinomial" else 2), id
            for tab_label in ["Entender", "Caso práctico", "Explorar"]:
                page.get_by_role("tab",name=tab_label,exact=True).click()
                stable(page)
                assert page.locator(".katex-error").count() == 0, (id,tab_label)
                if tab_label == "Caso práctico":
                    case_panel = page.get_by_role("tabpanel")
                    expect(case_panel.get_by_role("heading", name="5. Qué nos dice la respuesta", exact=True)).to_be_visible()
                    expect(case_panel.locator(".js-plotly-plot").first).to_be_visible()
            print("rendered",id,"formula and charts OK")
            if id in {"multinomial","genextreme","geom"}:
                page.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
                page.screenshot(path=str(output / f"{id}-light-charts.png"))
    select_family(page, "Continuas")
    page.locator('[data-testid="stSidebar"]').get_by_role("combobox").click()
    page.get_by_role("option",name=REGISTRY["f"].name,exact=True).click()
    page.get_by_role("heading",name=REGISTRY["f"].name,exact=True).wait_for()
    page.get_by_role("tab",name="Explorar",exact=True).click()
    stable(page)
    panel = page.get_by_role("tabpanel")
    inputs = show_explorer_parameters(page).get_by_role("spinbutton")
    inputs.nth(0).fill("0.1")
    inputs.nth(0).press("Enter")
    inputs.nth(1).fill("0.1")
    inputs.nth(1).press("Enter")
    panel.get_by_text(re.compile("Se limitó el ancho visible")).wait_for()
    plot_range = panel.locator(".js-plotly-plot").first.evaluate("e=>e._fullLayout.xaxis.range")
    assert plot_range[1] < 100, plot_range
    panel.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
    page.screenshot(path=str(output / "f-heavy-tail.png"))
    print("F heavy tail controlled window",plot_range)
    dark = browser.new_context(viewport={"width":1280,"height":1000},color_scheme="dark")
    dark_page = dark.new_page()
    dark_page.goto(url)
    dark_page.get_by_role("tab", name="Explorar", exact=True).wait_for()
    dark_page.get_by_role("tab", name="Explorar", exact=True).click()
    stable(dark_page)
    dark_page.locator(".js-plotly-plot").first.wait_for()
    dark_page.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
    dark_page.screenshot(path=str(output / "normal-dark-charts.png"))
    print("dark_bg",dark_page.locator('[data-testid="stApp"]').evaluate("e=>getComputedStyle(e).backgroundColor"))
    text_color = dark_page.locator(".gtitle").first.evaluate("e=>getComputedStyle(e).fill")
    print("dark_chart_text",text_color)
    assert text_color not in {"rgb(68, 68, 68)", "rgb(42, 63, 95)"}, "Dark chart labels do not inherit theme"
    mobile = browser.new_context(viewport={"width":390,"height":844},is_mobile=True)
    mobile_page = mobile.new_page()
    mobile_page.goto(url)
    mobile_page.get_by_role("tab", name="Caso práctico", exact=True).wait_for()
    assert mobile_page.get_by_role("tab", name="Caso práctico", exact=True).get_attribute("aria-selected") == "true"
    mobile_page.get_by_role("tabpanel").locator(".js-plotly-plot").first.wait_for()
    mobile_page.screenshot(path=str(output / "normal-mobile.png"))
    mobile_page.get_by_role("tabpanel").locator(".js-plotly-plot").first.scroll_into_view_if_needed()
    mobile_page.screenshot(path=str(output / "normal-mobile-charts.png"))
    mobile_page.get_by_role("tab", name="Explorar", exact=True).click()
    stable(mobile_page)
    mobile_page.get_by_role("tabpanel").locator(".js-plotly-plot").first.wait_for()
    assert mobile_page.locator(".katex-error").count() == 0
    print("mobile rendered at390px")
    browser.close()
