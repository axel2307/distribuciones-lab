"""Render every guided case in a fresh Chrome profile (no account/session).

Run against an already started Streamlit instance. LAB_URL may point at the
existing public app for the same read-only verification after publication.
"""
import json
import os
import re
from pathlib import Path
import sys
import time

from playwright.sync_api import sync_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.catalog import REGISTRY

OUTPUT = Path(os.getenv("LAB_QA_OUTPUT", "test-results/guided-cases"))
OUTPUT.mkdir(parents=True, exist_ok=True)
URL = os.getenv("LAB_URL", "http://127.0.0.1:8506")


def stable(page):
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') === 'notRunning'")


def app_surface(browser_page):
    """Community Cloud may wrap the same app in an iframe."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        for frame in browser_page.frames:
            if frame.locator('[data-testid="stApp"]').count():
                return frame
        browser_page.wait_for_timeout(100)
    raise AssertionError("The Streamlit app did not appear in the page or its frames")


def choose(page, distribution_id):
    kind = "Continuas" if REGISTRY[distribution_id].kind == "continuous" else "Discretas"
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_text(kind, exact=True).click()
    allowed_names = [spec.name for spec in REGISTRY.values()
                     if (spec.kind == "continuous") == (kind == "Continuas")]
    expect(page.get_by_role("heading", name=re.compile("^(" + "|".join(re.escape(name) for name in allowed_names) + ")$"))).to_be_visible()
    stable(page)
    sidebar.get_by_role("combobox").click()
    page.get_by_role("option", name=REGISTRY[distribution_id].name, exact=True).click()
    page.get_by_role("heading", name=REGISTRY[distribution_id].name, exact=True).wait_for()
    case_tab = page.get_by_role("tab", name="Caso práctico", exact=True)
    if case_tab.get_attribute("aria-selected") != "true":
        case_tab.click()
    stable(page)


def capture(page, name):
    page.page.screenshot(path=str(OUTPUT / name))


with sync_playwright() as p:
    executable = os.getenv("LAB_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    browser = p.chromium.launch(headless=True, executable_path=executable)
    context = browser.new_context(viewport={"width":1360, "height":1000})
    browser_page = context.new_page()
    browser_page.goto(URL)
    page = app_surface(browser_page)
    page.get_by_role("tab", name="Caso práctico", exact=True).wait_for(timeout=120000)
    stable(page)
    assert page.get_by_role("tab", name="Caso práctico", exact=True).get_attribute("aria-selected") == "true"
    results = []
    for distribution_id in REGISTRY:
        choose(page, distribution_id)
        panel = page.get_by_role("tabpanel")
        expect(panel.get_by_role("heading", name="3. Resolvámoslo paso a paso", exact=True)).to_be_visible()
        assert panel.locator(".katex").count() >= 4, distribution_id
        assert page.locator(".katex-error").count() == 0, distribution_id
        assert page.locator('[data-testid="stException"]').count() == 0, distribution_id
        assert panel.locator('[data-testid="stAlertContainer"]').count() >= 1, distribution_id
        plot = panel.locator(".js-plotly-plot").first
        plot.scroll_into_view_if_needed()
        expect(plot).to_be_visible()
        traces = plot.evaluate("e=>e._fullData.map(t=>({name:t.name,type:t.type,points:t.x?.length}))")
        assert traces, distribution_id
        text = panel.inner_text()
        assert "5. Qué nos dice la respuesta" in text, distribution_id
        assert "¿Qué pasaría" in text or "¿qué pasaría" in text, distribution_id
        results.append({"id":distribution_id,"latex_count":panel.locator(".katex").count(),"plot_traces":traces})
        if distribution_id in {"binom", "t", "chi2", "f", "beta", "multinomial", "genextreme"}:
            capture(page, f"{distribution_id}-event-desktop.png")
            panel.get_by_role("heading").first.scroll_into_view_if_needed()
            capture(page, f"{distribution_id}-question-desktop.png")
        print("rendered", distribution_id, flush=True)
    choose(page, "binom")
    panel = page.get_by_role("tabpanel")
    n_label = f"{REGISTRY['binom'].parameters[0].name} (n)"
    panel.get_by_role("spinbutton", name=n_label, exact=True).fill("10")
    panel.get_by_role("spinbutton", name=n_label, exact=True).press("Enter")
    expect(panel.get_by_text(re.compile("Se fija un grupo de 10 visitantes"))).to_be_visible()
    stable(page)
    panel.get_by_role("spinbutton", name="Probabilidad de éxito (p)", exact=True).fill("0.5")
    panel.get_by_role("spinbutton", name="Probabilidad de éxito (p)", exact=True).press("Enter")
    expect(panel.get_by_text(re.compile("con probabilidad 0,5, de manera independiente"))).to_be_visible()
    stable(page)
    event_select = panel.get_by_role("combobox").first
    event_select.click()
    page.get_by_role("option", name="P(X = k)", exact=True).click()
    expect(panel.get_by_role("spinbutton", name="Resultado entero k", exact=True)).to_be_visible()
    stable(page)
    panel.get_by_role("spinbutton", name="Resultado entero k", exact=True).fill("5")
    panel.get_by_role("spinbutton", name="Resultado entero k", exact=True).press("Enter")
    stable(page)
    expect(panel.locator('[data-testid="stMetricValue"]').first).to_have_text("0,24609375")
    assert "10" in panel.get_by_role("heading").first.inner_text() or "10" in panel.inner_text()
    plot = panel.locator(".js-plotly-plot").first
    plot.scroll_into_view_if_needed()
    assert page.locator(".katex-error").count() == 0
    capture(page, "binom-edited-event.png")
    # Browser refresh reconstructs the authored case; no persistence beyond the
    # Streamlit session is claimed.
    browser_page.reload()
    page = app_surface(browser_page)
    page.get_by_role("tab", name="Caso práctico", exact=True).wait_for()
    stable(page)
    assert page.locator('[data-testid="stException"]').count() == 0
    mobile = browser.new_context(viewport={"width":390,"height":844}, is_mobile=True)
    mobile_browser_page = mobile.new_page()
    mobile_browser_page.goto(URL)
    mobile_page = app_surface(mobile_browser_page)
    mobile_page.get_by_role("tab", name="Caso práctico", exact=True).wait_for()
    for distribution_id in ["t", "multinomial", "norm"]:
        mobile_page.locator('[data-testid="stExpandSidebarButton"]').click()
        mobile_page.wait_for_function("document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().left >= 0")
        choose(mobile_page, distribution_id)
        mobile_page.locator('[data-testid="stSidebar"]').get_by_role("button").filter(has_text="keyboard_double_arrow_left").click()
        mobile_page.wait_for_function("document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().right <= 1")
        panel = mobile_page.get_by_role("tabpanel")
        panel.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
        capture(mobile_page, f"{distribution_id}-event-mobile.png")
        overflow = mobile_page.evaluate("document.documentElement.scrollWidth > innerWidth")
        assert not overflow, distribution_id
        bounds = panel.bounding_box()
        assert bounds["x"] >= 0 and bounds["width"] <= 390, (distribution_id, bounds)
    mobile.close()
    context.close()
    browser.close()

(OUTPUT / "results.json").write_text(json.dumps({"url":URL,"cases":results,"edited_binomial":0.24609375,
    "viewports":[[1360,1000],[390,844]],"exceptions":0,"katex_errors":0}, indent=2), encoding="utf8")
print("Verified all 24 guided cases, edited binomial, refresh, desktop/mobile.")
