/* Lightweight JSON pretty printer for the dashboard. */
(function (global) {
  "use strict";

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
    );
  }

  function colorize(value, indent) {
    const pad = "  ".repeat(indent);
    if (value === null) return `<span class="null">null</span>`;
    if (typeof value === "boolean") return `<span class="b">${value}</span>`;
    if (typeof value === "number") return `<span class="n">${value}</span>`;
    if (typeof value === "string") return `<span class="s">"${esc(value)}"</span>`;
    if (Array.isArray(value)) {
      if (!value.length) return "[]";
      const parts = value.map((v) => pad + "  " + colorize(v, indent + 1));
      return "[\n" + parts.join(",\n") + "\n" + pad + "]";
    }
    if (typeof value === "object") {
      const keys = Object.keys(value);
      if (!keys.length) return "{}";
      const parts = keys.map(
        (k) => pad + "  " + `<span class="k">"${esc(k)}"</span>: ` + colorize(value[k], indent + 1)
      );
      return "{\n" + parts.join(",\n") + "\n" + pad + "}";
    }
    return esc(value);
  }

  function render(el, data, title) {
    if (!el) return;
    let obj = data;
    if (typeof data === "string") {
      try {
        obj = JSON.parse(data);
      } catch (_) {
        el.innerHTML = `<pre class="json-view">${esc(data)}</pre>`;
        return;
      }
    }
    const html = colorize(obj, 0);
    const label = title || "JSON";
    el.innerHTML = `<details class="json-block" open><summary>${esc(label)}</summary><div class="json-view">${html}</div></details>`;
  }

  global.JsonView = { render, colorize, esc };
})(window);
