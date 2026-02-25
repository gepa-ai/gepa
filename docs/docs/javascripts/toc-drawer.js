/**
 * TOC Drawer
 *
 * Creates a standalone fixed-position drawer for the table of contents,
 * toggled by the .toc-toggle button in the header.
 *
 * On regular pages Material already renders .md-sidebar--secondary with the
 * TOC; on blog posts (which override the container block) it may not exist.
 * This script handles both cases:
 *   1. If a Material TOC nav exists, clone its link list into the drawer.
 *   2. Otherwise, build the list from h2/h3 headings in .md-content.
 */
(function () {
  var toggle = document.querySelector(".toc-toggle");
  if (!toggle) return;

  /* ---- Build drawer element ---- */
  var drawer = document.createElement("nav");
  drawer.className = "toc-drawer";
  drawer.setAttribute("aria-label", "Table of contents");

  // Title
  var title = document.createElement("div");
  title.className = "toc-drawer__title";
  title.textContent = "Table of contents";
  drawer.appendChild(title);

  // Try to get TOC from Material's rendered sidebar
  var materialNav = document.querySelector(
    ".md-sidebar--secondary .md-nav--secondary"
  );

  if (materialNav) {
    // Clone the nav list so we don't move the original (needed for desktop sidebar)
    var clone = materialNav.cloneNode(true);
    drawer.appendChild(clone);
  } else {
    // Fallback: build TOC from content headings
    var list = document.createElement("ul");
    list.className = "md-nav__list";

    var headings = document.querySelectorAll(
      ".md-content h2[id], .md-content h3[id]"
    );
    headings.forEach(function (h) {
      var li = document.createElement("li");
      li.className = "md-nav__item";
      if (h.tagName === "H3") li.classList.add("toc-drawer__indent");

      var a = document.createElement("a");
      a.className = "md-nav__link";
      a.href = "#" + h.id;
      // Strip the permalink ¶ character that Material appends
      a.textContent = h.textContent.replace(/\s*¶\s*$/, "").trim();

      li.appendChild(a);
      list.appendChild(li);
    });

    var nav = document.createElement("nav");
    nav.className = "md-nav md-nav--secondary";
    nav.appendChild(list);
    drawer.appendChild(nav);
  }

  document.body.appendChild(drawer);

  /* ---- Overlay ---- */
  var overlay = document.createElement("div");
  overlay.className = "toc-overlay";
  document.body.appendChild(overlay);

  /* ---- Open / close helpers ---- */
  function open() {
    document.body.setAttribute("data-md-toc-open", "");
  }
  function close() {
    document.body.removeAttribute("data-md-toc-open");
  }

  toggle.addEventListener("click", function () {
    document.body.hasAttribute("data-md-toc-open") ? close() : open();
  });

  overlay.addEventListener("click", close);

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });

  // Close when a TOC link is tapped (mobile convenience)
  drawer.addEventListener("click", function (e) {
    if (e.target.closest("a")) close();
  });
})();
