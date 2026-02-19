/**
 * Auto-open a <details> element when the URL hash points to an anchor
 * inside or immediately before it.
 */
(function () {
  "use strict";

  function openDetailsForHash() {
    var hash = window.location.hash;
    if (!hash) return;

    var target = document.querySelector(hash);
    if (!target) return;

    // The anchor <span> may be wrapped in a <p> by the markdown parser.
    // Walk up through parents checking siblings to find the closest <details>.
    var details = target.closest("details");
    if (!details) {
      var el = target;
      while (el && !details) {
        var next = el.nextElementSibling;
        if (next && next.tagName === "DETAILS") {
          details = next;
          break;
        }
        // Walk up to parent and check its next sibling
        el = el.parentElement;
      }
    }
    if (!details) return;

    details.open = true;

    // Scroll the target into view after a brief delay to let the browser
    // lay out the newly-opened content.
    setTimeout(function () {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }

  // Run on initial load and when hash changes (e.g. clicking in-page links)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", openDetailsForHash);
  } else {
    openDetailsForHash();
  }
  window.addEventListener("hashchange", openDetailsForHash);
})();
