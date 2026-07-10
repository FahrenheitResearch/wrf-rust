// Progressive enhancement for the diagnostics reference table:
// column sorting and substring filtering. The table is complete without JS.
(function () {
  "use strict";

  var table = document.getElementById("var-table");
  var filter = document.getElementById("var-filter");
  var count = document.getElementById("row-count");
  if (!table) return;

  var tbody = table.tBodies[0];
  var originalRows = Array.prototype.slice.call(tbody.rows);
  var total = originalRows.length;
  var headers = table.tHead.rows[0].cells;
  var sortCol = -1;
  var sortDir = 1;

  function cellText(row, i) {
    return row.cells[i].textContent.trim().toLowerCase();
  }

  function applySort() {
    var rows;
    if (sortCol === -1) {
      rows = originalRows.slice();
    } else {
      rows = originalRows.slice().sort(function (a, b) {
        var x = cellText(a, sortCol);
        var y = cellText(b, sortCol);
        if (x === y) return 0;
        return x < y ? -sortDir : sortDir;
      });
    }
    for (var i = 0; i < rows.length; i++) tbody.appendChild(rows[i]);
  }

  function applyFilter() {
    var q = filter ? filter.value.trim().toLowerCase() : "";
    var visible = 0;
    for (var i = 0; i < originalRows.length; i++) {
      var row = originalRows[i];
      var hit = q === "" || row.textContent.toLowerCase().indexOf(q) !== -1;
      row.style.display = hit ? "" : "none";
      if (hit) visible++;
    }
    if (count) count.textContent = visible + " of " + total + " variables";
  }

  Array.prototype.forEach.call(headers, function (th, i) {
    if (!th.hasAttribute("data-sort")) return;
    th.addEventListener("click", function () {
      if (sortCol === i) {
        sortDir = -sortDir;
      } else {
        sortCol = i;
        sortDir = 1;
      }
      Array.prototype.forEach.call(headers, function (h) {
        h.removeAttribute("aria-sort");
      });
      th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
      applySort();
    });
  });

  if (filter) filter.addEventListener("input", applyFilter);
})();
