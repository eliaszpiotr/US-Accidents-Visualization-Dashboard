/* Restore double-click-to-reset on the accident map.
 *
 * plotly.js registers its map reset on the maplibre instance (map.on('dblclick') ->
 * viewInitial), but with dragmode 'lasso'/'select' it disables dragPan and takes over
 * the subplot div with its own dragElement, so maplibre never sees the clicks. Unlike
 * the geo subplot, the map subplot passes no 'doubleclick' handler to dragOptions,
 * so nothing runs. We reproduce the built-in reset from the same viewInitial snapshot.
 */
(function () {
  var MAP_ID = "accident-map";

  function findGraphDiv(host) {
    if (host.classList.contains("js-plotly-plot")) {
      return host;
    }
    return host.querySelector(".js-plotly-plot");
  }

  function resetMapView(event) {
    if (!event.target || !event.target.closest) {
      return;
    }
    var host = event.target.closest("#" + MAP_ID);
    if (!host || !window.Plotly) {
      return;
    }
    var graphDiv = findGraphDiv(host);
    var subplot =
      graphDiv &&
      graphDiv._fullLayout &&
      graphDiv._fullLayout.map &&
      graphDiv._fullLayout.map._subplot;
    var initial = subplot && subplot.viewInitial;
    if (!initial || !initial.center) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    // Going through relayout keeps Dash in the loop: the resulting plotly_relayout
    // feeds the map-viewport store, which hides the minimap viewport box.
    window.Plotly.relayout(graphDiv, {
      "map.center": { lon: initial.center.lon, lat: initial.center.lat },
      "map.zoom": initial.zoom,
      "map.bearing": initial.bearing || 0,
      "map.pitch": initial.pitch || 0,
    });
  }

  document.addEventListener("dblclick", resetMapView, true);
})();
