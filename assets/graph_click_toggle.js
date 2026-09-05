/* Make a repeated click on the same mark toggle its filter off.
 *
 * Every click filter in this dashboard is a toggle: clicking the same state, month
 * or bar segment again is supposed to clear it. The server-side handlers do exactly
 * that, but they never run for the second click. Dash only invokes a callback when
 * an Input value actually changes, and plotly's clickData for the same point is
 * deep-equal every time (curveNumber, pointNumber, location, z, customdata, bbox are
 * all deterministic) - so the second click looks like "no change" and is dropped.
 *
 * Clearing the prop back to null right after each click makes the next click a real
 * change again. dash_clientside.set_props writes the prop without declaring a
 * dependency, so this does not create a callback cycle (clickData -> filter-state ->
 * clickData would be rejected as circular). The handlers already treat a null
 * clickData as a no-op, so the extra null pass changes nothing.
 */
(function () {
  var DOUBLE_CLICK_GRACE_MS = 400;

  function clearClickData(event) {
    if (!event.target || !event.target.closest) {
      return;
    }
    var host = event.target.closest(".dash-graph[id]");
    if (!host || !host.id) {
      return;
    }
    // Legend entries and modebar buttons never produce clickData, so there is nothing
    // to reset - and touching a prop here would re-render the graph and undo the
    // legend toggle the user just made.
    if (event.target.closest(".legend, .modebar")) {
      return;
    }
    var graphDiv = host.classList.contains("js-plotly-plot")
      ? host
      : host.querySelector(".js-plotly-plot");
    if (!graphDiv || !graphDiv.on) {
      return;
    }
    // The delay does two jobs. It lets Dash dispatch this click's filter update before
    // the prop is touched, and it keeps the original value in place for the length of a
    // double-click, so double-clicking to reset a map view does not also toggle off the
    // selection - only a deliberate, separate second click does.
    setTimeout(function () {
      var clientside = window.dash_clientside;
      if (!clientside || typeof clientside.set_props !== "function") {
        return;
      }
      clientside.set_props(host.id, { clickData: null });
    }, DOUBLE_CLICK_GRACE_MS);
  }

  document.addEventListener("click", clearClickData, true);
})();
