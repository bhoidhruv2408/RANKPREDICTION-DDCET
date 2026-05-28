/* charts.js — Chart.js helper utilities for DDCET Analytics page */
/* Charts are initialised inline in analytics.html using Chart.js CDN     */

// Utility: create gradient for canvas charts
function makeGradient(ctx, color1, color2) {
  const g = ctx.createLinearGradient(0, 0, 0, 400);
  g.addColorStop(0, color1);
  g.addColorStop(1, color2);
  return g;
}
