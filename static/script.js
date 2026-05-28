/* DDQuest DDCET Predictor — script.js */

document.addEventListener('DOMContentLoaded', () => {

  // ── Marks slider / input sync ──────────────────────────
  const slider  = document.getElementById('marksSlider');
  const input   = document.getElementById('marksInput');
  const display = document.getElementById('marksDisplay');

  if (slider && input && display) {
    function syncMarks(val) {
      let n = parseFloat(val);
      if (isNaN(n)) n = 0;
      n = Math.min(200, Math.max(0, n));

      if (parseFloat(input.value) !== n) input.value = n;
      if (parseFloat(slider.value) !== n) slider.value = n;
      display.textContent = n.toFixed(1);

      const pct = (n / 200) * 100;
      slider.style.background =
        `linear-gradient(to right, #6366f1 ${pct}%, rgba(255,255,255,.08) ${pct}%)`;
    }

    slider.addEventListener('input', () => syncMarks(slider.value));
    input.addEventListener('input',  () => syncMarks(input.value));
    input.addEventListener('change', () => syncMarks(input.value));

    // Init to slider default
    syncMarks(slider.value);
  }

  // ── Form submit loader ─────────────────────────────────
  const form     = document.querySelector('.predict-form');
  const btnText  = document.querySelector('.btn-text');
  const btnLoad  = document.querySelector('.btn-loader');
  const btnArrow = document.querySelector('.btn-arrow');

  if (form) {
    form.addEventListener('submit', (e) => {
      const marksVal = parseFloat(document.getElementById('marksInput')?.value);
      if (isNaN(marksVal) || marksVal < 0 || marksVal > 200) {
        e.preventDefault();
        alert('Please enter valid marks between 0 and 200.');
        return;
      }
      if (btnText)  btnText.classList.add('hidden');
      if (btnLoad)  btnLoad.classList.remove('hidden');
      if (btnArrow) btnArrow.classList.add('hidden');
    });
  }

  // ── Animate cards on scroll ────────────────────────────
  const cards = document.querySelectorAll('.college-card, .feature-card');
  if ('IntersectionObserver' in window && cards.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((e, i) => {
        if (e.isIntersecting) {
          setTimeout(() => e.target.classList.add('fade-in'), i * 40);
          observer.unobserve(e.target);
        }
      });
    }, { threshold: 0.05 });

    cards.forEach(c => observer.observe(c));
  }

});
