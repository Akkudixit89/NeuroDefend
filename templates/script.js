<script>
  const accuracies = {{ accuracies|tojson }};
  const f1Val = Math.round(parseFloat("{{ metrics.f1_score }}") * 100);
  const prediction = "{{ prediction }}";

  // Create gradient color depending on prediction
  function getPredictionGradient(ctx, prediction) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    if (prediction === "Malicious") {
      gradient.addColorStop(0, "#ef4444");
      gradient.addColorStop(1, "#b91c1c");
    } else if (prediction === "Benign") {
      gradient.addColorStop(0, "#10b981");
      gradient.addColorStop(1, "#047857");
    } else {
      gradient.addColorStop(0, "#9ca3af");
      gradient.addColorStop(1, "#6b7280");
    }
    return gradient;
  }

  // Accuracy Bar Chart
  const accCtx = document.getElementById('accChart').getContext('2d');
  new Chart(accCtx, {
    type: 'bar',
    data: {
      labels: Object.keys(accuracies),
      datasets: [{
        label: 'Accuracy (%)',
        data: Object.values(accuracies),
        backgroundColor: Object.keys(accuracies).map(() => getPredictionGradient(accCtx, prediction)),
        borderRadius: 10
      }]
    },
    options: {
      animation: { duration: 1800, easing: 'easeOutElastic' },
      plugins: {
        tooltip: { enabled: true },
        legend: { display: false }
      },
      scales: { y: { beginAtZero: true, max: 100 } }
    }
  });

  // F1 Score Gauge (doughnut)
  const f1Ctx = document.getElementById('f1Gauge').getContext('2d');
  new Chart(f1Ctx, {
    type: 'doughnut',
    data: {
      labels: ['F1', 'Remaining'],
      datasets: [{
        data: [f1Val, 100 - f1Val],
        backgroundColor: [getPredictionGradient(f1Ctx, prediction), '#e5e7eb'],
        cutout: '75%',
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        datalabels: {
          display: true,
          color: '#111827',
          formatter: () => f1Val + '%'
        }
      },
      animation: { duration: 1500, easing: 'easeInOutCubic' }
    },
    plugins: [ChartDataLabels]
  });

  // Confusion Matrix (heat-style horizontal bars)
  const cmCtx = document.getElementById('cmHeat').getContext('2d');
  new Chart(cmCtx, {
    type: 'bar',
    data: {
      labels: ['Pred Benign', 'Pred Malicious'],
      datasets: [
        {
          label: 'True Benign',
          data: [{{ cm[0][0] }}, {{ cm[0][1] }}],
          backgroundColor: '#3b82f6'
        },
        {
          label: 'True Malicious',
          data: [{{ cm[1][0] }}, {{ cm[1][1] }}],
          backgroundColor: '#f59e0b'
        }
      ]
    },
    options: {
      indexAxis: 'y',
      animation: { duration: 1800 },
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { beginAtZero: true } }
    }
  });

  // Dynamic Performance Trend (Line)
  const perfCtx = document.getElementById('performanceChart').getContext('2d');
  new Chart(perfCtx, {
    type: 'line',
    data: {
      labels: ['Precision', 'Recall', 'F1', 'Accuracy'],
      datasets: [{
        label: 'Performance (%)',
        data: [
          {{ metrics.precision }},
          {{ metrics.recall }},
          {{ metrics.f1_score }},
          {{ metrics.accuracy }}
        ],
        borderColor: getPredictionGradient(perfCtx, prediction),
        backgroundColor: 'rgba(59,130,246,0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 5,
        pointHoverRadius: 7
      }]
    },
    options: {
      animation: { duration: 2000, easing: 'easeOutQuart' },
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 100 } }
    }
  });
</script>
