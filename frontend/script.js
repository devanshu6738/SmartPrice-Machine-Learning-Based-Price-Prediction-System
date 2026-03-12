const form = document.getElementById('predict-form');
const result = document.getElementById('result');

const API_URL = 'http://localhost:5000/predict';

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  result.textContent = 'Predicting...';

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  for (const key of ['ram', 'storage', 'processor_speed', 'battery_capacity', 'camera_mp']) {
    payload[key] = Number(payload[key]);
  }

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    result.textContent = `Estimated price: ?${data.predicted_price}`;
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
  }
});
