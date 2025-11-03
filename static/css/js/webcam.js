const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

let stream;

// Start camera
startBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    startBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');

    simulatePredictions(); // simulate ML predictions for now
  } catch (err) {
    alert('Camera access denied.');
  }
});

// Stop camera
stopBtn.addEventListener('click', () => {
  if (stream) {
    const tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
  }
  video.srcObject = null;
  startBtn.classList.remove('hidden');
  stopBtn.classList.add('hidden');
});

// Simulated prediction updates (replace with backend fetch later)
function simulatePredictions() {
  setInterval(() => {
    const cnnAcc = Math.floor(Math.random() * 100);
    const resAcc = Math.floor(Math.random() * 100);

    document.getElementById('cnn-label').textContent = `Prediction: Cat (${cnnAcc}%)`;
    document.getElementById('cnn-bar').style.width = `${cnnAcc}%`;

    document.getElementById('resnet-label').textContent = `Prediction: Dog (${resAcc}%)`;
    document.getElementById('resnet-bar').style.width = `${resAcc}%`;
  }, 2000);
}
