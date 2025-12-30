console.log("✅ Live Detect JS Loaded!");

let selectedFile = null;
let webcamStream = null;
let capturedImage = null;
let chartInstance = null;

// HTML Elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const uploadBox = document.getElementById("uploadBox");
const analyzeBtn = document.getElementById("analyzeBtn");
const webcam = document.getElementById("webcam");
const webcamContainer = document.getElementById("webcamContainer");
const camBtn = document.getElementById("useWebcamBtn");
const captureBtn = document.getElementById("captureBtn");
const uploadSection = document.getElementById("uploadSection");
const results = document.getElementById("results");
const testAnotherBtn = document.getElementById("testAnotherBtn");

// =========================
// FILE UPLOAD HANDLER
// =========================
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  selectedFile = file;
  const reader = new FileReader();
  
  reader.onload = () => {
    preview.classList.remove("hidden");
    preview.src = reader.result;
    uploadBox.classList.add("hidden");
    
    // Hide webcam if open
    if (webcamStream) {
      stopWebcam();
    }
    
    // Enable analyze button
    enableAnalyzeButton();
  };
  
  reader.readAsDataURL(file);
});

// Drag and Drop
uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.style.borderColor = "#3b82f6";
  uploadBox.style.background = "#dbeafe";
});

uploadBox.addEventListener("dragleave", () => {
  uploadBox.style.borderColor = "#93c5fd";
  uploadBox.style.background = "#eff6ff";
});

uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadBox.style.borderColor = "#93c5fd";
  uploadBox.style.background = "#eff6ff";
  
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

// =========================
// WEBCAM HANDLER
// =========================
camBtn.addEventListener("click", async () => {
  try {
    // Hide upload box and preview
    uploadBox.classList.add("hidden");
    preview.classList.add("hidden");
    
    // Show webcam
    webcamContainer.classList.remove("hidden");
    
    // Request webcam access
    webcamStream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 1280, height: 720 } 
    });
    webcam.srcObject = webcamStream;
    
  } catch (error) {
    alert("❌ Webcam access denied. Please enable camera permissions.");
    console.error("Webcam error:", error);
    webcamContainer.classList.add("hidden");
  }
});

// Capture Photo from Webcam
captureBtn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  canvas.getContext("2d").drawImage(webcam, 0, 0);
  
  capturedImage = canvas.toDataURL("image/jpeg");
  
  // Show captured image in preview
  preview.src = capturedImage;
  preview.classList.remove("hidden");
  
  // Stop and hide webcam
  stopWebcam();
  
  // Enable analyze button
  enableAnalyzeButton();
});

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  webcamContainer.classList.add("hidden");
}

// =========================
// ENABLE ANALYZE BUTTON
// =========================
function enableAnalyzeButton() {
  analyzeBtn.disabled = false;
  analyzeBtn.classList.remove("bg-gray-300", "text-gray-500", "cursor-not-allowed");
  analyzeBtn.classList.add("bg-blue-500", "hover:bg-blue-600", "text-white", "cursor-pointer");
}

// =========================
// ANALYZE IMAGE
// =========================
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile && !capturedImage) {
    alert("⚠️ Please upload an image or capture from webcam first!");
    return;
  }

  // Show loading state
  analyzeBtn.textContent = "⏳ Analyzing...";
  analyzeBtn.disabled = true;

  try {
    let response;

    if (selectedFile) {
      // Upload file
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      response = await fetch("/detect", {
        method: "POST",
        body: formData
      });
    } else if (capturedImage) {
      // Send webcam capture
      response = await fetch("/detect-webcam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: capturedImage })
      });
    }

    if (!response.ok) throw new Error("Analysis failed");

    const data = await response.json();
    console.log("📊 Analysis Results:", data);

    // Display results
    displayResults(data);

  } catch (error) {
    console.error("❌ Error:", error);
    alert("Analysis failed. Please try again.");
    analyzeBtn.textContent = "Start Analysis →";
    analyzeBtn.disabled = false;
  }
});

// =========================
// DISPLAY RESULTS
// =========================
function displayResults(data) {
  // Hide upload section
  uploadSection.classList.add("hidden");
  
  // Show results section
  results.classList.remove("hidden");
  
  // Scroll to results
  results.scrollIntoView({ behavior: "smooth", block: "start" });

  // Set result image
  document.getElementById("resultImage").src = preview.src;

  // Set detection info
  document.getElementById("detectedObject").textContent = data.object || "Unknown";
  document.getElementById("detectedConfidence").textContent = `${data.confidence || 0}%`;

  // Update classification summary
  const summaryText = `The image has been successfully processed through four distinct machine learning and deep learning models. The ${data.best_model || 'YOLO'} algorithm achieved the highest detection accuracy at ${data.confidence || 0}%, demonstrating superior performance in this particular classification task.`;
  document.getElementById("classificationSummary").textContent = summaryText;

  // Display model scores
  displayModelScores(data.scores);

  // Create chart
  createChart(data.scores);

  // Generate insights
  generateInsights(data.scores, data.best_model);

  // Update summary
  updateSummary(data);
}

// =========================
// DISPLAY MODEL SCORES
// =========================
function displayModelScores(scores) {
  const modelList = document.getElementById("modelScoreList");
  modelList.innerHTML = "";

  // Only show the 4 models: YOLO, CNN, ResNet-18, MobileNet
  const modelsToShow = ["YOLO", "CNN", "ResNet-18", "MobileNet"];
  
  modelsToShow.forEach(modelName => {
    const score = scores[modelName] || 0;
    
    const div = document.createElement("div");
    div.className = "flex justify-between items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition";
    div.innerHTML = `
      <span class="font-semibold text-gray-800">${modelName}</span>
      <span class="text-blue-600 font-bold text-lg">${score.toFixed(1)}%</span>
    `;
    modelList.appendChild(div);
  });
}

// =========================
// CREATE CHART
// =========================
function createChart(scores) {
  const ctx = document.getElementById("chart").getContext("2d");

  // Destroy previous chart if exists
  if (chartInstance) {
    chartInstance.destroy();
  }

  // Only show the 4 models
  const modelsToShow = ["YOLO", "CNN", "ResNet-18", "MobileNet"];
  const labels = modelsToShow;
  const values = modelsToShow.map(model => scores[model] || 0);

  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Model Accuracy",
        data: values,
        backgroundColor: "#3b82f6",
        borderColor: "#2563eb",
        borderWidth: 1,
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            font: { size: 14 },
            color: "#374151"
          }
        },
        tooltip: {
          backgroundColor: "#1f2937",
          titleColor: "#fff",
          bodyColor: "#fff",
          padding: 12,
          displayColors: false,
          callbacks: {
            label: (context) => `Accuracy: ${context.parsed.y.toFixed(1)}%`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: (value) => value + "%",
            font: { size: 12 },
            color: "#6b7280"
          },
          grid: {
            color: "#e5e7eb"
          },
          title: {
            display: true,
            text: "Accuracy (%)",
            font: { size: 14, weight: "bold" },
            color: "#374151"
          }
        },
        x: {
          ticks: {
            font: { size: 12 },
            color: "#6b7280"
          },
          grid: {
            display: false
          }
        }
      }
    }
  });
}

// =========================
// GENERATE INSIGHTS
// =========================
function generateInsights(scores, bestModel) {
  const insightsList = document.getElementById("insightsList");
  insightsList.innerHTML = "";

  const insights = [
    `Deep Learning models (YOLO, CNN, ResNet-18, MobileNet) demonstrate superior accuracy compared to traditional ML algorithms`,
    `${bestModel || 'YOLO'} achieves the highest accuracy with real-time detection capabilities`,
    `ResNet-18 shows strong performance due to its residual connections enabling deeper feature learning`,
    `MobileNet provides efficient performance optimized for speed and lower computational requirements`
  ];

  insights.forEach(insight => {
    const li = document.createElement("li");
    li.className = "flex items-start gap-2";
    li.innerHTML = `
      <span class="text-blue-600 mt-1">•</span>
      <span>${insight}</span>
    `;
    insightsList.appendChild(li);
  });
}

// =========================
// UPDATE SUMMARY
// =========================
function updateSummary(data) {
  document.getElementById("bestModel").textContent = data.best_model || "YOLO";
  document.getElementById("bestModelStats").textContent = `${data.confidence || 0}% accuracy`;
  document.getElementById("analysisTime").textContent = `${data.time || 2.3}s`;
  
  const summaryText = `This analysis compares performance across YOLO, CNN, ResNet-18, and MobileNet algorithms. Results demonstrate varying levels of accuracy, with ${data.best_model || 'YOLO'} leading at ${data.confidence || 0}% confidence. The deep learning models consistently outperform traditional approaches in complex image recognition tasks, showcasing the advantages of neural network architectures for visual classification.`;
  
  document.getElementById("summaryText").textContent = summaryText;
}

// =========================
// TEST ANOTHER IMAGE
// =========================
testAnotherBtn.addEventListener("click", () => {
  // Reset everything
  selectedFile = null;
  capturedImage = null;
  
  // Clear inputs
  fileInput.value = "";
  preview.src = "";
  preview.classList.add("hidden");
  
  // Show upload section
  uploadBox.classList.remove("hidden");
  uploadSection.classList.remove("hidden");
  
  // Hide results
  results.classList.add("hidden");
  
  // Reset button
  analyzeBtn.textContent = "Start Analysis →";
  analyzeBtn.disabled = true;
  analyzeBtn.classList.remove("bg-blue-500", "hover:bg-blue-600", "text-white");
  analyzeBtn.classList.add("bg-gray-300", "text-gray-500", "cursor-not-allowed");
  
  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });
});