window.onbeforeunload = function () {
    console.log("PAGE RELOADING");
};
const API_BASE = "http://127.0.0.1:8001";
let userPrompt = "";
let detectedTask = "";

async function interpret() {
    userPrompt = document.getElementById("inputText").value;

    let res = await fetch(`${API_BASE}/interpret`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({task: userPrompt})
    });

    let data = await res.json();
    detectedTask = data.task;

    document.getElementById("result").innerText =
        "Detected task: " + detectedTask;

    document.getElementById("confirmBox").style.display = "block";
}

function confirmTrain() {
    document.getElementById("confirmBox").style.display = "none";
    document.getElementById("modeSelection").style.display = "block";
}

function editInput() {
    document.getElementById("confirmBox").style.display = "none";
}

function selectAuto() {
    document.getElementById("warningBox").style.display = "block";
}

async function confirmAuto() {
    document.getElementById("warningBox").style.display = "none";
    document.getElementById("modeSelection").style.display = "none";
    
    let res = await fetch(`${API_BASE}/auto-train`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({task: userPrompt})
    });

    let data = await res.json();
    let jobId = data.job_id;

    if (jobId) {
        alert("🤖 Job queued! ID: " + jobId + "\nWe will poll for the result...");
        pollJobStatus(jobId);
    } else {
        alert("❌ Error: Response did not include job_id. Data: " + JSON.stringify(data));
    }
}

async function pollJobStatus(jobId) {
    const statusDiv = document.getElementById("result");
    statusDiv.innerText = "⏳ Training in progress (ID: " + jobId + ")...";

    const interval = setInterval(async () => {
        try {
            let res = await fetch(`${API_BASE}/jobs/${jobId}`);
            let data = await res.json();

            if (data.status === "completed") {
                clearInterval(interval);
                statusDiv.innerHTML = "✅ <b>Training Complete!</b><br>Accuracy: " + data.message;
                alert("✅ Model trained successfully!");
            } else if (data.status === "failed") {
                clearInterval(interval);
                statusDiv.innerHTML = "❌ <b>Training Failed:</b> " + data.message;
                alert("❌ Training failed: " + data.message);
            } else {
                statusDiv.innerText = "⏳ Status: " + data.status + "...";
            }
        } catch (err) {
            console.error("Poll error:", err);
        }
    }, 2000);
}

function downloadModel() {
    window.open(`${API_BASE}/download_model`);
}

function selectManual() {
    document.getElementById("uploadSection").style.display = "block";
    document.getElementById("warningBox").style.display = "none";
}

async function trainManual() {
    const fileInput = document.getElementById("fileInput");
    const files = fileInput.files;

    if (files.length === 0) {
        alert("Please select at least one CSV file!");
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }
    formData.append("task", userPrompt);

    document.getElementById("uploadSection").style.display = "none";
    document.getElementById("modeSelection").style.display = "none";

    try {
        let res = await fetch(`${API_BASE}/train/manual`, {
            method: "POST",
            body: formData
        });

        let data = await res.json();
        let jobId = data.job_id;

        if (jobId) {
            alert("📁 Manual job queued! ID: " + jobId);
            pollJobStatus(jobId);
        } else {
            alert("❌ Error: " + JSON.stringify(data));
        }
    } catch (err) {
        console.error("Upload error:", err);
        alert("❌ Failed to upload dataset.");
    }
}

// Hook up the train button from index.html (ensure ID matches)
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("trainBtn");
    if (btn) {
        btn.onclick = trainManual;
    }
});