const API_BASE = window.location.origin;

function showTab(tabId) {
    console.log("Switching to tab:", tabId);
    
    // Toggle Sections
    const trainSection = document.getElementById('train-section');
    const labSection = document.getElementById('lab-section');
    
    if (trainSection) trainSection.style.display = tabId === 'train-section' ? 'block' : 'none';
    if (labSection) labSection.style.display = tabId === 'lab-section' ? 'block' : 'none';
    
    // Toggle Button Active State
    const trainBtn = document.getElementById('btn-train');
    const labBtn = document.getElementById('id-lab-btn');
    
    if (trainBtn) trainBtn.classList.toggle('active', tabId === 'train-section');
    if (labBtn) labBtn.classList.toggle('active', tabId === 'lab-section');

    if (tabId === 'lab-section') {
        refreshLabModels();
    }
}

async function refreshLabModels(modelToSelect = null) {
    console.log("Refreshing Lab Models...");
    try {
        const res = await fetch(`${API_BASE}/lab/models`);
        const models = await res.json();
        const select = document.getElementById('labModelSelect');
        if (!select) return;
        
        select.innerHTML = '<option value="">-- Select a .pkl --</option>';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === modelToSelect) opt.selected = true;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to refresh models:", err);
    }
}

window.onbeforeunload = function () {
    console.log("PAGE RELOADING");
};
let userPrompt = "";
let detectedTask = "";
let currentLabAnalysis = null;
let lastInjectedFilePath = null;

async function injectCustomData() {
    const fileInput = document.getElementById("labDataInjection");
    const file = fileInput.files[0];
    const alertBox = document.getElementById("labIntelligenceAlert");
    const alertMsg = document.getElementById("labAlertMessage");
    
    if (!file) {
        lastInjectedFilePath = null;
        alertBox.style.display = "none";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_classes", JSON.stringify(currentLabAnalysis.new_labels || []));
    formData.append("goal_description", userPrompt); // Pass the instruction for relevance check

    try {
        let res = await fetch(`${API_BASE}/lab/inject`, {
            method: "POST",
            body: formData
        });
        let data = await res.json();
        
        if (data.is_valid) {
            lastInjectedFilePath = data.file_path;
            alertBox.style.display = "block";
            
            if (data.warnings && data.warnings.length > 0) {
                const isCritical = data.warnings.some(w => w.includes("CRITICAL"));
                
                alertBox.style.background = isCritical ? "rgba(255, 0, 0, 0.1)" : "rgba(255, 165, 0, 0.15)";
                alertBox.style.borderColor = isCritical ? "red" : "orange";
                alertBox.style.color = isCritical ? "#ff4d4d" : "orange";

                let html = `<b>Intelligence Audit:</b><ul>` + 
                           data.warnings.map(w => `<li>${w}</li>`).join('') + 
                           `</ul>`;
                
                if (isCritical) {
                    html += `<div style="margin-top:10px; display:flex; gap:10px;">
                                <button onclick="discardInjection()" style="background:#444; font-size:0.75rem;">🤖 Discard & Use AI Data</button>
                                <button onclick="document.getElementById('labDataInjection').click()" style="background:#007bff; font-size:0.75rem;">📁 Upload New File</button>
                             </div>`;
                } else {
                    html += `<br><i>AI will automatically supplement the missing data.</i>`;
                }
                alertMsg.innerHTML = html;
            } else {
                alertBox.style.background = "rgba(0, 255, 0, 0.1)";
                alertBox.style.borderColor = "green";
                alertBox.style.color = "green";
                alertMsg.innerText = "✅ Data is 100% relevant and sufficient!";
            }
        } else {
            alert("❌ CSV Error: " + data.message);
            fileInput.value = "";
        }
    } catch (err) {
        console.error(err);
    }
}

function discardInjection() {
    document.getElementById("labDataInjection").value = "";
    document.getElementById("labIntelligenceAlert").style.display = "none";
    lastInjectedFilePath = null;
    alert("Discarded. System will now use its internal Knowledge Base.");
}

async function viewLineage() {
    const modelName = document.getElementById("labModelSelect").value;
    if (!modelName) return alert("Please select a model first.");

    const modal = document.getElementById("lineageModal");
    const content = document.getElementById("lineageContent");
    modal.style.display = "flex";
    content.innerHTML = "📡 Loading history...";

    try {
        let res = await fetch(`${API_BASE}/lab/lineage/${modelName}`);
        let history = await res.json();

        if (history.length === 0) {
            content.innerHTML = `<p style="color:#888;">No version history found for this model yet.</p>`;
            return;
        }

        content.innerHTML = history.map((h, i) => `
            <div style="margin-bottom: 15px; padding: 12px; background: rgba(255,255,255,0.05); border-left: 4px solid #007bff; border-radius: 4px;">
                <div style="display:flex; justify-content:space-between;">
                    <b style="color:#007bff;">Version ${h.version}</b>
                    <span style="font-size:0.8rem; color:#888;">${h.id.substring(0,8)}</span>
                </div>
                <p style="font-size:0.9rem; margin: 8px 0;">"${h.instruction}"</p>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:5px;">
                    <span style="font-size:0.8rem; color: #28a745;">🎯 Accuracy: ${h.accuracy || 'N/A'}</span>
                    <button onclick="window.location.href='${API_BASE}/download_model?job_id=${h.id}'" style="width:auto; padding: 3px 10px; font-size:0.75rem;">📥 Download</button>
                </div>
                ${i < history.length - 1 ? '<div style="text-align:center; margin-top:10px; color:#444;">↓</div>' : ''}
            </div>
        `).join('');

    } catch (err) {
        content.innerHTML = "❌ Error fetching lineage.";
    }
}

function closeLineage() {
    document.getElementById("lineageModal").style.display = "none";
}

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

let currentTrainModelName = null;

async function pollJobStatus(jobId) {
    const statusDiv = document.getElementById("result");
    statusDiv.innerText = "⏳ Training in progress (ID: " + jobId + ")...";

    const interval = setInterval(async () => {
        try {
            let res = await fetch(`${API_BASE}/jobs/${jobId}`);
            let data = await res.json();

            if (data.status === "completed") {
                clearInterval(interval);
                statusDiv.innerHTML = "✅ <b>Training Complete!</b><br>" + data.message + 
                                     `<br><button class='secondary' onclick="downloadModel('${jobId}')" style='margin-top:10px;'>📥 Download .pkl Model</button>`;
                
                // Show testing playground
                document.getElementById("trainTestingSection").style.display = "block";
                
                // Extract model name from path
                if (data.model_path) {
                    currentTrainModelName = data.model_path.split('/').pop().split('\\').pop();
                    
                    // Toggle file/text input based on modality
                    if (data.intent && data.intent.modality === "image") {
                        document.getElementById("trainTestTextInput").style.display = "none";
                        document.getElementById("trainTestFileInput").style.display = "block";
                    } else {
                        document.getElementById("trainTestTextInput").style.display = "block";
                        document.getElementById("trainTestFileInput").style.display = "none";
                    }
                }
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

function downloadModel(jobId) {
    let url = `${API_BASE}/download_model`;
    if (jobId) {
        url += `?job_id=${jobId}`;
    }
    window.open(url);
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

// --- TAB MANAGEMENT ---
function showTab(tabId) {
    document.querySelectorAll('.container').forEach(c => c.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    document.getElementById(tabId).style.display = 'block';
    event.currentTarget.classList.add('active');
}

// --- MODEL LAB LOGIC ---

async function refreshLabModels() {
    const select = document.getElementById("labModelSelect");
    try {
        let res = await fetch(`${API_BASE}/lab/models`);
        let models = await res.json();
        
        select.innerHTML = '<option value="">-- Select a .pkl --</option>';
        models.forEach(m => {
            let option = document.createElement("option");
            option.value = m;
            option.text = m;
            select.appendChild(option);
        });
    } catch (err) {
        console.error("Failed to fetch models", err);
    }
}

async function uploadLabModel() {
    const fileInput = document.getElementById("labModelUpload");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const resultDiv = document.getElementById("labResult");
    resultDiv.innerHTML = "📤 Uploading external model...";

    try {
        let res = await fetch(`${API_BASE}/lab/upload`, {
            method: "POST",
            body: formData
        });
        let data = await res.json();
        alert(data.message);
        await refreshLabModels();
        document.getElementById("labModelSelect").value = data.filename;
        resultDiv.innerHTML = `✅ Model ${data.filename} ready for analysis.`;
    } catch (err) {
        console.error("Upload error", err);
        alert("❌ Failed to upload model.");
    }
}

async function analyzeLab() {
    const modelName = document.getElementById("labModelSelect").value;
    const instruction = document.getElementById("labInstruction").value;
    const resultDiv = document.getElementById("labResult");

    if (!modelName || !instruction) {
        alert("Please select a model and provide an instruction.");
        return;
    }

    resultDiv.innerHTML = "🧪 Analyzing model metadata & instruction...";
    
    try {
        let res = await fetch(`${API_BASE}/lab/analyze`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                model_name: modelName,
                instruction: instruction
            })
        });
        
        currentLabAnalysis = await res.json();
        
        document.getElementById("labActionText").innerHTML = 
            `<b>Action:</b> ${currentLabAnalysis.action}<br>` +
            `<b>Strategy:</b> ${currentLabAnalysis.message}<br>` +
            `<b>Classes to Add:</b> ${currentLabAnalysis.new_labels?.join(', ') || 'None'}<br>` +
            `<b>Classes to Remove:</b> ${currentLabAnalysis.remove_labels?.join(', ') || 'None'}`;
        
        document.getElementById("labProposal").style.display = "block";
        resultDiv.innerHTML = "✅ Analysis complete. Ready to execute.";
    } catch (err) {
        resultDiv.innerHTML = "❌ Failed to analyze instruction.";
    }
}

async function executeLab() {
    const modelName = document.getElementById("labModelSelect").value;
    const instruction = document.getElementById("labInstruction").value;
    const resultDiv = document.getElementById("labResult");

    document.getElementById("labProposal").style.display = "none";
    resultDiv.innerHTML = "⚡ Initiating Deep Manipulation (Knowledge Distillation)...";

    try {
        let res = await fetch(`${API_BASE}/lab/execute`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                model_name: modelName,
                instruction: instruction,
                analysis: currentLabAnalysis,
                injected_file_path: lastInjectedFilePath,
                auto_fill_gaps: document.getElementById("labAutoFillGaps").checked
            })
        });
        
        let data = await res.json();
        if (data.job_id) {
            pollLabStatus(data.job_id);
        }
    } catch (err) {
        resultDiv.innerHTML = "❌ Execution failed.";
    }
}

async function sendLabFeedback() {
    const feedback = document.getElementById("labChatInput").value;
    const modelName = document.getElementById("labModelSelect").value;
    const resultDiv = document.getElementById("labResult");

    if (!feedback) return;

    resultDiv.innerHTML = "💬 Refining requirements...";
    
    try {
        let res = await fetch(`${API_BASE}/lab/chat`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                model_name: modelName,
                feedback: feedback,
                previous_analysis: currentLabAnalysis,
                history: [] 
            })
        });
        
        currentLabAnalysis = await res.json();
        
        // Update the display with the new analysis
        document.getElementById("labActionText").innerHTML = 
            `<b>Action:</b> ${currentLabAnalysis.action}<br>` +
            `<b>Strategy:</b> ${currentLabAnalysis.message}<br>` +
            `<b>Classes to Add:</b> ${currentLabAnalysis.new_labels?.join(', ') || 'None'}<br>` +
            `<b>Classes to Remove:</b> ${currentLabAnalysis.remove_labels?.join(', ') || 'None'}`;
        
        document.getElementById("labChatInput").value = "";
        resultDiv.innerHTML = "✅ Proposal refined.";
    } catch (err) {
        console.error("Refine error", err);
        resultDiv.innerHTML = "❌ Failed to refine proposal.";
    }
}

function pollLabStatus(jobId) {
    const resultDiv = document.getElementById("labResult");
    const interval = setInterval(async () => {
        try {
            let res = await fetch(`${API_BASE}/jobs/${jobId}`);
            let data = await res.json();
            
            if (data.status === "completed") {
                clearInterval(interval);
                let analyticsHtml = "";
                if (data.accuracy && data.accuracy.overall) {
                    const stats = data.accuracy;
                    analyticsHtml = `
                        <div style="margin-top:15px; padding:12px; background:rgba(0,0,0,0.2); border-radius:8px; font-size:0.85rem; text-align:left;">
                            <p style="color:#007bff; font-weight:bold; margin-bottom:10px;">📊 Performance Comparison:</p>
                            <table style="width:100%; border-collapse:collapse;">
                                <tr style="border-bottom:1px solid #333;">
                                    <th style="text-align:left;">Model</th>
                                    <th style="text-align:left;">Accuracy</th>
                                </tr>
                                <tr>
                                    <td>Parent</td>
                                    <td>${(stats.overall.parent_accuracy * 100).toFixed(1)}%</td>
                                </tr>
                                <tr style="color:#28a745; font-weight:bold;">
                                    <td>Student</td>
                                    <td>${(stats.overall.student_accuracy * 100).toFixed(1)}%</td>
                                </tr>
                            </table>
                        </div>`;
                }
                
                resultDiv.innerHTML = `
                    🏁 <b>Manipulation Complete!</b><br>
                    ${data.message}<br>
                    ${analyticsHtml}
                    <br>
                    <div style="display: flex; gap: 10px;">
                        <button class="primary" style="flex:1;" onclick="window.location.href='${API_BASE}/lab/download_model?job_id=${jobId}'">
                            📥 Download Model (.pkl)
                        </button>
                        <button class="secondary" style="flex:1;" onclick="window.location.href='${API_BASE}/lab/download_report/${jobId}'">
                            📑 Download Audit Report
                        </button>
                    </div>
                    <p style="font-size: 0.8rem; color: #888; margin-top: 10px;">The model has been saved to your /models folder as well.</p>
                `;
                
                // Show Live Test Lab
                document.getElementById('liveTestArea').style.display = 'block';
                currentJobId = jobId; // Store for testing
                
                alert("Deep Manipulation Successful!");
                refreshLabModels();
            } else if (data.status === "failed") {
                clearInterval(interval);
                resultDiv.innerHTML = `❌ <b>Failed:</b> ${data.message}`;
            } else {
                resultDiv.innerText = `⚡ Status: ${data.status}...`;
            }
        } catch (err) { console.error(err); }
    }, 2000);
}

async function uploadLabModel() {
    const fileInput = document.getElementById('labModelUpload');
    const resultDiv = document.getElementById('labResult');
    const statusBadge = document.getElementById('labUploadStatus');
    
    if (fileInput.files.length === 0) return;

    // Step 1: Initialize Intake
    statusBadge.style.display = 'block';
    statusBadge.innerHTML = `📦 Intake: <b>${fileInput.files[0].name}</b>`;
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `📡 <b>Step 1/3:</b> Preparing data...`;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        resultDiv.innerHTML = `🚀 <b>Step 2/3:</b> Transmitting to Intelligence Lab...`;
        
        // Using relative path for maximum stability
        let res = await fetch(`/lab/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || "Server rejected the model");
        }
        
        resultDiv.innerHTML = `🧠 <b>Step 3/3:</b> Processing intelligence...`;
        let data = await res.json();
        
        resultDiv.innerHTML = `✅ <b>Final:</b> ${data.message}`;
        statusBadge.innerHTML = `✅ <b>Stored:</b> ${fileInput.files[0].name}`;
        statusBadge.style.borderColor = "#28a745";
        statusBadge.style.animation = "none";
        
        refreshLabModels(fileInput.files[0].name);
    } catch (err) {
        resultDiv.innerHTML = `❌ <b>Transmission Error:</b> ${err.message}`;
        statusBadge.innerHTML = `❌ <b>Failed:</b> ${fileInput.files[0].name}`;
        statusBadge.style.background = "rgba(220, 53, 69, 0.1)";
        statusBadge.style.borderColor = "#dc3545";
        statusBadge.style.animation = "none";
    }
}

async function analyzeLab() {
    let model = document.getElementById('labModelSelect').value;
    const instruction = document.getElementById('labInstruction').value;
    const proposalDiv = document.getElementById('labProposal');
    const actionText = document.getElementById('labActionText');
    const statusBadge = document.getElementById('labUploadStatus');

    // Fallback: If dropdown is empty but they just uploaded a file, use that file
    if (!model && statusBadge.innerHTML.includes("Stored:")) {
        const fileInput = document.getElementById('labModelUpload');
        if (fileInput.files.length > 0) {
            model = fileInput.files[0].name;
            // Also update the dropdown visually so they see it
            const select = document.getElementById('labModelSelect');
            if (select) select.value = model;
        }
    }

    if (!model || !instruction) return alert("Please select a model and describe the modification.");

    proposalDiv.style.display = 'block';
    actionText.innerHTML = "📡 AI is interpreting manipulation plan...";

    try {
        let res = await fetch(`${API_BASE}/lab/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: model, instruction: instruction })
        });
        currentLabAnalysis = await res.json();
        
        actionText.innerHTML = `
            <div style="color: #60a5fa; font-weight: bold; margin-bottom: 5px;">PROPOSED ACTION: ${currentLabAnalysis.action}</div>
            <div style="font-size: 0.9rem; color: #ccc;">${currentLabAnalysis.reasoning}</div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">New Targets: ${currentLabAnalysis.new_labels.join(', ')}</div>
        `;
    } catch (err) {
        actionText.innerHTML = `<span style="color:red;">Error: ${err.message}</span>`;
    }
}

async function sendLabFeedback() {
    const feedback = document.getElementById('labChatInput').value;
    const actionText = document.getElementById('labActionText');
    if (!feedback) return;

    actionText.innerHTML = "🔄 Refining plan based on feedback...";
    try {
        let res = await fetch(`${API_BASE}/lab/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_name: document.getElementById('labModelSelect').value,
                instruction: document.getElementById('labInstruction').value,
                feedback: feedback,
                previous_analysis: currentLabAnalysis
            })
        });
        currentLabAnalysis = await res.json();
        
        actionText.innerHTML = `
            <div style="color: #a78bfa; font-weight: bold; margin-bottom: 5px;">REFINED ACTION: ${currentLabAnalysis.action}</div>
            <div style="font-size: 0.9rem; color: #ccc;">${currentLabAnalysis.reasoning}</div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">Updated Targets: ${currentLabAnalysis.new_labels.join(', ')}</div>
        `;
    } catch (err) { console.error(err); }
}

async function injectCustomData() {
    const file = document.getElementById('labDataInjection').files[0];
    const alertBox = document.getElementById('labIntelligenceAlert');
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_classes", JSON.stringify(currentLabAnalysis.new_labels || []));
    formData.append("goal_description", document.getElementById('labInstruction').value);

    alertBox.style.display = "block";
    alertBox.className = "card intelligence-alert";
    alertBox.innerHTML = "🛡️ Auditing Data Relevance...";

    try {
        let res = await fetch(`${API_BASE}/lab/inject`, {
            method: "POST",
            body: formData
        });
        let data = await res.json();
        
        if (data.status === "REJECTED" || data.status === "CRITICAL") {
            alertBox.style.background = "rgba(220, 53, 69, 0.2)";
            alertBox.style.borderColor = "#dc3545";
            alertBox.innerHTML = `
                ❌ <b>RELEVANCE GUARD:</b> ${data.message}
                <br><br>
                <div style="display: flex; gap: 10px;">
                    <button class="primary" onclick="discardInjection()">🤖 Discard & Use AI Data</button>
                    <button class="secondary" onclick="document.getElementById('labDataInjection').click()">📁 Upload New File</button>
                </div>
            `;
        } else {
            alertBox.style.background = "rgba(40, 167, 69, 0.2)";
            alertBox.style.borderColor = "#28a745";
            alertBox.innerHTML = `✅ <b>DATA VERIFIED:</b> ${data.message}`;
        }
    } catch (err) { console.error(err); }
}

async function executeLab() {
    const model = document.getElementById('labModelSelect').value;
    const resultDiv = document.getElementById('labResult');
    const autoFill = document.getElementById('labAutoFillGaps').checked;

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = "⚡ Initializing Deep Manipulation Engine...";

    try {
        let res = await fetch(`${API_BASE}/lab/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: model,
                instruction: document.getElementById('labInstruction').value,
                analysis: currentLabAnalysis,
                auto_fill_gaps: autoFill
            })
        });
        let data = await res.json();
        pollLabStatus(data.job_id);
    } catch (err) {
        resultDiv.innerHTML = `<span style="color:red;">Execution failed: ${err.message}</span>`;
    }
}

async function testLabModel() {
    let model = document.getElementById('labModelSelect').value;
    const text = document.getElementById('labTestInput').value;
    const resultBox = document.getElementById('labTestResult');
    const predText = document.getElementById('testPredictionText');
    const confText = document.getElementById('testConfidenceText');
    const probBox = document.getElementById('testProbabilities');
    const statusBadge = document.getElementById('labUploadStatus');

    if (!model && statusBadge.innerHTML.includes("Stored:")) {
        const fileInput = document.getElementById('labModelUpload');
        if (fileInput.files.length > 0) {
            model = fileInput.files[0].name;
            const select = document.getElementById('labModelSelect');
            if (select) select.value = model;
        }
    }

    if (!model) return alert("Please select a model to test.");
    if (!text) return alert("Please enter some text to test.");

    resultBox.style.display = "block";
    predText.innerHTML = "⏳ Predicting...";
    confText.innerHTML = "--";
    probBox.innerHTML = "";

    try {
        const formData = new FormData();
        formData.append("model_name", model);
        
        const fileInput = document.getElementById('labTestFileInput');
        if (fileInput.files.length > 0) {
            formData.append("file", fileInput.files[0]);
        } else {
            formData.append("text", text);
        }

        let res = await fetch(`${API_BASE}/lab/predict`, {
            method: 'POST',
            body: formData
        });
        
        let data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        predText.innerHTML = data.prediction;
        confText.innerHTML = (data.confidence * 100).toFixed(1);
        
        let probHtml = "<b>All Probabilities:</b><br>";
        for (const [cls, prob] of Object.entries(data.all_probs || {})) {
            probHtml += `${cls}: ${(prob * 100).toFixed(1)}%<br>`;
        }
        probBox.innerHTML = probHtml;
    } catch (err) {
        predText.innerHTML = `<span style="color:red;">Error: ${err.message}</span>`;
        confText.innerHTML = "--";
    }
}

async function testTrainModel() {
    if (!currentTrainModelName) return alert("No trained model found.");
    
    const textInput = document.getElementById('trainTestTextInput').value;
    const fileInput = document.getElementById('trainTestFileInput');
    const resultBox = document.getElementById('trainTestResult');
    const predText = document.getElementById('trainTestPredictionText');
    const confText = document.getElementById('trainTestConfidenceText');
    const probBox = document.getElementById('trainTestProbabilities');

    resultBox.style.display = "block";
    predText.innerHTML = "⏳ Predicting...";
    confText.innerHTML = "--";
    probBox.innerHTML = "";

    try {
        const formData = new FormData();
        formData.append("model_name", currentTrainModelName);
        
        if (fileInput.files.length > 0) {
            formData.append("file", fileInput.files[0]);
        } else {
            formData.append("text", textInput);
        }

        let res = await fetch(`${API_BASE}/lab/predict`, {
            method: 'POST',
            body: formData
        });
        
        let data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        predText.innerHTML = data.prediction;
        confText.innerHTML = (data.confidence * 100).toFixed(1);
        
        let probHtml = "<b>All Probabilities:</b><br>";
        for (const [cls, prob] of Object.entries(data.all_probs || {})) {
            probHtml += `${cls}: ${(prob * 100).toFixed(1)}%<br>`;
        }
        probBox.innerHTML = probHtml;
    } catch (err) {
        predText.innerHTML = `<span style="color:red;">Error: ${err.message}</span>`;
        confText.innerHTML = "--";
    }
}

async function refineTrainModel() {
    if (!currentTrainModelName) return alert("No model to refine.");
    
    const instruction = document.getElementById('trainRefineInstruction').value;
    if (!instruction) return alert("Please tell the AI what is wrong with the model.");
    
    const autoFill = document.getElementById('trainRefineAutoFill').checked;
    const fileInput = document.getElementById('trainRefineData');
    
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "🧠 AI is analyzing your refinement request...";
    
    // Hide testing sections during training
    document.getElementById("trainTestingSection").style.display = "none";
    
    try {
        // 1. Analyze what needs to be done
        let res = await fetch(`${API_BASE}/lab/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: currentTrainModelName, instruction: instruction })
        });
        let analysis = await res.json();
        
        // 2. Upload file if needed
        let injected_path = null;
        if (fileInput.files.length > 0) {
            resultDiv.innerHTML = "📤 Uploading correct data...";
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            formData.append("target_classes", JSON.stringify(analysis.new_labels || []));
            formData.append("goal_description", instruction);
            
            let injectRes = await fetch(`${API_BASE}/lab/inject`, { method: "POST", body: formData });
            let injectData = await injectRes.json();
            injected_path = injectData.path || null;
        }
        
        // 3. Execute
        resultDiv.innerHTML = "🚀 Dispatching Deep Manipulation Engine...";
        let execRes = await fetch(`${API_BASE}/lab/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: currentTrainModelName,
                instruction: instruction,
                analysis: analysis,
                auto_fill_gaps: autoFill,
                injected_file_path: injected_path
            })
        });
        
        let execData = await execRes.json();
        
        // 4. Poll
        pollJobStatus(execData.job_id);
        
    } catch (err) {
        resultDiv.innerHTML = `<span style="color:red;">Refinement failed: ${err.message}</span>`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyzeLabBtn");
    if (analyzeBtn) analyzeBtn.onclick = analyzeLab;

    const executeBtn = document.getElementById("executeLabBtn");
    if (executeBtn) executeBtn.onclick = executeLab;

    const injectInput = document.getElementById("labDataInjection");
    if (injectInput) injectInput.onchange = injectCustomData;

    const btn = document.getElementById("trainBtn");
    if (btn) btn.onclick = trainManual;
});