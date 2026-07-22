// Replace the existing btnFace click handler in static/index.html with this.
// Captures a short burst instead of a single frame, so one unlucky
// auto-exposure frame can't fail the check on its own.

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function captureBurst(count = 5, intervalMs = 110) {
    const frames = [];
    for (let i = 0; i < count; i++) {
        frames.push(captureFrame());
        if (i < count - 1) await sleep(intervalMs);
    }
    return frames;
}

btnFace.addEventListener('click', async () => {
    btnFace.disabled = true;
    statusText.innerText = "Analyzing Face (burst capture)...";
    resultCanvas.style.display = 'none';

    const frames = await captureBurst();

    try {
        const response = await fetch('/v1/face-quality', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ face_frames: frames })
        });
        const data = await response.json();
        jsonOutput.innerText = JSON.stringify(data, null, 2);
        statusText.innerText = `Face check complete (${data.frames_analyzed ?? 1} frames analyzed).`;
    } catch (err) {
        jsonOutput.innerText = "Error: " + err;
    }
    btnFace.disabled = false;
});
