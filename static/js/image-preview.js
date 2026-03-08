const input = document.getElementById("image-upload");
const dropzone = document.getElementById("upload-dropzone");
const preview = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const previewPanel = preview ? preview.closest(".image-preview") : null;
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const uploadSubmit = document.getElementById("upload-submit");
const clearFile = document.getElementById("clear-file");
const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

if (!input || !dropzone || !preview || !previewPlaceholder || !previewPanel || !fileName || !fileSize || !uploadSubmit || !clearFile || !uploadForm || !uploadStatus) {
    console.warn("Upload UI elements not found.");
} else {

let activePreviewUrl = null;

function formatSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "-";
    }
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function clearPreviewUrl() {
    if (activePreviewUrl) {
        URL.revokeObjectURL(activePreviewUrl);
        activePreviewUrl = null;
    }
}

function setStatus(message, state = "idle") {
    uploadStatus.textContent = message;
    uploadStatus.dataset.state = state;
}

function resetUploadState() {
    clearPreviewUrl();
    input.value = "";
    preview.src = "";
    fileName.textContent = "No image selected";
    fileSize.textContent = "-";
    uploadSubmit.disabled = true;
    clearFile.disabled = false;
    uploadSubmit.textContent = "Upload Image";
    previewPlaceholder.style.display = "flex";
    dropzone.classList.remove("has-file");
    dropzone.classList.remove("is-uploading");
    previewPanel.classList.remove("has-image");
    setStatus("Select an image to enable upload.", "idle");
}

function validateFile(file) {
    if (!file) {
        return { valid: false, message: "No image selected." };
    }

    if (!file.type || !ALLOWED_TYPES.has(file.type)) {
        return { valid: false, message: "Unsupported format. Use JPG, PNG, or WEBP." };
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
        return { valid: false, message: "Image exceeds 10 MB. Please choose a smaller file." };
    }

    return { valid: true };
}

function applyFile(file) {
    const validation = validateFile(file);
    if (!validation.valid) {
        resetUploadState();
        setStatus(validation.message, "error");
        return;
    }

    clearPreviewUrl();
    activePreviewUrl = URL.createObjectURL(file);
    preview.src = activePreviewUrl;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    uploadSubmit.disabled = false;
    previewPlaceholder.style.display = "none";
    dropzone.classList.add("has-file");
    previewPanel.classList.add("has-image");
    setStatus("Image ready to upload.", "success");
}

function handleInputChange() {
    const file = input.files && input.files[0];
    applyFile(file);
}

dropzone.addEventListener("click", () => input.click());
dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
    }
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (event) => {
    const droppedFiles = event.dataTransfer && event.dataTransfer.files;
    if (!droppedFiles || droppedFiles.length === 0) {
        return;
    }

    input.files = droppedFiles;
    applyFile(droppedFiles[0]);
});

input.addEventListener("change", handleInputChange);
clearFile.addEventListener("click", resetUploadState);
uploadForm.addEventListener("submit", (event) => {
    const file = input.files && input.files[0];
    const validation = validateFile(file);

    if (!validation.valid) {
        event.preventDefault();
        setStatus(validation.message, "error");
        uploadSubmit.disabled = true;
        return;
    }

    uploadSubmit.disabled = true;
    clearFile.disabled = true;
    uploadSubmit.textContent = "Uploading...";
    dropzone.classList.add("is-uploading");
    setStatus("Uploading image for analysis...", "info");
});

window.addEventListener("beforeunload", clearPreviewUrl);
}
