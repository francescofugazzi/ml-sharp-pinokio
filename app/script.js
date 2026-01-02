// 1. Manage Job Selection & Deletion
function selectJob(jobName) {
    // Update visual selection state
    document.querySelectorAll('.job-list-item').forEach(item => {
        item.classList.remove('selected');
    });
    const selectedItem = document.querySelector('[data-job="' + jobName.replace(/'/g, "\\'") + '"]');
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }

    // Find the textbox and set value
    const textboxContainer = document.getElementById('job_selector_input');
    if (textboxContainer) {
        const textbox = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input');
        if (textbox) {
            textbox.value = jobName;
            textbox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
}

// Global variable for deletion context
let pendingDeleteJob = null;
let pendingDeleteFileBtn = null; // Store the button to click after confirmation

function deleteJob(jobName) {
    pendingDeleteJob = jobName;
    pendingDeleteFileBtn = null;
    showDeleteModal('Job', jobName);
}

function showDeleteModal(type, name) {
    let modal = document.getElementById('delete-confirm-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'delete-confirm-modal';
        modal.className = 'delete-modal-overlay';
        modal.innerHTML = `
            <div class="delete-modal-box">
                <div class="delete-modal-icon">🗑️</div>
                <div class="delete-modal-title">Delete ${type}?</div>
                <div class="delete-modal-message" id="delete-modal-message"></div>
                <div class="delete-modal-buttons">
                    <button class="delete-modal-btn cancel" onclick="hideDeleteModal()">Cancel</button>
                    <button class="delete-modal-btn confirm" onclick="confirmDelete()">Delete</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Customize message based on type
    let msg = '';
    if (type === 'Job') {
        msg = 'This will permanently delete job "' + name + '" and all its files.';
    } else {
        msg = 'Are you sure you want to delete this file?';
    }

    document.getElementById('delete-modal-message').textContent = msg;
    // Update title logic if needed, but generic "Delete Type?" is okay or we update it:
    modal.querySelector('.delete-modal-title').textContent = `Delete ${type}?`;

    modal.classList.add('visible');
}

function hideDeleteModal() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) {
        modal.classList.remove('visible');
    }
    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

function confirmDelete() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('visible');

    // Handle Job Deletion
    if (pendingDeleteJob) {
        const deleteContainer = document.getElementById('job_delete_input');
        if (deleteContainer) {
            const textbox = deleteContainer.querySelector('textarea') || deleteContainer.querySelector('input');
            if (textbox) {
                textbox.value = pendingDeleteJob;
                textbox.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    }

    // Handle File Deletion
    if (pendingDeleteFileBtn) {
        // Extract filename from the row to send to Python
        const row = pendingDeleteFileBtn.closest('tr') || pendingDeleteFileBtn.closest('.file-preview-item');
        let fileName = '';
        if (row) {
            // Strategy: Iterate TDs to find the one with the filename
            const tds = row.querySelectorAll('td');
            if (tds.length > 0) {
                // It's a table row
                for (let i = 0; i < tds.length; i++) {
                    const td = tds[i];
                    // Skip if it contains the delete button
                    if (td.querySelector('button') || td.querySelector('.remove-button')) continue;
                    // Skip if it seems to be the download/size cell (heuristic)
                    const text = (td.innerText || td.textContent).trim();
                    if (!text) continue;
                    if (td.classList.contains('download')) continue;
                    // Check for size pattern (e.g. "1.2 MB", "500 KB")
                    // Use string check or simple regex 
                    // To avoid escaping hell: check if it ENDS with B and has numbers
                    if (/[\d\.]+\s*(KB|MB|GB|B)$/i.test(text)) continue;

                    // Found a candidate
                    fileName = text;
                    break;
                }
            }

            // Fallback if no specific TD found or not a table
            if (!fileName) {
                // Try to split row text and avoid size
                let fullText = (row.innerText || row.textContent).trim();
                // Safer split: remove CR then split by LF
                let parts = fullText.replace(/\r/g, '').split('\n');
                // Filter out parts that look like size
                for (let p of parts) {
                    p = p.trim();
                    if (p && !/[\d\.]+\s*(KB|MB|GB|B)$/i.test(p) && p !== '✕') {
                        fileName = p;
                        break;
                    }
                }
            }
        }

        if (fileName) {
            const fileInput = document.getElementById('file_delete_input');
            if (fileInput) {
                const textbox = fileInput.querySelector('textarea') || fileInput.querySelector('input');
                if (textbox) {
                    textbox.value = fileName;
                    textbox.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        }

        // Also click the UI button to clear it visually immediately (client-side)
        // We need to bypass our own interceptor
        pendingDeleteFileBtn.dataset.confirmed = 'true';
        pendingDeleteFileBtn.click();
    }

    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

// Function called by the Delete Job button via js parameter
function triggerDeleteCurrentJob() {
    const selectorContainer = document.getElementById('job_selector_input');
    if (selectorContainer) {
        const textbox = selectorContainer.querySelector('textarea') || selectorContainer.querySelector('input');
        if (textbox && textbox.value) {
            deleteJob(textbox.value);
        } else {
            alert('No job selected to delete.');
        }
    } else {
        console.error("Job selector input not found");
    }
}

// 2. Manage File List Buttons (Robust)

// Global click listener for file deletion confirmation
// We use capture phase (true) to intercept before Gradio/Svelte
document.addEventListener('click', function (e) {
    // Check if target is a remove button or inside one
    const btn = e.target.closest('button[aria-label="Remove this file"], button.label-clear-button');

    if (btn) {
        // Double check we are inside the file list
        if (document.getElementById('det_files_list') && document.getElementById('det_files_list').contains(btn)) {

            // If already confirmed programmatically, let it pass
            if (btn.dataset.confirmed === 'true') {
                btn.dataset.confirmed = 'false'; // reset
                return; // Allow default Gradio action
            }

            // Otherwise stop it and show modal
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation(); // Crucial to stop other listeners

            pendingDeleteFileBtn = btn;
            showDeleteModal('File', 'selected file'); // We could try to extract filename but generic is safer
        }
    }
}, true); // Capture phase!

// Polling loop to hide image delete buttons
// This is more robust than MutationObserver against Svelte re-renders
setInterval(() => {
    const listIds = ['det_files_list', 'new_result_files_list'];

    listIds.forEach(id => {
        const fileList = document.getElementById(id);
        if (!fileList) return;

        const removeBtns = fileList.querySelectorAll('button[aria-label="Remove this file"], button.label-clear-button');

        removeBtns.forEach(btn => {
            const row = btn.closest('tr') || btn.closest('.file-preview-item');
            if (row) {
                const text = (row.innerText || row.textContent).toLowerCase();
                const isImage = text.includes('.png') || text.includes('.jpg') || text.includes('.jpeg') || text.includes('.webp');

                // Logic: Hide if it's the New Job list (all files) OR if it's an image in the Detail list
                const shouldHide = (id === 'new_result_files_list') || isImage;

                if (shouldHide) {
                    // HIDE but keep layout space (Alignment Fix)
                    if (btn.style.visibility !== 'hidden') {
                        btn.style.visibility = 'hidden';
                        btn.style.opacity = '0';
                        btn.style.pointerEvents = 'none';
                        btn.style.display = '';
                    }
                } else {
                    // Ensure visible
                    if (btn.style.visibility === 'hidden') {
                        btn.style.visibility = 'visible';
                        btn.style.opacity = '1';
                        btn.style.pointerEvents = '';
                    }
                }
            }
        });
    });
}, 500); // Check every 500ms

// Keep the Event Listeners for explicit actions to trigger immediate Reset
function checkAndResetLibrary() {
    const resetBtn = document.getElementById('btn_lib_reset');
    if (resetBtn) resetBtn.click();
}

document.addEventListener('click', (e) => {
    const closeBtn = e.target.closest('button[aria-label="Close"]');
    if (closeBtn) checkAndResetLibrary();
}, true);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') checkAndResetLibrary();
});
