// File upload handling
const fileUpload = document.getElementById('fileUpload');
const fileInfo = document.getElementById('fileInfo');
const questionsTextarea = document.getElementById('questions');
const questionCount = document.getElementById('questionCount');
const qaForm = document.getElementById('qaForm');
const submitBtn = document.getElementById('submitBtn');
const resultsSection = document.getElementById('results');
const answersContainer = document.getElementById('answersContainer');
const errorSection = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

// Update file info display
fileUpload.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const fileSizeKB = (file.size / 1024).toFixed(2);
        fileInfo.innerHTML = `
            <div class="file-name">
                <span>✅</span>
                <span>${file.name}</span>
            </div>
            <div class="file-size">${fileSizeKB} KB</div>
        `;
    } else {
        fileInfo.innerHTML = '<span class="placeholder">Choose a PDF, DOCX, or TXT file</span>';
    }
});

// Update question count
questionsTextarea.addEventListener('input', function(e) {
    const questions = e.target.value.split('\n').filter(q => q.trim() !== '');
    const count = questions.length;
    questionCount.textContent = `${count} question${count !== 1 ? 's' : ''}`;
});

// Form submission
qaForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Hide previous results/errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loading').style.display = 'inline';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', fileUpload.files[0]);
    formData.append('questions', questionsTextarea.value);
    
    try {
        const response = await fetch('/api/v1/hackrx/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data.answers);
        }
    } catch (error) {
        showError('An error occurred while processing your request. Please try again.');
        console.error('Error:', error);
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline';
        submitBtn.querySelector('.btn-loading').style.display = 'none';
    }
});

// Display results
function displayResults(answers) {
    answersContainer.innerHTML = '';
    
    answers.forEach((item, index) => {
        const answerCard = document.createElement('div');
        answerCard.className = 'answer-card';
        answerCard.innerHTML = `
            <div class="answer-question">
                Q${index + 1}: ${escapeHtml(item.question)}
            </div>
            <div class="answer-text">${formatAnswer(item.answer)}</div>
        `;
        answersContainer.appendChild(answerCard);
    });
    
    resultsSection.style.display = 'block';
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Format answer with markdown-like support
function formatAnswer(text) {
    // Convert markdown-like bold to HTML
    text = text.replace(/\*\*(.+?)\*\*/g, '$1');
    
    // Convert emojis and preserve line breaks
    text = escapeHtml(text);
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-resize textarea
questionsTextarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});
