// State
let counters = {
    raw: 0,
    debounce: 0,
    throttle: 0,
    ahf: 0
};

let logs = [];
let chart;
let startTime = Date.now();

// --- Algorithms ---

// 1. Standard Debounce
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// 2. Standard Throttle
function throttle(func, limit) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 3. Adaptive Hybrid Filter (AHF) - From Paper
class AdaptiveHybridFilter {
    constructor(callback, config = { alpha: 0.3, beta: 0.1, lambdaInit: 0.005 }) {
        this.callback = callback;
        this.config = config;
        this.lambda = config.lambdaInit;
        this.lastEventTime = -Infinity;
        this.lastEmitTime = -Infinity;
        this.timer = null;
    }

    process(event) {
        const now = Date.now();

        // Update rate estimate (lambda)
        if (this.lastEventTime > 0) {
            const tau = now - this.lastEventTime;
            if (tau > 0) {
                // Exponential Moving Average for rate estimation
                // lambda is events/ms
                this.lambda = (1 - this.config.beta) * this.lambda + this.config.beta * (1 / tau);
            }
        }
        this.lastEventTime = now;

        // Update UI for lambda
        document.getElementById('ahfLambda').innerText = (this.lambda * 1000).toFixed(1) + " Hz";

        // Compute optimal parameters based on Theorem 3.1 & 3.2
        // Delta* = (1/lambda) * ln((1-alpha)/alpha)
        // T* = 1 / (2 * lambda)

        // Clamp lambda to avoid division by zero or extreme values
        const safeLambda = Math.max(this.lambda, 0.0001);

        const deltaOpt = (1 / safeLambda) * Math.log((1 - this.config.alpha) / this.config.alpha);
        const tOpt = 1 / (2 * safeLambda);

        // Hybrid Logic
        if (now - this.lastEmitTime < tOpt) {
            // Throttle behavior: too soon since last emit, just reset debounce timer
            if (this.timer) clearTimeout(this.timer);
            this.timer = setTimeout(() => {
                this.emit(event);
            }, deltaOpt);
        } else {
            // Debounce behavior
            if (this.timer) clearTimeout(this.timer);
            this.timer = setTimeout(() => {
                this.emit(event);
            }, deltaOpt);
        }
    }

    emit(event) {
        this.lastEmitTime = Date.now();
        this.callback(event);
    }
}

// --- Experiment Setup ---

const rawHandler = () => {
    counters.raw++;
    updateUI();
    addLog('Raw event');
};

const debounceHandler = debounce(() => {
    counters.debounce++;
    updateUI();
    addLog('Debounce trigger', 'orange');
}, 300);

const throttleHandler = throttle(() => {
    counters.throttle++;
    updateUI();
    addLog('Throttle trigger', 'blue');
}, 100);

const ahfFilter = new AdaptiveHybridFilter(() => {
    counters.ahf++;
    updateUI();
    addLog('AHF trigger', 'green');
});

const ahfHandler = (e) => ahfFilter.process(e);

// --- UI Logic ---

function updateUI() {
    document.getElementById('rawCount').innerText = counters.raw;
    document.getElementById('debounceCount').innerText = counters.debounce;
    document.getElementById('throttleCount').innerText = counters.throttle;
    document.getElementById('ahfCount').innerText = counters.ahf;

    // Calculate reductions
    const calcRed = (val) => counters.raw > 0 ? ((1 - val / counters.raw) * 100).toFixed(1) + '%' : '0%';
    document.getElementById('debounceRed').innerText = calcRed(counters.debounce);
    document.getElementById('throttleRed').innerText = calcRed(counters.throttle);
    document.getElementById('ahfRed').innerText = calcRed(counters.ahf);

    updateChart();
}

function addLog(msg, color = 'white') {
    const list = document.getElementById('logList');
    const li = document.createElement('li');
    li.style.color = color;
    li.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    list.prepend(li);
    if (list.children.length > 5) list.removeChild(list.lastChild);
}

// Chart.js Setup
function initChart() {
    const ctx = document.getElementById('eventChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Raw', 'Debounce (300ms)', 'Throttle (100ms)', 'AHF (Adaptive)'],
            datasets: [{
                label: 'Event Count',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    '#95a5a6',
                    '#e67e22',
                    '#3498db',
                    '#2ecc71'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function updateChart() {
    if (!chart) return;
    chart.data.datasets[0].data = [
        counters.raw,
        counters.debounce,
        counters.throttle,
        counters.ahf
    ];
    chart.update('none'); // 'none' mode for performance
}

// Event Listeners
const area = document.getElementById('interactionArea');
const input = document.getElementById('testInput');
const instruction = document.getElementById('instructionText');
const scenarioSelect = document.getElementById('scenarioSelect');
const resetBtn = document.getElementById('resetBtn');

function handleEvent(e) {
    rawHandler(e);
    debounceHandler(e);
    throttleHandler(e);
    ahfHandler(e);
}

// Scenario Switching
scenarioSelect.addEventListener('change', (e) => {
    if (e.target.value === 'mousemove') {
        input.style.display = 'none';
        instruction.style.display = 'block';
        area.onmousemove = handleEvent;
        input.onkeyup = null;
    } else {
        input.style.display = 'block';
        instruction.style.display = 'none';
        area.onmousemove = null;
        input.onkeyup = handleEvent;
    }
    reset();
});

// Initial Setup: Mousemove
area.onmousemove = handleEvent;

function reset() {
    counters = { raw: 0, debounce: 0, throttle: 0, ahf: 0 };
    logs = [];
    document.getElementById('logList').innerHTML = '';
    // Reset AHF internal state
    ahfFilter.lambda = 0.005;
    updateUI();
}

resetBtn.addEventListener('click', reset);

// Start
initChart();
