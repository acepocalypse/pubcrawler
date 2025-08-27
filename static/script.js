// --- Global Configuration & State ---
const SOURCE_META = {
    google_scholar: { key: 'google_scholar', name: 'GS', fullName: 'Google Scholar', color: '#4286f5', icon: 'fab fa-google' },
    scopus: { key: 'scopus', name: 'Scopus', fullName: 'Scopus', color: '#eb6601', icon: 'fas fa-database' },
    wos: { key: 'wos', name: 'WoS', fullName: 'Web of Science', color: '#5e33c0', icon: 'fas fa-globe' },
    orcid: { key: 'orcid', name: 'ORCID', fullName: 'ORCID', color: '#a5cd3b', icon: 'fab fa-orcid' }
};

let currentSearchResults = null;
let originalPublications = [];
let filteredPublications = [];
let discoveredProfiles = null;
let selectedProfiles = {};
let sourceFilterState = { google_scholar: 0, scopus: 0, wos: 0, orcid: 0 }; // 0: any, 1: include, -1: exclude

// --- DOM Element References ---
const searchForm = document.getElementById('search-form');
const searchBtn = document.getElementById('search-btn');
const searchBtnText = document.getElementById('search-btn-text');
const searchSpinner = document.getElementById('search-spinner');
const searchProgress = document.getElementById('search-progress');
const searchProgressBar = document.getElementById('search-progress-bar');

const autofillBtn = document.getElementById('autofill-btn');
const autofillBtnText = document.getElementById('autofill-btn-text');
const autofillSpinner = document.getElementById('autofill-spinner');
const autofillProgress = document.getElementById('autofill-progress');
const autofillProgressBar = document.getElementById('autofill-progress-bar');

const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');

const profileModal = document.getElementById('profile-modal');
const closeModal = document.getElementById('close-modal');
const cancelSelection = document.getElementById('cancel-selection');
const applySelection = document.getElementById('apply-selection');
const selectAllBtn = document.getElementById('select-all-btn');
const discoveryResults = document.getElementById('discovery-results');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', initializeEventListeners);

function initializeEventListeners() {
    searchForm.addEventListener('submit', handleSearch);
    autofillBtn.addEventListener('click', handleAutofill);

    // UI Toggles & Modals
    document.getElementById('api-keys-toggle').addEventListener('click', toggleApiKeysSection);
    closeModal.addEventListener('click', closeProfileModal);
    cancelSelection.addEventListener('click', closeProfileModal);
    applySelection.addEventListener('click', applySelectedProfiles);
    selectAllBtn.addEventListener('click', selectAllProfiles);
    profileModal.addEventListener('click', (e) => {
        if (e.target === profileModal) closeProfileModal();
    });

    // Page Actions
    document.getElementById('export-csv').addEventListener('click', exportToCSV);
    document.getElementById('print-btn').addEventListener('click', () => window.print());
    document.getElementById('recent-searches').addEventListener('click', showRecentSearches);
    document.getElementById('settings-btn').addEventListener('click', showSettings);
}

function initializeFilterEventListeners() {
    document.getElementById('search-filter').addEventListener('input', applyFilters);
    document.getElementById('year-min').addEventListener('input', applyFilters);
    document.getElementById('year-max').addEventListener('input', applyFilters);
    document.getElementById('source-filter').addEventListener('change', applyFilters);
    document.getElementById('sort-filter').addEventListener('change', applyFilters);
    document.getElementById('clear-filters').addEventListener('click', clearFilters);
    initSourceIconFilters();
}

// --- Core Application Logic ---
async function handleSearch(event) {
    event.preventDefault();
    const firstName = document.getElementById('first-name').value.trim();
    const lastName = document.getElementById('last-name').value.trim();
    if (!firstName || !lastName) {
        showError('Please enter both first and last name');
        return;
    }


    setLoadingState(true, 'search');
    hideError();
    hideResults();
    // Reset filters and clear table before new search
    clearFilters();
    const tableBody = document.getElementById('publications-table-body');
    if (tableBody) tableBody.innerHTML = '';

    try {
        // Split WoS IDs by comma, trim each, and filter out empty
        const wosRaw = document.getElementById('wos-id').value.trim();
        const wosIds = wosRaw.split(',').map(x => x.trim()).filter(Boolean);
        const searchData = {
            first_name: firstName,
            last_name: lastName,
            google_scholar_id: document.getElementById('google-scholar-id').value.trim(),
            scopus_id: document.getElementById('scopus-id').value.trim(),
            wos_id: wosIds.length > 1 ? wosIds : wosRaw,
            orcid_id: document.getElementById('orcid-id').value.trim(),
            affiliation: document.getElementById('affiliation').value.trim(),
            api_keys: {
                scopus_api_key: document.getElementById('scopus-api-key').value.trim(),
                wos_api_key: document.getElementById('wos-api-key').value.trim(),
                orcid_client_id: document.getElementById('orcid-client-id').value.trim(),
                orcid_client_secret: document.getElementById('orcid-client-secret').value.trim()
            }
        };
        const searchResp = await fetch('/api/search/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchData)
        });
        const searchJson = await searchResp.json();
        if (!searchResp.ok) throw new Error(searchJson.error || 'Failed to start search');

        const jobResult = await pollJob(searchJson.job_id, 'search');
        currentSearchResults = jobResult;
        displayResults(jobResult);
    } catch (error) {
        console.error('Search error:', error);
        showError(error.message || 'An error occurred during search');
    } finally {
        setLoadingState(false, 'search');
    }
}

async function handleAutofill() {
    clearIdInputs();
    const firstName = document.getElementById('first-name').value.trim();
    const lastName = document.getElementById('last-name').value.trim();
    if (!firstName || !lastName) {
        showError('Please enter both first and last name before using autofill');
        return;
    }

    setLoadingState(true, 'autofill');
    hideError();

    try {
        const requestData = {
            first_name: firstName,
            last_name: lastName,
            affiliation: document.getElementById('affiliation').value.trim(),
            api_keys: {
                scopus_api_key: document.getElementById('scopus-api-key').value.trim(),
                wos_api_key: document.getElementById('wos-api-key').value.trim(),
                orcid_client_id: document.getElementById('orcid-client-id').value.trim(),
                orcid_client_secret: document.getElementById('orcid-client-secret').value.trim()
            }
        };
        const startResp = await fetch('/api/discover-profiles/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        const startJson = await startResp.json();
        if (!startResp.ok) throw new Error(startJson.error || 'Failed to start profile discovery');

        const jobResult = await pollJob(startJson.job_id, 'autofill');
        discoveredProfiles = jobResult;
        showProfileModal(jobResult);
    } catch (error) {
        console.error('Profile discovery error:', error);
        showError(error.message || 'Failed to discover profiles');
    } finally {
        setLoadingState(false, 'autofill');
    }
}

async function pollJob(jobId, kind) {
    const isSearch = kind === 'search';
    const bar = isSearch ? searchProgressBar : autofillProgressBar;
    const progressEl = isSearch ? searchProgress : autofillProgress;
    const textEl = progressEl.querySelector('[id$="-progress-text"]');

    while (true) {
        let resp;
        try {
            resp = await fetch(`/api/progress/${jobId}`);
        } catch (e) {
            throw new Error('Network error while checking progress');
        }
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Failed to check progress');
        
        if (typeof data.percent === 'number') {
            updateProgressBar(bar, data.percent);
            if (textEl && data.message) {
                textEl.textContent = `${data.message} (${Math.round(data.percent)}%)`;
            }
        } else if (data.message && textEl) {
            textEl.textContent = data.message;
        }

        if (data.done) {
            if (data.error) throw new Error(data.error);
            updateProgressBar(bar, 100);
            return data.result;
        }
        await new Promise(r => setTimeout(r, 700));
    }
}

// --- UI Display & Manipulation ---
function displayResults(data) {
    document.getElementById('researcher-display-name').textContent = formatName(data.researcher.name);
    
    const infoParts = [];
    if (data.researcher.affiliation) infoParts.push(`Affiliation: ${data.researcher.affiliation}`);
    if (data.researcher.gs_id) infoParts.push(`Google Scholar ID: ${data.researcher.gs_id}`);
    if (data.researcher.scopus_id) infoParts.push(`Scopus ID: ${data.researcher.scopus_id}`);
    if (data.researcher.wos_id) infoParts.push(`WoS ID: ${data.researcher.wos_id}`);
    if (data.researcher.orcid_id) infoParts.push(`ORCID ID: ${data.researcher.orcid_id}`);
    document.getElementById('researcher-info').textContent = infoParts.join(' • ');

    // Populate new summary cards
    const pubs = data.publications || [];
    const totalPubs = pubs.length;
    const avgCoverage = data.summary?.coverage_report?.average_coverage || 0;
    const fullyCovered = pubs.filter(p => p.coverage_count === 4).length;
    const gaps = totalPubs - fullyCovered;

    document.getElementById('summary-total-pubs').textContent = totalPubs;
    document.getElementById('summary-avg-coverage').textContent = `${Math.round(avgCoverage * 100)}%`;
    document.getElementById('summary-fully-covered').textContent = fullyCovered;
    document.getElementById('summary-gaps').textContent = gaps;

    // Populate old summary line (can be removed later if redundant)
    document.getElementById('total-publications').textContent = data.summary.total_publications;
    document.getElementById('sources-used').textContent = data.summary.sources_used.join(', ');
    document.getElementById('average-coverage').textContent = `${Math.round((data.summary.coverage_report?.average_coverage || 0) * 100)}%`;

    originalPublications = [...data.publications];
    filteredPublications = [...data.publications];

    initializeYearRange();
    document.getElementById('sort-filter').value = 'most-cited';
    
    initializeFilterEventListeners();
    applyFilters();

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createPublicationRow(pub) {
    const row = document.createElement('tr');
    row.className = 'publication-row';

    const authorsText = pub.authors ? pub.authors.slice(0, 4).join(', ') + (pub.authors.length > 4 ? `, et al.` : '') : 'No authors listed';
    const coverageCount = Object.values(pub.coverage).filter(Boolean).length;
    const normalizedDoi = normalizeDoiValue(pub.doi);
    
    const getSourceLink = (sourceFullName) => pub.links?.find(l => l.source === sourceFullName)?.url;

    const coverageHtml = `
        <div class="flex flex-col items-center space-y-2">
            <div class="flex justify-center space-x-2">
                ${Object.values(SOURCE_META).map(meta => {
                    const isActive = pub.coverage[meta.key];
                    const linkUrl = getSourceLink(meta.fullName);
                    const iconHtml = `<div class="database-icon ${isActive ? 'active' : ''} w-6 h-6 rounded-full flex items-center justify-center" style="background-color: ${isActive ? meta.color : '#e5e7eb'}" title="${meta.fullName}"><i class="${meta.icon} ${isActive ? 'text-white' : 'text-gray-400'} text-xs"></i></div>`;
                    return (linkUrl && isActive) ? `<a href="${linkUrl}" target="_blank" rel="noopener noreferrer">${iconHtml}</a>` : iconHtml;
                }).join('')}
            </div>
            <div class="text-xs ${coverageCount === 4 ? 'text-green-600 font-medium' : coverageCount >= 2 ? 'text-yellow-600' : 'text-red-600'}">${coverageCount}/4 databases</div>
        </div>`;

    const citationHtml = `<div class="citation-grid mx-auto">
        ${Object.values(SOURCE_META).map(source => {
            const citations = pub.source_citations?.[source.key];
            const isIndexed = source.key === 'orcid' ? pub.coverage?.orcid === true : citations != null;
            const count = source.key === 'orcid' ? 'N/A' : (isIndexed ? citations : 0);
            const boxStyle = isIndexed ? `background-color: ${source.color}; color: white; border: 2px solid ${source.color};` : `background-color: #f3f4f6; color: #9ca3af; border: 2px solid #e5e7eb;`;
            const titleText = `${source.fullName}: ${isIndexed ? (source.key === 'orcid' ? 'N/A' : `${count} citations`) : 'Not indexed'}`;
            const linkUrl = getSourceLink(source.fullName);
            const boxContent = `<div class="count">${count}</div><div class="label">${source.name}</div>`;
            return (linkUrl) ? `<a href="${linkUrl}" target="_blank" rel="noopener noreferrer" class="citation-box" style="${boxStyle}" title="${titleText}">${boxContent}</a>` : `<div class="citation-box" style="${boxStyle}" title="${titleText}">${boxContent}</div>`;
        }).join('')}
    </div>`;

    row.innerHTML = `
        <td data-label="Title & Authors" class="py-4 px-6">
            <div class="font-medium">${pub.title || 'Untitled'}</div>
            <div class="text-sm text-gray-600 mt-1">${authorsText}</div>
        </td>
        <td data-label="Year" class="py-4 px-6">${pub.year || 'N/A'}</td>
        <td data-label="Journal" class="py-4 px-6">${pub.journal || 'Unknown'}</td>
        <td data-label="DOI" class="py-4 px-6">${normalizedDoi ? `<a href="https://doi.org/${normalizedDoi}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline text-sm">${normalizedDoi}</a>` : '<span class="text-gray-400 text-sm">No DOI</span>'}</td>
        <td data-label="Citations by Database" class="py-4 px-6 citations-cell">${citationHtml}</td>
        <td data-label="Coverage" class="py-4 px-6">${coverageHtml}</td>
    `;
    return row;
}

function showProfileModal(discoveryResult) {
    selectedProfiles = {}; // Reset selections
    
    if (!discoveryResult.candidates?.length) {
        discoveryResults.innerHTML = `<div class="text-center py-10"><i class="fas fa-search text-gray-400 text-4xl mb-3"></i><h3 class="text-lg font-medium">No Profiles Found</h3><p class="text-gray-600 text-sm">No matching profiles for "${discoveryResult.query.first_name} ${discoveryResult.query.last_name}".</p></div>`;
    } else {
        const groups = Object.fromEntries(Object.keys(SOURCE_META).map(k => [k, []]));
        discoveryResult.candidates.forEach((c, idx) => {
            const key = normalizeSourceKey(c.source);
            if (groups[key]) groups[key].push({ candidate: c, gindex: idx });
        });

        const section = (key) => {
            const meta = SOURCE_META[key];
            const items = groups[key];
            const content = items.length ? items.map(({ candidate, gindex }) => createCandidateCard(candidate, gindex)).join('') : `<div class="text-sm text-gray-500 italic px-1">No ${meta.label} profiles found</div>`;
            return `<section class="border border-gray-200 rounded-xl overflow-hidden"><div class="source-section-header px-4 py-3 border-b flex items-center justify-between"><div class="flex items-center gap-2"><span class="w-7 h-7 rounded-full flex items-center justify-center" style="background-color:${meta.color}"><i class="${meta.icon} text-white text-xs"></i></span><h5 class="font-medium">${meta.fullName}</h5></div><span class="text-xs px-2 py-0.5 rounded-full bg-gray-200 text-gray-700">${items.length}</span></div><div class="p-3 space-y-3">${content}</div></section>`;
        };
        
        discoveryResults.innerHTML = `
            <div class="mb-4"><div class="split"><div><h4 class="text-lg font-medium">${discoveryResult.candidates.length} potential profiles for "${discoveryResult.query.first_name} ${discoveryResult.query.last_name}"</h4><p class="text-gray-600 text-sm mt-1">Review and select matching profiles.</p></div><div class="hidden sm:flex items-center justify-end gap-2"><span class="badge" style="background:#dcfce7;color:#166534">High match</span><span class="badge" style="background:#fef9c3;color:#854d0e">Medium</span><span class="badge" style="background:#fee2e2;color:#991b1b">Low</span></div></div></div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">${Object.keys(SOURCE_META).map(section).join('')}</div>`;
        
        discoveryResults.querySelectorAll('.candidate-select').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const gindex = parseInt(e.target.dataset.gindex, 10);
                const candidate = discoveryResult.candidates[gindex];
                if (!candidate) return;

                if (e.target.checked) selectedProfiles[candidate.source] = candidate;
                else delete selectedProfiles[candidate.source];

                e.target.closest('.candidate-card')?.classList.toggle('selected', e.target.checked);
                updateApplyButtonState();
            });
        });
    }
    profileModal.classList.remove('hidden');
    updateApplyButtonState();
}

function createCandidateCard(candidate, index) {
    const conf = candidate.confidence_score || 0;
    const confMeta = conf >= 0.8 ? { bg: '#dcfce7', fg: '#166534', label: `High • ${(conf*100).toFixed(0)}%` } : conf >= 0.6 ? { bg: '#fef9c3', fg: '#854d0e', label: `Medium • ${(conf*100).toFixed(0)}%` } : { bg: '#fee2e2', fg: '#991b1b', label: `Low • ${(conf*100).toFixed(0)}%` };
    const meta = SOURCE_META[normalizeSourceKey(candidate.source)] || { icon: 'fas fa-database', color: '#6b7280' };
    const pubCountRaw = candidate.publication_count ?? candidate.verification_info?.publication_count ?? candidate.verification_info?.document_count ?? null;
    const pubCountIsMin = !!candidate.verification_info?.publication_count_is_min;
    const pubCount = (pubCountRaw != null && !isNaN(pubCountRaw)) ? Number(pubCountRaw) : null;
    const samplePubs = candidate.sample_publications?.slice(0, 2).map(p => `<li class="truncate">• ${p}</li>`).join('') || '<li class="italic text-gray-500">No sample publications available</li>';

    return `<div class="candidate-card p-3 flex gap-3 items-stretch hover:shadow-lg transition-shadow duration-150"><div class="flex flex-col items-center justify-start pt-2 pr-2"><label class="inline-flex items-center cursor-pointer select-none"><input type="checkbox" class="candidate-select h-6 w-6 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition" data-gindex="${index}"></label></div><div class="flex-1 flex flex-col justify-between"><div><div class="flex flex-wrap items-center gap-2 mb-1"><span class="font-semibold text-gray-900 text-base">${formatName(candidate.name)}</span><span class="badge" style="background:${confMeta.bg};color:${confMeta.fg}">${confMeta.label}</span>${candidate.profile_url ? `<a href="${candidate.profile_url}" target="_blank" class="ml-1 text-blue-600 hover:text-blue-800 text-xs" title="Open profile"><i class="fas fa-external-link-alt"></i></a>` : ''}</div>${candidate.affiliation ? `<div class="meta text-sm text-gray-600 mb-1">${candidate.affiliation}</div>` : ''}</div><div class="flex flex-wrap items-center gap-4 text-xs text-gray-500 mt-2 mb-2"><span><strong>ID:</strong> <span class="text-gray-700">${candidate.profile_id}</span></span>${pubCount != null ? `<span><strong>Publications:</strong> <span class="text-gray-700">${pubCount}${pubCountIsMin ? '+' : ''}</span></span>` : ''}</div><div class="bg-gray-50 rounded-md px-3 py-2 mt-1"><ul class="sample-pubs text-xs text-gray-700 leading-tight">${samplePubs}</ul></div></div></div>`;
}

// --- Filtering & Sorting ---
function clearFilters() {
    document.getElementById('search-filter').value = '';
    document.getElementById('source-filter').value = 'all';
    document.getElementById('sort-filter').value = 'most-cited';
    document.getElementById('year-min').value = '';
    document.getElementById('year-max').value = '';
    Object.keys(sourceFilterState).forEach(k => sourceFilterState[k] = 0);
    document.querySelectorAll('.source-toggle').forEach(el => updateSourceToggleVisual(el, 0));
    applyFilters();
}

function applyFilters() {
    if (!originalPublications.length) return;
    let filtered = [...originalPublications];

    // Text, Year, and Source filters
    const searchTerm = document.getElementById('search-filter').value.toLowerCase().trim();
    const yearMin = parseInt(document.getElementById('year-min').value) || 0;
    const yearMax = parseInt(document.getElementById('year-max').value) || Infinity;
    
    filtered = filtered.filter(pub => {
        if (searchTerm && !((pub.title || '').toLowerCase().includes(searchTerm) || (pub.authors || []).join(' ').toLowerCase().includes(searchTerm) || (pub.journal || '').toLowerCase().includes(searchTerm))) return false;
        if ((pub.year || 0) < yearMin || (pub.year || 0) > yearMax) return false;
        return true;
    });

    // Source icon filters (override dropdown)
    if (Object.values(sourceFilterState).some(v => v !== 0)) {
        filtered = filtered.filter(pub => Object.entries(sourceFilterState).every(([key, state]) => (state === 1 && pub.coverage[key]) || (state === -1 && !pub.coverage[key]) || state === 0));
    } else { // Fallback to dropdown
        const sourceFilter = document.getElementById('source-filter').value;
        if (sourceFilter === 'all-four') filtered = filtered.filter(pub => Object.values(pub.coverage).filter(Boolean).length === 4);
        else if (sourceFilter === 'multiple') filtered = filtered.filter(pub => (pub.coverage_count || 0) > 1);
        else if (sourceFilter !== 'all') filtered = filtered.filter(pub => pub.coverage[sourceFilter.replace('-', '_')]);
    }

    // Sorting
    const sortFilter = document.getElementById('sort-filter').value;
    filtered.sort((a, b) => {
        switch (sortFilter) {
            case 'newest': return (b.year || 0) - (a.year || 0);
            case 'oldest': return (a.year || 0) - (b.year || 0);
            case 'title': return (a.title || '').localeCompare(b.title || '');
            case 'journal': return (a.journal || '').localeCompare(b.journal || '');
            default: return (b.citations || 0) - (a.citations || 0); // most-cited
        }
    });

    filteredPublications = filtered;
    updateResultsDisplay(filtered);
}

function updateResultsDisplay(publications) {
    const tableBody = document.getElementById('publications-table-body');
    tableBody.innerHTML = '';
    publications.forEach(pub => tableBody.appendChild(createPublicationRow(pub)));
    document.getElementById('filtered-count').textContent = publications.length;
    document.getElementById('total-count').textContent = originalPublications.length;
    document.getElementById('total-publications').textContent = publications.length;
}

// --- Helper & Utility Functions ---
function setLoadingState(isLoading, type) {
    const elements = {
        search: { btn: searchBtn, textEl: searchBtnText, spinner: searchSpinner, progressEl: searchProgress, loadingText: 'Searching...', defaultText: 'Find Publications', progressText: 'Fetching publications…' },
        autofill: { btn: autofillBtn, textEl: autofillBtnText, spinner: autofillSpinner, progressEl: autofillProgress, loadingText: 'Discovering...', defaultText: 'Autofill IDs', progressText: 'Discovering profiles…' }
    };
    const config = elements[type];
    if (!config) return;

    config.btn.disabled = isLoading;
    config.btn.classList.toggle('loading', isLoading);
    config.textEl.textContent = isLoading ? config.loadingText : config.defaultText;
    config.spinner.classList.toggle('hidden', !isLoading);
    toggleProgressDisplay(config.progressEl, isLoading, config.progressText);
    if (isLoading) {
        updateProgressBar(config.progressEl.querySelector('.progress-bar'), 0);
    }
}

function toggleProgressDisplay(progressEl, show, text = '') {
    progressEl.classList.toggle('hidden', !show);
    if (show) {
        const textEl = progressEl.querySelector('[id$="-progress-text"]');
        if (textEl) textEl.textContent = text;
    }
}

function updateProgressBar(el, percent) {
    if (!el) return;
    const p = Math.max(0, Math.min(100, Number(percent) || 0));
    el.style.width = `${p}%`;
    el.parentElement?.setAttribute('aria-valuenow', String(Math.round(p)));
}

function showError(message) {
    document.getElementById('error-text').textContent = message;
    errorMessage.classList.remove('hidden');
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() { errorMessage.classList.add('hidden'); }
function hideResults() { resultsSection.classList.add('hidden'); }
function formatName(name) { return (name || '').trim().replace(/([a-z])([A-Z]\.)/g, '$1 $2').replace(/\.(?=[A-Z])/g, '. ').replace(/\s{2,}/g, ' '); }
function normalizeDoiValue(val) { const s = (val || '').trim().replace(/^https?:\/\/(dx\.)?doi\.org\//i, '').replace(/^doi:\s*/i, '').split(/[?#\s]/)[0].replace(/\s*/g, ''); return /^10\.\d{4,9}\/\S+$/i.test(s) ? s.toLowerCase() : null; }
function normalizeSourceKey(src) { const s = (src || '').toLowerCase(); if (s.includes('google')) return 'google_scholar'; if (s.includes('scopus')) return 'scopus'; if (s.includes('web of science') || s === 'wos') return 'wos'; if (s.includes('orcid')) return 'orcid'; return 'other'; }

function toggleApiKeysSection() {
    const section = document.getElementById('api-keys-section');
    const chevron = document.getElementById('api-keys-chevron');
    const isHidden = section.classList.toggle('hidden');
    chevron.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
}

function initializeYearRange() {
    const years = originalPublications.map(p => p.year).filter(Boolean).sort((a, b) => a - b);
    if (years.length) {
        const [min, max] = [years[0], years[years.length - 1]];
        ['year-min', 'year-max'].forEach((id, i) => {
            const el = document.getElementById(id);
            el.value = [min, max][i];
            el.placeholder = [min, max][i];
        });
    }
}

function initSourceIconFilters() {
    document.querySelectorAll('.source-toggle').forEach(el => {
        const key = el.dataset.source;
        updateSourceToggleVisual(el, sourceFilterState[key] || 0);
        el.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            sourceFilterState[key] = (sourceFilterState[key] === 0) ? 1 : (sourceFilterState[key] === 1 ? -1 : 0);
            updateSourceToggleVisual(el, sourceFilterState[key]);
            applyFilters();
        });
    });
}

function updateSourceToggleVisual(el, state) {
    const key = el.dataset.source;
    const iconWrap = el.querySelector('.icon-wrap');
    const banIcon = el.querySelector('.ban-icon');
    const meta = SOURCE_META[key];
    iconWrap.className = 'icon-wrap w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition';
    banIcon.classList.toggle('hidden', state !== -1);
    if (state === 1) { // include
        iconWrap.style.backgroundColor = meta.color;
        iconWrap.classList.add('text-white', 'ring-2', 'ring-offset-2');
        el.title = `${meta.fullName}: Include (click to exclude)`;
    } else if (state === -1) { // exclude
        iconWrap.style.backgroundColor = '#ffffff';
        iconWrap.classList.add('text-gray-400', 'border', 'border-gray-300', 'opacity-70');
        el.title = `${meta.fullName}: Exclude (click for any)`;
    } else { // any
        iconWrap.style.backgroundColor = '#e5e7eb';
        iconWrap.classList.add('text-gray-500');
        el.title = `${meta.fullName}: Any (click to include)`;
    }
}

function applySelectedProfiles() {
    // Collect IDs for each source (support multiple per source)
    const idMap = { google_scholar: 'google-scholar-id', scopus: 'scopus-id', wos: 'wos-id', orcid: 'orcid-id' };
    const idsBySource = { google_scholar: [], scopus: [], wos: [], orcid: [] };
    // Allow multiple selection for any source, not just one per source
    Object.values(discoveredProfiles.candidates || []).forEach((candidate, idx) => {
        const key = normalizeSourceKey(candidate.source);
        const cb = document.querySelector(`.candidate-select[data-gindex='${idx}']`);
        if (cb && cb.checked) {
            idsBySource[key].push(candidate.profile_id);
        }
    });
    // Set input values, joining multiple IDs with commas for all sources
    Object.entries(idMap).forEach(([key, inputId]) => {
        if (idsBySource[key].length) {
            document.getElementById(inputId).value = idsBySource[key].join(', ');
        }
    });
    showSuccess(`Applied IDs for: ${Object.keys(idsBySource).filter(k => idsBySource[k].length).join(', ')}`);
    closeProfileModal();
}

function selectAllProfiles() {
    discoveryResults.querySelectorAll('.candidate-select:not(:checked)').forEach(cb => {
        cb.checked = true;
        cb.dispatchEvent(new Event('change', { bubbles: true }));
    });
}

function closeProfileModal() { profileModal.classList.add('hidden'); selectedProfiles = {}; }
function clearIdInputs() { ['google-scholar-id', 'scopus-id', 'wos-id', 'orcid-id'].forEach(id => document.getElementById(id).value = ''); }
function updateApplyButtonState() { const hasSelections = Object.keys(selectedProfiles).length > 0; applySelection.disabled = !hasSelections; applySelection.className = hasSelections ? 'px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700' : 'px-4 py-2 bg-gray-300 text-gray-500 rounded-lg cursor-not-allowed'; }
function showRecentSearches() { alert('Recent searches feature coming soon!'); }
function showSettings() { alert('Settings panel coming soon!'); }

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded z-50';
    successDiv.innerHTML = `<div class="flex items-center"><i class="fas fa-check-circle mr-2"></i><span>${message}</span><button onclick="this.parentElement.parentElement.remove()" class="ml-4"><i class="fas fa-times"></i></button></div>`;
    document.body.appendChild(successDiv);
    setTimeout(() => successDiv.remove(), 5000);
}

async function exportToCSV() {
    if (!currentSearchResults) return;
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                publications: filteredPublications.length ? filteredPublications : originalPublications,
                researcher: currentSearchResults.researcher,
                format: 'csv'
            })
        });
        if (!response.ok) throw new Error('Export failed');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `publications_${new Date().toISOString().slice(0,10)}.csv`;
        if (contentDisposition) {
            const match = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(contentDisposition);
            if (match?.[1]) filename = match[1].replace(/['"]/g, '');
        }
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Export error:', error);
        showError('Export failed. Please try again.');
    }
}