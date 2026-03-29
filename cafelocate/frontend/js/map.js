// CafeSuitabilityAnalysis - Map Module
// Handles Leaflet map initialization, markers, and user interactions

class MapManager {
    constructor() {
        this.storageKey = 'cafesuitabilityanalysis_map_state';
        this.historyStoragePrefix = 'cafesuitabilityanalysis_history_';
        this.map = null;
        this.marker = null;
        this.circle = null;
        this.cafeMarkers = [];
        this.selectedLocation = null;
        this.selectedCafeType = null;
        this.analysisRadius = 500;
        this.initialized = false;
        this.lastAnalysisData = null;
        this.lastAmenitiesReport = null;
        this.lastPopulationData = null;
        this.lastPopulationDensity = null;
    }

    init() {
        // Guard: only initialize once, and only when the map div is visible
        const mapEl = document.getElementById('map');
        if (!mapEl || this.initialized) return;

        // Check map page is active
        const mapPage = document.getElementById('map-page');
        if (!mapPage || !mapPage.classList.contains('active')) return;

        this.initialized = true;
        this.initializeMap();
        this.setupEventListeners();
        this.restorePersistedState();
        this.loadDatasetStats();
        this.updateHistoryVisibility();
    }

    initializeMap() {
        const mapEl = document.getElementById('map');
        if (!mapEl) return;

        // Destroy existing map instance if any (safety)
        if (this.map) {
            this.map.remove();
            this.map = null;
        }

        this.map = L.map('map').setView([27.7172, 85.3240], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19,
        }).addTo(this.map);

        const kathmanduBounds = L.latLngBounds(
            [27.55, 85.10],
            [27.90, 85.55]
        );
        this.map.setMaxBounds(kathmanduBounds);
    }

    setupEventListeners() {
        if (!this.map) return;

        this.map.on('click', async (e) => {
            await this.handleMapClick(e.latlng);
        });

        const cafeTypeSelect = document.getElementById('cafe-type-select');
        if (cafeTypeSelect) {
            cafeTypeSelect.addEventListener('change', (e) => {
                this.selectedCafeType = e.target.value;
                this.persistState();
                if (this.selectedLocation && this.selectedCafeType) {
                    this.analyzeLocation();
                }
            });
        }

        const radiusSlider = document.getElementById('radius-slider');
        const radiusValue = document.getElementById('radius-value');
        if (radiusSlider && radiusValue) {
            radiusSlider.addEventListener('input', (e) => {
                this.analysisRadius = parseInt(e.target.value);
                radiusValue.textContent = this.analysisRadius;
                this.persistState();
                if (this.circle && this.selectedLocation) {
                    this.circle.setRadius(this.analysisRadius);
                }
            });

            radiusSlider.addEventListener('change', (e) => {
                if (this.selectedLocation && this.selectedCafeType) {
                    this.analyzeLocation();
                }
            });
        }

        const fullReportBtn = document.getElementById('full-report-btn');
        if (fullReportBtn) {
            fullReportBtn.addEventListener('click', () => {
                this.showFullReport();
            });
        }

        const downloadPdfBtn = document.getElementById('download-pdf-btn');
        if (downloadPdfBtn) {
            downloadPdfBtn.addEventListener('click', () => {
                this.downloadReportAsPdf();
            });
        }

        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.hideFullReport());
        }

        const modal = document.getElementById('full-report-modal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.hideFullReport();
            });
        }

        document.addEventListener('keydown', (e) => {
            const activePage = document.getElementById('map-page');
            const mapPageActive = activePage && activePage.classList.contains('active');
            if (!mapPageActive) return;
            if (e.key === 'Enter' && !e.target.closest('#login-form') && !e.target.closest('#register-form')) {
                e.preventDefault();
            }
        });
    }

    async handleMapClick(latlng) {
        const { lat, lng } = latlng;

        try {
            if (!window.apiManager) {
                throw new Error('API manager not initialized');
            }

            const validation = await window.apiManager.validateLocation(lat, lng);
            if (!validation.is_valid) {
                if (window.uiManager) {
                    window.uiManager.showNotification(validation.message, 'warning');
                }
                return;
            }
        } catch (error) {
            console.error('Location validation failed:', error);
            if (window.uiManager) {
                window.uiManager.showNotification(
                    error.message || 'Unable to validate the selected location.',
                    'error'
                );
            }
            return;
        }

        this.clearMarkerAndCircle();

        this.marker = L.marker([lat, lng]).addTo(this.map);
        this.marker.bindPopup(
            `<b>📍 Selected Location</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`
        ).openPopup();

        this.circle = L.circle([lat, lng], {
            color: '#6c5ce7',
            fillColor: '#6c5ce7',
            fillOpacity: 0.08,
            weight: 2,
            radius: this.analysisRadius
        }).addTo(this.map);

        this.selectedLocation = { lat, lng };
        this.updateCoordinatesDisplay(lat, lng);
        this.persistState();

        if (this.selectedCafeType) {
            this.analyzeLocation();
        } else {
            this.showCafeTypePrompt();
        }
    }

    async analyzeLocation() {
        if (!this.selectedLocation || !this.selectedCafeType) return;

        const { lat, lng } = this.selectedLocation;
        const viewport = this.captureViewport();
        this.showLoading(true);

        try {
            if (!window.apiManager) {
                throw new Error('API manager not initialized');
            }

            const analysisData = await window.apiManager.getSuitabilityAnalysis(
                lat, lng, this.selectedCafeType, this.analysisRadius
            );

            this.lastAnalysisData = analysisData;
            this.storeCurrentAnalysisInHistory(analysisData);
            this.displayAnalysisResults(analysisData);

            if (analysisData.top5) {
                this.displayNearbyCafes(analysisData.top5);
            }

            // Get ML predictions
            try {
                const mlPredictions = await window.apiManager.getMLPrediction(
                    lat, lng, this.selectedCafeType, this.analysisRadius
                );
                this.displayMLPredictions(mlPredictions);
            } catch (mlError) {
                console.warn('ML prediction failed (non-critical):', mlError);
                // Don't fail the whole analysis if ML prediction fails
            }

            this.loadHistoryForCurrentType();

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showAnalysisError(error.message);
        } finally {
            this.showLoading(false);
            this.restoreViewport(viewport);
        }
    }

    displayNearbyCafes(cafes) {
        this.clearCafeMarkers();

        cafes.forEach((cafe, index) => {
            if (!cafe.latitude || !cafe.longitude) return;

            const marker = L.circleMarker([cafe.latitude, cafe.longitude], {
                radius: 8,
                fillColor: '#fdcb6e',
                color: '#e17055',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            // Enhanced popup with rating and review data
            const ratingSection = cafe.rating ? `
                <div style="margin: 8px 0; padding: 8px 0; border-top: 1px solid #eee;">
                    <div><strong>${cafe.star_rating || cafe.rating}</strong></div>
                    <small style="color: #666;">${cafe.rating_category || ''}</small><br>
                    <small style="color: #888;">${cafe.review_summary || `${cafe.review_count || 0} reviews`}</small>
                </div>
            ` : '<div style="color: #999;"><small>Not yet rated</small></div>';

            const cafeTypeFormatted = (cafe.cafe_type || '').replace(/_/g, ' ');
            
            marker.bindPopup(`
                <div style="min-width: 200px; font-family: Arial, sans-serif;">
                    <b style="font-size: 14px;">${cafe.name}</b><br>
                    <small style="color: #666;">Type: ${cafeTypeFormatted}</small>
                    ${ratingSection}
                </div>
            `, {
                maxWidth: 250,
                className: 'cafe-popup'
            });

            this.cafeMarkers.push(marker);
        });
    }

    displayAnalysisResults(data) {
        const suitability = data.suitability || {};
        const prediction = data.prediction || {};

        // Suitability score
        const scoreEl = document.getElementById('suitability-score');
        const levelEl = document.getElementById('suitability-level');
        const numericScore = Number(suitability.score);
        if (scoreEl) {
            scoreEl.textContent = Number.isFinite(numericScore) ? numericScore.toFixed(2) : '-';
            this.updateScoreCircle(Number.isFinite(numericScore) ? numericScore : 0);
        }
        if (levelEl) {
            levelEl.textContent = suitability.level || prediction.predicted_suitability || '-';
        }

        // Model Predictions
        const rfScoreEl = document.getElementById('rf-score');
        const xgbScoreEl = document.getElementById('xgb-score');
        const ensembleScoreEl = document.getElementById('ensemble-score');

        if (rfScoreEl) {
            const rfScore = prediction.random_forest_score;
            rfScoreEl.textContent = rfScore !== null && rfScore !== undefined ? Number(rfScore).toFixed(2) : '-';
        }
        if (xgbScoreEl) {
            const xgbScore = prediction.xgboost_score;
            xgbScoreEl.textContent = xgbScore !== null && xgbScore !== undefined ? Number(xgbScore).toFixed(2) : '-';
        }
        if (ensembleScoreEl) {
            const ensembleScore = prediction.ensemble_score;
            ensembleScoreEl.textContent = ensembleScore !== null && ensembleScore !== undefined ? Number(ensembleScore).toFixed(2) : '-';
        }

        // ML Prediction card
        const predictionTypeEl = document.querySelector('#prediction-card .prediction-type');
        const predictionConfEl = document.querySelector('#prediction-card .prediction-confidence');
        if (predictionTypeEl) {
            predictionTypeEl.textContent =
                prediction.recommended_cafe_type ||
                'No recommendation';
        }
        if (predictionConfEl) {
            predictionConfEl.textContent = '';
        }

        // Top 5 cafes
        const top5List = document.getElementById('top5-list');
        if (top5List && data.top5) {
            if (data.top5.length === 0) {
                top5List.innerHTML = '<p class="no-data">No cafes found in this area</p>';
            } else {
                top5List.innerHTML = data.top5.map((cafe, idx) => {
                    const ratingHtml = cafe.rating ? `
                        <div class="cafe-rating">
                            <span class="rating-stars">${cafe.star_rating || cafe.rating}</span>
                            <span class="rating-category">${cafe.rating_category || ''}</span>
                        </div>
                    ` : '<div class="cafe-rating"><span class="no-rating">Not rated</span></div>';
                    
                    const reviewsInfo = `<small class="review-info">${cafe.review_summary || `${cafe.review_count || 0} reviews`}</small>`;
                    
                    return `
                    <div class="cafe-item">
                        <div class="cafe-info">
                            <div class="cafe-rank">#${idx + 1}</div>
                            <div class="cafe-details">
                                <div class="cafe-name">${cafe.name}</div>
                                <div class="cafe-type">${(cafe.cafe_type || '').replace(/_/g, ' ')}</div>
                                ${ratingHtml}
                                ${reviewsInfo}
                            </div>
                        </div>
                    </div>
                `;
                }).join('');
            }
        }

        // Metrics - conditionally show based on actual values
        const competitorEl = document.getElementById('competitor-count');
        const roadEl = document.getElementById('road-length');
        const popEl = document.getElementById('population-density');

        const competitorValue = suitability.competitor_count ?? data.nearby_count;
        const roadValue = suitability.road_distance_m != null ? suitability.road_distance_m : suitability.road_length_m;
        const popValue = suitability.population_density;

        // Hide/show metric boxes based on whether they have values
        if (competitorEl) {
            const metricBox = competitorEl.closest('.metric');
            if (competitorValue != null) {
                competitorEl.textContent = competitorValue;
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }

        if (roadEl) {
            const metricBox = roadEl.closest('.metric');
            if (roadValue != null) {
                roadEl.textContent = roadValue + 'm';
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }

        if (popEl) {
            const metricBox = popEl.closest('.metric');
            if (popValue != null) {
                this.lastPopulationDensity = popValue;
                popEl.textContent = Number(popValue).toLocaleString('en-US', { maximumFractionDigits: 1 }) + '/km²';
                if (metricBox) metricBox.style.display = 'block';
            } else {
                if (metricBox) metricBox.style.display = 'none';
            }
        }

        // Display demographic information
        this.displayDemographicInfo(data.demographics);
    }

    displayDemographicInfo(demographics) {
        if (!demographics) return;

        const demographicEl = document.getElementById('demographic-info');
        if (!demographicEl) return;

        const html = `
            <div class="demographic-section">
                <h3>📊 Local Demographics</h3>
                <div class="demographic-grid">
                    <div class="demographic-item">
                        <span class="demographic-label">Ward #</span>
                        <span class="demographic-value">${demographics.ward_number || '-'}</span>
                    </div>
                    <div class="demographic-item">
                        <span class="demographic-label">Population</span>
                        <span class="demographic-value">${demographics.population ? demographics.population.toLocaleString() : '-'}</span>
                    </div>
                    <div class="demographic-item">
                        <span class="demographic-label">Avg Household Size</span>
                        <span class="demographic-value">${demographics.average_household_size || '-'} persons</span>
                    </div>
                    <div class="demographic-item">
                        <span class="demographic-label">Area</span>
                        <span class="demographic-value">${demographics.area_sqkm || '-'} km²</span>
                    </div>
                    <div class="demographic-item">
                        <span class="demographic-label">Density Category</span>
                        <span class="demographic-value demographic-category">${demographics.density_category || '-'}</span>
                    </div>
                    <div class="demographic-item">
                        <span class="demographic-label">Market Potential</span>
                        <span class="demographic-value demographic-market">${demographics.market_potential || '-'}</span>
                    </div>
                </div>
            </div>
        `;

        demographicEl.innerHTML = html;
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        if (!scoreCircle) return;

        const normalizedScore = Math.max(0, Math.min(10, Number(score) || 0));
        const fillPercent = normalizedScore * 10;
        let color = '#00b894'; // green
        if (normalizedScore < 4) color = '#e17055';     // red
        else if (normalizedScore < 7) color = '#fdcb6e'; // yellow

        scoreCircle.style.background = `conic-gradient(${color} 0% ${fillPercent}%, #e9ecef ${fillPercent}% 100%)`;
    }

    displayMLPredictions(data) {
        if (!data || !data.predictions) return;

        const mlSection = document.getElementById('ml-predictions-section');
        if (!mlSection) return;

        const { random_forest, xgboost, ensemble } = data.predictions;

        // Show the ML predictions section
        mlSection.style.display = 'block';

        // Random Forest
        const rfScore = document.getElementById('ml-rf-score');
        const rfInfo = document.getElementById('ml-rf-info');
        if (rfScore && random_forest) {
            rfScore.textContent = Number(random_forest.score || 0).toFixed(2);
            const r2 = data.model_info?.random_forest_r2 || '-';
            if (rfInfo) rfInfo.textContent = `R² = ${r2.toFixed ? r2.toFixed(4) : r2} (Regression accuracy)`;
        }

        // XGBoost
        const xgbTier = document.getElementById('ml-xgb-tier');
        const xgbConf = document.getElementById('ml-xgb-confidence');
        const xgbInfo = document.getElementById('ml-xgb-info');
        if (xgbTier && xgboost) {
            xgbTier.textContent = xgboost.tier || '-';
            const tierColor = {
                'High': '#00b894',
                'Medium': '#fdcb6e',
                'Low': '#e17055'
            }[xgboost.tier] || '#999';
            xgbTier.style.color = tierColor;
            
            if (xgbConf) {
                const confPercent = Math.round((xgboost.confidence || 0) * 100);
                xgbConf.textContent = confPercent;
            }

            const accuracy = data.model_info?.xgboost_accuracy || '-';
            if (xgbInfo) xgbInfo.textContent = `Confidence: ${(xgboost.confidence * 100).toFixed(1)}%`;
        }

        // Ensemble
        const ensembleEmoji = document.getElementById('ml-ensemble-emoji');
        const ensembleTier = document.getElementById('ml-ensemble-tier');
        const ensembleRec = document.getElementById('ml-ensemble-rec');
        if (ensembleTier && ensemble) {
            ensembleTier.textContent = ensemble.tier || '-';
            
            if (ensembleEmoji) {
                ensembleEmoji.textContent = ensemble.emoji || '🟡';
            }

            if (ensembleRec) {
                ensembleRec.textContent = ensemble.recommendation || 'No recommendation';
            }
        }

        console.log('ML Predictions displayed:', data);
    }

    updateCoordinatesDisplay(lat, lng) {
        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) {
            coordsEl.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
    }

    showCafeTypePrompt() {
        const selectEl = document.getElementById('cafe-type-select');
        if (selectEl) {
            selectEl.style.borderColor = '#e17055';
            selectEl.style.boxShadow = '0 0 0 3px rgba(231,112,85,0.2)';
            selectEl.focus();
            setTimeout(() => {
                selectEl.style.borderColor = '';
                selectEl.style.boxShadow = '';
            }, 3000);
        }

        if (window.uiManager) {
            window.uiManager.showNotification('Please select a cafe type first!', 'warning');
        }
    }

    showLoading(show) {
        const loadingEl = document.getElementById('loading-indicator');
        if (loadingEl) {
            loadingEl.style.display = show ? 'block' : 'none';
        }
    }

    showAnalysisError(message) {
        const top5List = document.getElementById('top5-list');
        if (top5List) {
            top5List.innerHTML = `<p class="no-data" style="color:#e17055">Error: ${message || 'Analysis failed. Is the backend running?'}</p>`;
        }
        if (window.uiManager) {
            window.uiManager.showNotification('Analysis failed. Check backend connection.', 'error');
        }
    }

    async loadDatasetStats() {
        if (!window.apiManager) return;

        const statsEl = document.getElementById('dataset-stats');
        if (statsEl) {
            statsEl.innerHTML = '<p>Loading dataset insights...</p>';
        }

        try {
            const stats = await window.apiManager.getCafeDatasetStats();
            this.displayDatasetStats(stats);
        } catch (error) {
            console.warn('Unable to fetch dataset stats', error);
            if (statsEl) {
                statsEl.innerHTML = '<p style="color:#e17055">Unable to load dataset stats.</p>';
            }
        }
    }

    displayDatasetStats(stats) {
        const statsEl = document.getElementById('dataset-stats');
        if (!statsEl) return;

        if (!stats || typeof stats !== 'object') {
            statsEl.innerHTML = '<p style="color:#e17055">Dataset stats unavailable.</p>';
            return;
        }

        const typeList = stats.top_type_ranking || [];
        statsEl.innerHTML = `
            <div class="dataset-card-grid">
                <div class="dataset-card"><strong>Total Cafes:</strong><br>${this.formatNumber(stats.total_cafes || 0)}</div>
                <div class="dataset-card"><strong>Open Cafes:</strong><br>${this.formatNumber(stats.open_cafes || 0)}</div>
                <div class="dataset-card"><strong>Avg Rating:</strong><br>${stats.avg_rating !== null ? stats.avg_rating.toFixed(2) : 'N/A'}</div>
                <div class="dataset-card"><strong>Avg Reviews:</strong><br>${this.formatNumber(stats.avg_review_count || 0)}</div>
            </div>
            <div class="dataset-type-ranking">
                <strong>Top Cafe Types:</strong>
                <ul>${typeList.map(item => `<li>${item.type} — ${item.count}</li>`).join('')}</ul>
            </div>
        `;
    }

    formatNumber(value) {
        return Number(value || 0).toLocaleString('en-US');
    }

    persistState() {
        try {
            const state = {
                selectedLocation: this.selectedLocation,
                selectedCafeType: this.selectedCafeType,
                analysisRadius: this.analysisRadius,
            };
            sessionStorage.setItem(this.storageKey, JSON.stringify(state));
        } catch (_) {}
    }

    restorePersistedState() {
        try {
            const raw = sessionStorage.getItem(this.storageKey);
            if (!raw) return;

            const state = JSON.parse(raw);
            if (state.selectedCafeType) {
                this.selectedCafeType = state.selectedCafeType;
                const cafeTypeSelect = document.getElementById('cafe-type-select');
                if (cafeTypeSelect) {
                    cafeTypeSelect.value = state.selectedCafeType;
                }
            }

            if (state.analysisRadius) {
                this.analysisRadius = parseInt(state.analysisRadius);
                const radiusSlider = document.getElementById('radius-slider');
                const radiusValue = document.getElementById('radius-value');
                if (radiusSlider) radiusSlider.value = this.analysisRadius;
                if (radiusValue) radiusValue.textContent = this.analysisRadius;
            }

            if (state.selectedLocation && typeof state.selectedLocation.lat === 'number' && typeof state.selectedLocation.lng === 'number') {
                const { lat, lng } = state.selectedLocation;
                this.selectedLocation = { lat, lng };
                this.clearMarkerAndCircle();
                this.marker = L.marker([lat, lng]).addTo(this.map);
                this.circle = L.circle([lat, lng], {
                    color: '#6c5ce7',
                    fillColor: '#6c5ce7',
                    fillOpacity: 0.08,
                    weight: 2,
                    radius: this.analysisRadius
                }).addTo(this.map);
                this.updateCoordinatesDisplay(lat, lng);
            }
        } catch (_) {}
    }

    captureViewport() {
        if (!this.map) return null;
        return {
            center: this.map.getCenter(),
            zoom: this.map.getZoom(),
        };
    }

    restoreViewport(viewport) {
        if (!this.map || !viewport) return;

        requestAnimationFrame(() => {
            this.map.invalidateSize(false);
            this.map.setView(viewport.center, viewport.zoom, { animate: false });
        });
    }

    updateHistoryVisibility() {
        const historySection = document.getElementById('history-section');
        if (!historySection) return;

        const isLoggedIn = !!(window.authManager && window.authManager.isAuthenticated());
        historySection.style.display = isLoggedIn ? 'block' : 'none';

        if (!isLoggedIn) {
            const historyList = document.getElementById('history-list');
            if (historyList) {
                historyList.innerHTML = '<p class="no-data">Log in and analyze locations to build your cafe history.</p>';
            }
        }
    }

    getHistoryStorageKey() {
        const user = window.authManager && window.authManager.getUser();
        if (!user || !user.id) return null;
        return `${this.historyStoragePrefix}${user.id}`;
    }

    getStoredHistory() {
        const key = this.getHistoryStorageKey();
        if (!key) return [];

        try {
            const raw = localStorage.getItem(key);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (_) {
            return [];
        }
    }

    setStoredHistory(items) {
        const key = this.getHistoryStorageKey();
        if (!key) return;

        try {
            localStorage.setItem(key, JSON.stringify(items));
        } catch (_) {}
    }

    storeCurrentAnalysisInHistory(analysisData) {
        if (!(window.authManager && window.authManager.isAuthenticated()) || !this.selectedLocation || !this.selectedCafeType) {
            return;
        }

        const score = Number(analysisData?.suitability?.score);
        if (!Number.isFinite(score)) return;

        const currentItem = {
            latitude: this.selectedLocation.lat,
            longitude: this.selectedLocation.lng,
            cafe_type: this.selectedCafeType,
            radius: this.analysisRadius,
            suitability_score: score,
            suitability_level: analysisData?.suitability?.level || '-',
            created_at: new Date().toISOString(),
        };

        const existing = this.getStoredHistory();
        const isDuplicate = existing.some(item => {
            return (
                item.cafe_type === currentItem.cafe_type &&
                Number(item.radius) === Number(currentItem.radius) &&
                Math.abs(Number(item.latitude) - currentItem.latitude) < 0.000001 &&
                Math.abs(Number(item.longitude) - currentItem.longitude) < 0.000001 &&
                Math.abs(Number(item.suitability_score) - currentItem.suitability_score) < 0.01
            );
        });

        if (isDuplicate) return;

        const updated = [currentItem, ...existing].slice(0, 30);
        this.setStoredHistory(updated);
    }

    async loadHistoryForCurrentType() {
        this.updateHistoryVisibility();

        if (!(window.authManager && window.authManager.isAuthenticated()) || !this.selectedCafeType) {
            return;
        }

        const historyList = document.getElementById('history-list');
        if (historyList) {
            historyList.innerHTML = '<p class="no-data">Loading your saved locations...</p>';
        }

        const historyItems = this.getStoredHistory()
            .filter(item => item.cafe_type === this.selectedCafeType)
            .slice(0, 8);

        this.renderHistorySection(historyItems);
    }

    renderHistorySection(historyItems) {
        const historyList = document.getElementById('history-list');
        if (!historyList) return;

        const currentScore = Number(this.lastAnalysisData?.suitability?.score || 0);
        const currentLevel = this.lastAnalysisData?.suitability?.level || '-';
        const currentLocation = this.selectedLocation;
        const currentRadius = this.analysisRadius;
        const comparableItems = (historyItems || []).filter(item => {
            if (!currentLocation) return true;

            const sameLat = Math.abs(Number(item.latitude) - currentLocation.lat) < 0.000001;
            const sameLng = Math.abs(Number(item.longitude) - currentLocation.lng) < 0.000001;
            const sameRadius = Number(item.radius) === Number(currentRadius);
            const sameScore = Math.abs(Number(item.suitability_score) - currentScore) < 0.01;

            return !(sameLat && sameLng && sameRadius && sameScore);
        });

        const currentSummary = currentLocation ? `
            <div class="history-item">
                <strong>Current Pin</strong>
                <div class="history-meta">
                    ${currentLocation.lat.toFixed(5)}, ${currentLocation.lng.toFixed(5)} • ${currentRadius}m • ${this.formatCafeType(this.selectedCafeType)}
                </div>
                <div class="history-compare">
                    Score: ${currentScore.toFixed(2)} (${currentLevel})
                </div>
            </div>
        ` : '';

        if (comparableItems.length === 0) {
            historyList.innerHTML = `
                ${currentSummary}
                <p class="no-data">This is your first saved location for this cafe type. Pin another location to compare.</p>
            `;
            return;
        }

        historyList.innerHTML = currentSummary + comparableItems.map(item => {
            const previousScore = Number(item.suitability_score || 0);
            const diff = currentScore - previousScore;
            const diffLabel = diff === 0
                ? 'Same score as current pin'
                : `${diff > 0 ? '+' : ''}${diff.toFixed(2)} vs current pin`;

            const dateLabel = new Date(item.created_at).toLocaleString();
            return `
                <div class="history-item">
                    <strong>${this.formatCafeType(item.cafe_type)}</strong>
                    <div class="history-meta">
                        ${Number(item.latitude).toFixed(5)}, ${Number(item.longitude).toFixed(5)} • ${item.radius}m • ${dateLabel}
                    </div>
                    <div class="history-compare">
                        Previous: ${previousScore.toFixed(2)} (${item.suitability_level})<br>
                        Current: ${currentScore.toFixed(2)} (${currentLevel})<br>
                        ${diffLabel}
                    </div>
                </div>
            `;
        }).join('');

        this.restoreViewport(this.captureViewport());
    }

    formatCafeType(cafeType) {
        return (cafeType || '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, char => char.toUpperCase());
    }

    clearMarkerAndCircle() {
        if (this.marker) { this.map.removeLayer(this.marker); this.marker = null; }
        if (this.circle) { this.map.removeLayer(this.circle); this.circle = null; }
    }

    clearCafeMarkers() {
        this.cafeMarkers.forEach(m => this.map.removeLayer(m));
        this.cafeMarkers = [];
    }

    clearMap() {
        if (this.map) {
            this.clearMarkerAndCircle();
            this.clearCafeMarkers();
        }
        this.selectedLocation = null;
        this.selectedCafeType = null;
        this.lastAnalysisData = null;
        this.clearResultsDisplay();
        this.updateHistoryVisibility();
        sessionStorage.removeItem(this.storageKey);
    }

    clearResultsDisplay() {
        const scoreEl = document.getElementById('suitability-score');
        if (scoreEl) scoreEl.textContent = '-';
        const levelEl = document.getElementById('suitability-level');
        if (levelEl) levelEl.textContent = '-';

        const predType = document.querySelector('#prediction-card .prediction-type');
        const predConf = document.querySelector('#prediction-card .prediction-confidence');
        if (predType) predType.textContent = '-';
        if (predConf) predConf.textContent = '-';

        const top5List = document.getElementById('top5-list');
        if (top5List) top5List.innerHTML = '<p class="no-data">Click on the map to see results</p>';

        // Hide all metric boxes and reset stored values
        ['competitor-count', 'road-length', 'population-density'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = '-';
                const metricBox = el.closest('.metric');
                if (metricBox) metricBox.style.display = 'none';
            }
        });
        this.lastPopulationDensity = null;

        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) coordsEl.textContent = '';

        const historyList = document.getElementById('history-list');
        if (historyList && window.authManager && window.authManager.isAuthenticated()) {
            historyList.innerHTML = '<p class="no-data">Analyze a location to compare it with your saved history.</p>';
        }
    }

    showFullReport() {
        const modal = document.getElementById('full-report-modal');
        const reportContent = document.getElementById('report-content');
        if (!modal || !reportContent || !this.selectedLocation) {
            if (window.uiManager) {
                window.uiManager.showNotification('Pin a location first to generate a report.', 'warning');
            }
            return;
        }

        // Show loading state
        reportContent.innerHTML = '<div style="padding: 20px; text-align: center;"><p>⏳ Loading detailed report...</p></div>';
        modal.style.display = 'block';

        // Fetch amenities and population data
        this.fetchReportData().then(() => {
            reportContent.innerHTML = this.generateFullReport();
            // Attach event handlers for "See more" buttons after content is inserted
            this.attachAmenityHandlers();
        }).catch(error => {
            console.error('Error fetching report data:', error);
            reportContent.innerHTML = '<div style="padding: 20px; color: #e17055;"><p>⚠️ Error loading report data. Please try again.</p></div>';
        });
    }

    attachAmenityHandlers() {
        const buttons = document.querySelectorAll('.see-more-amenities');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = btn.getAttribute('data-type');
                this.showFullAmenityList(type, btn);
            });
        });
    }

    showFullAmenityList(type, triggerButton) {
        if (!this.lastAmenitiesReport) return;
        const report = this.lastAmenitiesReport.amenities_report || {};
        const data = report[type];
        if (!data) return;

        // Build full list HTML with extra details (name, distance if available)
        const itemsHtml = data.amenities.map(a => {
            const name = a.name || 'Unnamed';
            const details = [];
            if (a.amenity_type) details.push(a.amenity_type);
            if (a.latitude && a.longitude) details.push(`${a.latitude.toFixed(6)}, ${a.longitude.toFixed(6)}`);
            if (a.distance != null) details.push(`${Math.round(a.distance)} m`);
            return `<li style="margin:6px 0"><strong>${name}</strong>${details.length?` — <span style="color:#666">${details.join(' · ')}</span>`:''}</li>`;
        }).join('');

        // Replace the parent <ul> content where the button was located
        const li = triggerButton.closest('li');
        if (li) {
            const parentUl = li.parentElement;
            if (parentUl) {
                parentUl.innerHTML = itemsHtml + `<li style="margin-top:8px"><button class="collapse-amenities" style="background:none;border:none;color:#0984e3;cursor:pointer;padding:0">Show less</button></li>`;
                const collapseBtn = parentUl.querySelector('.collapse-amenities');
                if (collapseBtn) collapseBtn.addEventListener('click', () => {
                    // Re-render the report to restore original truncated view
                    const reportContent = document.getElementById('report-content');
                    if (reportContent) reportContent.innerHTML = this.generateFullReport();
                    this.attachAmenityHandlers();
                });
            }
        }
    }

    async fetchReportData() {
        if (!this.selectedLocation) return;

        const { lat, lng } = this.selectedLocation;

        try {
            // Fetch amenities report (schools, hospitals, bus stops, cafes)
            const amenitiesResponse = await fetch(
                `http://localhost:8000/api/amenities-report/`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat: lat,
                        lng: lng,
                        radius: this.analysisRadius
                    })
                }
            );

            if (amenitiesResponse.ok) {
                this.lastAmenitiesReport = await amenitiesResponse.json();
            }

            // Fetch area population
            const populationResponse = await fetch(
                `http://localhost:8000/api/area-population/?lat=${lat}&lng=${lng}&radius=${this.analysisRadius}`
            );

            if (populationResponse.ok) {
                this.lastPopulationData = await populationResponse.json();
            }
        } catch (error) {
            console.error('Error fetching report data:', error);
            this.lastAmenitiesReport = null;
            this.lastPopulationData = null;
        }
    }

    hideFullReport() {
        const modal = document.getElementById('full-report-modal');
        if (modal) modal.style.display = 'none';
    }

    generateFullReport() {
        if (!this.selectedLocation) return '<p>No location selected.</p>';

        const { lat, lng } = this.selectedLocation;
        const score = document.getElementById('suitability-score')?.textContent || '-';
        const competitors = document.getElementById('competitor-count')?.textContent || '-';
        const populationDensitySidebar = document.getElementById('population-density')?.textContent || '-';
        const cafeTypeFormatted = this.selectedCafeType
            ? this.selectedCafeType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            : 'Not selected';

        const prediction = this.lastAnalysisData?.prediction || {};
        // Get amenities report and population data
        const amenitiesReport = this.lastAmenitiesReport?.amenities_report || {};
        const populationData = this.lastPopulationData || {};

        // Use the stored population density from the sidebar (API calculated value)
        let formattedDensityString = '-';
        if (this.lastPopulationDensity != null) {
            formattedDensityString = Number(this.lastPopulationDensity).toLocaleString('en-US', { maximumFractionDigits: 1 });
        }

        return `
            <div class="report-section">
                <h3>📍 Location Details</h3>
                <div class="report-grid">
                    <div class="report-item"><strong>Coordinates:</strong><br>${lat.toFixed(6)}, ${lng.toFixed(6)}</div>
                    <div class="report-item"><strong>Analysis Radius:</strong><br>${this.analysisRadius} meters</div>
                    <div class="report-item"><strong>Cafe Type:</strong><br>${cafeTypeFormatted}</div>
                </div>
            </div>

            <div class="report-section">
                <h3>📊 Suitability Analysis</h3>
                <div class="report-grid">
                    <div class="report-item"><strong>Overall Score:</strong><br>${score} / 10</div>
                    <div class="report-item"><strong>Competitors Nearby:</strong><br>${competitors}</div>
                    <div class="report-item"><strong>Population Density:</strong><br>${formattedDensityString}/km²</div>
                    <div class="report-item"><strong>Total Population (in radius):</strong><br>${populationData && populationData.total_population ? Number(populationData.total_population).toLocaleString() : '0'}</div>
                </div>
            </div>

            ${populationData.total_population ? `
            <div class="report-section">
                <h3>👥 Population in Selected Area</h3>
                <div class="report-grid">
                    <div class="report-item">
                        <strong>Total Population:</strong><br>
                        <span style="font-size: 1.4em; color: #6c5ce7; font-weight: bold;">
                            ${Number(populationData.total_population).toLocaleString()}
                        </span>
                    </div>
                    <div class="report-item">
                        <strong>Affected Wards:</strong><br>
                        ${populationData.affected_ward_count || 0} ward${populationData.affected_ward_count !== 1 ? 's' : ''}
                    </div>
                </div>
                ${populationData.affected_wards && populationData.affected_wards.length > 0 ? `
                <div style="margin-top: 15px; padding: 10px; background: #f5f6fa; border-radius: 5px;">
                    <strong>Ward Details:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        ${populationData.affected_wards.map(ward => `
                            <li>Ward ${ward.ward_number}: ${Number(ward.population).toLocaleString()} population, ${ward.population_density.toFixed(0)}/km²</li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
            ` : ''}

            ${Object.keys(amenitiesReport).length > 0 ? `
            <div class="report-section">
                <h3>🏘️ Amenities in Selected Area</h3>
                <div class="report-grid">
                    ${Object.entries(amenitiesReport).map(([type, data]) => `
                        <div class="report-item">
                            <strong>${type.replace(/_/g, ' ').toUpperCase()}:</strong><br>
                            <span style="font-size: 1.3em; color: #00b894;">${data.count}</span>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: #f5f6fa; border-radius: 5px;">
                    <strong>Amenity Listings:</strong>
                    ${Object.entries(amenitiesReport).map(([type, data]) => {
                        if (data.count === 0) return '';
                        return `
                            <div style="margin: 10px 0;">
                                <strong>${type.replace(/_/g, ' ').toUpperCase()} (${data.count}):</strong>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                                    ${data.amenities.slice(0, 5).map(amenity => `
                                        <li>${amenity.name || 'Unnamed'}</li>
                                    `).join('')}
                                            ${data.count > 5 ? `<li><button class="see-more-amenities" data-type="${type}" style="background:none;border:none;color:#0984e3;cursor:pointer;padding:0;margin:0">See more (${data.count - 5})</button></li>` : ''}
                                </ul>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            ` : ''}

            <div class="report-insights">
                <h4>💡 Key Insights & Recommendations</h4>
                <ul>
                    <li><strong>Location Strength:</strong> ${this._getLocationStrength(parseInt(score))}</li>
                    <li><strong>Competition Level:</strong> ${this._getCompetitionLevel(competitors)}</li>
                    <li><strong>Market Potential:</strong> ${this._getMarketPotential(formattedDensityString)}</li>
                    <li><strong>Recommendation:</strong> ${parseFloat(score) >= 7
                        ? 'This location shows good potential for a cafe business.'
                        : parseFloat(score) >= 4
                            ? 'This location is workable, but competition and access should be reviewed carefully.'
                            : 'Consider alternative locations with less competition or better road access.'}</li>
                </ul>
            </div>
        `;
    }

    downloadReportAsPdf() {
        if (!this.selectedLocation) {
            if (window.uiManager) {
                window.uiManager.showNotification('Pin a location first to generate a report.', 'warning');
            }
            return;
        }

        // Generate the report HTML
        const reportHtml = this.generateFullReport();
        
        // Create a temporary container for PDF generation
        const element = document.createElement('div');
        element.style.padding = '20px';
        element.style.backgroundColor = '#fff';
        element.innerHTML = `
            <h1>CafeSuitabilityAnalysis - Location Analysis Report</h1>
            <p style="color: #666; margin-bottom: 20px;">Generated on ${new Date().toLocaleDateString()}</p>
            ${reportHtml}
        `;

        // Configure PDF generation options
        const opt = {
            margin: 10,
            filename: 'cafesuitabilityanalysis-report.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
        };

        // Generate and download PDF
        html2pdf().set(opt).from(element).save();
    }

    _getLocationStrength(score) {
        if (score >= 7) return "Excellent - high success potential";
        if (score >= 4) return "Good - moderate success potential";
        if (score >= 2) return "Fair - consider improvements";
        return "Poor - high risk, explore alternatives";
    }

    _getCompetitionLevel(count) {
        const n = parseInt(count) || 0;
        if (n < 5) return "Low competition - great opportunity";
        if (n < 15) return "Moderate competition - viable market";
        return "High competition - saturated market";
    }

    _getMarketPotential(population) {
        const density = parseInt((population || '').replace(/[^0-9]/g, '')) || 0;
        if (density > 15000) return "High population density - strong market";
        if (density > 8000) return "Moderate density - decent market";
        return "Low density – limited foot traffic expected";
    }
}

// Do NOT auto-init here — init() is called by authManager.showMapPage()
document.addEventListener('DOMContentLoaded', () => {
    window.mapManager = new MapManager();
});
